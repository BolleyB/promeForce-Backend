import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from astrapy import DataAPIClient
from langdetect import detect, LangDetectException

# Llama Index Components
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.postprocessor import TimeWeightedPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.readers.web import SimpleWebPageReader

# Load environment variables
load_dotenv()

class AstraDBConfig(BaseModel):
    endpoint: str = os.getenv("ASTRA_DB_ENDPOINT")
    token: str = os.getenv("ASTRA_DB_TOKEN")
    collection: str = "sf_data"
    embedding_dim: int = 1536
    namespace: str = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

class APIConfig(BaseModel):
    openai_key: str = os.getenv("OPENAI_API_KEY")
    serpapi_key: str = os.getenv("SERPAPI_KEY")
    sportsdb_key: str = os.getenv("SPORTSDB_KEY", "392246")

app = FastAPI(title="SponsorForce AI Backend")
config = APIConfig()
db_config = AstraDBConfig()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize Astra DB
client = DataAPIClient(db_config.token)
db = client.get_database(
    api_endpoint=db_config.endpoint,
    namespace=db_config.namespace
)

try:
    astra_vector_store = AstraDBVectorStore(
        token=db_config.token,
        api_endpoint=db_config.endpoint,
        collection_name=db_config.collection,
        embedding_dimension=db_config.embedding_dim,
        namespace=db_config.namespace
    )
    print("✅ Astra DB connection established")
except Exception as e:
    print(f"❌ Astra DB connection failed: {e}")
    raise

# AI Models Configuration
embed_model = OpenAIEmbedding(
    api_key=config.openai_key,
    model_name="text-embedding-3-small"
)

llm = LlamaOpenAI(
    model="gpt-4-turbo",
    api_key=config.openai_key,
    temperature=0.3
)

class TemporalProcessor:
    @staticmethod
    def parse_time_phrases(query: str) -> Dict:
        current_date = datetime.now()
        time_map = {
            'today': current_date,
            'tomorrow': current_date + timedelta(days=1),
            'this weekend': current_date + timedelta(
                days=(5 - current_date.weekday()) % 7  # Next Saturday
            ),
            'next week': current_date + timedelta(weeks=1)
        }
        
        found_phrases = [phrase for phrase in time_map.keys() if phrase in query.lower()]
        temporal_expression = found_phrases[0] if found_phrases else None
        target_date_value = time_map.get(temporal_expression, current_date)
        
        return {
            'temporal_expression': temporal_expression,
            'reference_date': current_date.strftime('%Y-%m-%d'),
            'target_date': target_date_value.strftime('%Y-%m-%d')
        }


class LanguageHandler:
    @staticmethod
    def detect_language(query: str) -> str:
        try:
            return detect(query)
        except LangDetectException:
            return 'en'

def build_custom_prompt(lang: str, time_context: dict) -> str:
    return f"""
    As a sponsorship strategy expert, respond with:
    
    1. Time Awareness ({time_context['temporal_expression'] or 'N/A'}):
    - Focus on period between {time_context['reference_date']} and {time_context['target_date']}
    - Update outdated temporal references in context
    
    2. Language Matching:
    - Respond in {lang}
    - Use industry terms in {lang}
    
    3. Actionable Content:
    [Implementation Steps]
    1. Practical first step with timeline
    2. Measurement metrics
    3. Risk mitigation
    
    [Supporting Data]
    - Market trends 2023-2024
    - ROI projections
    - Case studies from database
    
    4. Website Redirection:
    Always include: "Explore more at SponsorForce.net"
    
    Query: {{query}}
    """

def create_query_engine(lang: str, time_context: dict):
    return VectorStoreIndex.from_vector_store(
        astra_vector_store,
        embed_model=embed_model
    ).as_query_engine(
        similarity_top_k=15,
        vector_store_query_mode="default",
        response_mode="compact",
        node_postprocessors=[
            TimeWeightedPostprocessor(
                time_decay=0.7,
                time_access_refresh=True
            )
        ],
        text_qa_template=build_custom_prompt(lang, time_context)
    )

def validate_document(doc):
    return len(doc.text) > 100 and 'date' in doc.metadata

async def initialize_documents() -> VectorStoreIndex:
    try:
        web_reader = SimpleWebPageReader()
        local_reader = SimpleDirectoryReader("./data/")
        
        web_docs, local_docs = await asyncio.gather(
            web_reader.load_data_async([
                "https://www.sponsorforce.net/#/portal/home",
                "https://www.sponsorforce.net/#/portal/topic",
                "https://www.sponsorforce.net/#/portal/resource"
            ]),
            local_reader.load_data_async()
        )
        
        valid_docs = [
            doc for doc in [*web_docs, *local_docs]
            if validate_document(doc)
        ]
        
        return VectorStoreIndex.from_documents(
            documents=valid_docs,
            storage_context=StorageContext.from_defaults(
                vector_store=astra_vector_store
            ),
            embed_model=embed_model
        )
    except Exception as e:
        print(f"❌ Document initialization failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    global search_index
    try:
        collection = db.get_collection(db_config.collection)
        count = collection.estimated_document_count()
        
        if count == 0:
            print("🆕 Initializing new collection with documents")
            search_index = await initialize_documents()
        else:
            print(f"🔍 Found {count} existing documents")
            search_index = VectorStoreIndex.from_vector_store(
                astra_vector_store,
                embed_model=embed_model
            )
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

class QueryRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = None
    top_k: int = 15

@app.middleware("http")
async def validate_query(request: Request, call_next):
    if request.url.path == "/query" and request.method == "POST":
        try:
            body = await request.json()
            if len(body.get('query', '')) < 3:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Query must be at least 3 characters"},
                    headers={"Access-Control-Allow-Origin": "*"}
                )
        except:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request format"},
                headers={"Access-Control-Allow-Origin": "*"}
            )
    return await call_next(request)

@app.options("/query")
async def options_query():
    return JSONResponse(
        content={"message": "Preflight request accepted"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "600"
        }
    )

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        query_lower = request.query.lower()
        # Handle sports queries
        if "live scores" in query_lower:
            url = f"https://www.thesportsdb.com/api/v2/json/{config.sportsdb_key}/livescore/soccer"
            return await fetch_sports_data(url)
            
        if "upcoming fixtures" in query_lower:
            team_name = query_lower.split("for")[-1].strip()
            team_url = f"https://www.thesportsdb.com/api/v1/json/{config.sportsdb_key}/searchteams.php?t={team_name}"
            team_data = await fetch_sports_data(team_url)
            
            if not team_data.get("teams"):
                return {"response": f"No team found: {team_name}"}
                
            team_id = team_data["teams"][0]["idTeam"]
            fixtures_url = f"https://www.thesportsdb.com/api/v1/json/{config.sportsdb_key}/eventsnext.php?id={team_id}"
            return await fetch_sports_data(fixtures_url)

        # Process sponsorship queries
        time_ctx = TemporalProcessor.parse_time_phrases(request.query)
        lang = LanguageHandler.detect_language(request.query)
        
        query_engine = create_query_engine(lang, time_ctx)
        
        response = await query_engine.aquery(
            request.query,
            similarity_top_k=request.top_k
        )
        
        return {
            "response": response.response,
            "metadata": {
                "language": lang,
                "temporal_context": time_ctx,
                "sources": [
                    {**node.metadata, "score": node.score}
                    for node in response.source_nodes
                ],
                "confidence": sum(n.score for n in response.source_nodes)/len(response.source_nodes)
            },
            "website_redirect": "https://www.sponsorforce.net/#/portal/resource"
        }
        
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"Access-Control-Allow-Origin": "*"}
        )


@app.get("/collection-info")
async def get_collection_info():
    try:
        collection = db.get_collection(db_config.collection)
        count_result = collection.count_documents({})
        count = count_result["status"]["count"] if isinstance(count_result, dict) else count_result
        sample = collection.find_one({})["data"]["document"]
        
        return {
            "total_documents": count,
            "sample_document": sample
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DocumentUpdate(BaseModel):
    documents: List[Dict]

@app.post("/update-documents")
async def update_documents(update: DocumentUpdate):
    try:
        index = VectorStoreIndex.from_vector_store(
            astra_vector_store,
            embed_model=embed_model
        )
        index.insert(update.documents)
        return {"message": f"Added {len(update.documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_sports_data(url: str) -> Dict:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"Sports API error: {e.response.status_code}")
            return {"error": str(e)}

@app.get("/health")
async def health_check():
    try:
        collection_names = [c.name for c in db.list_collections()]
        return {
            "status": "healthy",
            "database": "connected",
            "collections": len(collection_names),
            "collection_names": collection_names
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.getenv("SSL_KEY_PATH"),
        ssl_certfile=os.getenv("SSL_CERT_PATH")
    )