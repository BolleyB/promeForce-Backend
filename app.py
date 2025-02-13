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

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    model="gpt-3.5-turbo",
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
        
        found_phrases = [
            phrase for phrase in time_map.keys()
            if phrase in query.lower()
        ]
        
        return {
            'temporal_expression': found_phrases[0] if found_phrases else None,
            'reference_date': current_date.strftime('%Y-%m-%d'),
            'target_date': time_map.get(found_phrases[0], current_date).strftime('%Y-%m-%d')
        }

class LanguageHandler:
    @staticmethod
    def detect_language(query: str) -> str:
        try:
            return detect(query)
        except LangDetectException:
            return 'en'

def build_prompt_template(lang: str = "en") -> str:
    base_template = """
    As a sponsorship strategy expert, provide:
    1. Time-aware insights for {time_context}
    2. {lang}-specific response
    3. Actionable steps with metrics
    4. Relevant case studies from our database
    
    Current Date: {current_date}
    Query: {query}
    
    Response Guidelines:
    - Structure with clear sections
    - Include 3-5 implementation steps
    - Reference latest industry trends
    - Add website redirect at end
    """
    return base_template.format(
        time_context="{time_context}",
        lang=lang,
        current_date=datetime.now().strftime('%Y-%m-%d'),
        query="{query}"
    )

def create_query_engine():
    return VectorStoreIndex.from_vector_store(
        astra_vector_store,
        embed_model=embed_model
    ).as_query_engine(
        similarity_top_k=15,
        vector_store_query_mode="hybrid",
        response_mode="compact",
        node_postprocessors=[TimeWeightedPostprocessor()],
        text_qa_template=build_prompt_template()
    )

def validate_document(doc):
    return len(doc.text) > 100 and 'date' in doc.metadata

async def initialize_documents():
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
        
        return VectorStoreIndex(
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
            print("🆕 Initializing new collection")
            search_index = await initialize_documents()
        else:
            print(f"🔍 Found {count} existing documents")
            search_index = create_query_engine()
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

class QueryRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = None
    top_k: int = 15

@app.middleware("http")
async def validate_query(request: Request, call_next):
    if request.url.path == "/query":
        try:
            body = await request.json()
            if len(body.get('query', '')) < 3:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Query must be at least 3 characters"}
                )
        except:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request format"}
            )
    return await call_next(request)

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        # Process temporal context
        time_ctx = TemporalProcessor.parse_time_phrases(request.query)
        
        # Detect language
        lang = LanguageHandler.detect_language(request.query)
        
        # Build filters
        filters = {}
        if time_ctx['target_date']:
            filters = {"date": {"$gte": time_ctx['target_date']}}
        
        # Execute query
        response = await search_index.aquery(
            request.query,
            filters=filters,
            similarity_top_k=request.top_k
        )
        
        # Format response
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        collection = db.get_collection(db_config.collection)
        count = collection.count_documents({})
        return {
            "status": "healthy",
            "documents": count['status']['count'],
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.getenv("SSL_KEY_PATH"),
        ssl_certfile=os.getenv("SSL_CERT_PATH")
    )