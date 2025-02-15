import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import httpx
from astrapy import DataAPIClient

# Llama Index & Database Components
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
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

# ✅ Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Astra DB with DataAPIClient
client = DataAPIClient(db_config.token)
db = client.get_database(
    api_endpoint=db_config.endpoint,
    namespace=db_config.namespace
)

# Initialize Astra DB with enhanced configuration
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

# Initialize AI models
embed_model = OpenAIEmbedding(
    api_key=config.openai_key,
    model_name="text-embedding-3-small"
)

llm = LlamaOpenAI(
    model="gpt-4",
    api_key=config.openai_key,
    temperature=0.3
)

def verify_collection_config():
    """Verify or create collection with vector configuration"""
    try:
        # Check if collection exists
        collection_names = [c.name for c in db.list_collections()]
        
        if db_config.collection not in collection_names:
            print(f"🆕 Creating collection {db_config.collection}")
            db.create_collection(
                name=db_config.collection,
                options={
                    "vector": {
                        "dimension": db_config.embedding_dim,
                        "metric": "cosine"
                    }
                }
            )
            print(f"✅ Collection {db_config.collection} created")
        else:
            print(f"🔍 Collection {db_config.collection} exists")

        # Verify collection can be accessed
        collection = db.get_collection(db_config.collection)
        print(f"✅ Collection verification successful")
        
    except Exception as e:
        print(f"❌ Collection verification failed: {e}")
        raise



def create_index_from_existing() -> VectorStoreIndex:
    """Create index from existing vector store"""
    return VectorStoreIndex.from_vector_store(
        vector_store=astra_vector_store,
        embed_model=embed_model,
        storage_context=StorageContext.from_defaults(vector_store=astra_vector_store)
    )

async def initialize_documents() -> VectorStoreIndex:
    """Initialize document sources and create vector index"""
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
        
        return VectorStoreIndex(
            documents=[*web_docs, *local_docs],
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
    verify_collection_config()
    
    try:
        collection = db.get_collection(db_config.collection)
        count_result = collection.estimated_document_count()
        count = count_result["status"]["count"] if isinstance(count_result, dict) else count_result
        
        if count == 0:
            print("🆕 Initializing new collection with documents")
            search_index = await initialize_documents()
        else:
            print(f"🔍 Found existing collection with {count} documents")
            search_index = create_index_from_existing()
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

class QueryRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = None
    top_k: int = 5

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        # Handle sports-related queries first
        query_lower = request.query.lower()
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

        # Process general queries
        query_engine = search_index.as_query_engine(
            similarity_top_k=10,
            vector_store_query_mode="sparse_hybrid",
            response_mode="compact"
        )
        
        response = await query_engine.aquery(request.query)
        return {
            "response": response.response,
            "sources": [node.metadata for node in response.source_nodes]
        }
        
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")

@app.get("/collection-info")
async def get_collection_info():
    """Validate collection contents"""
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
    """Handle incremental document updates"""
    try:
        index = create_index_from_existing()
        index.insert(update.documents)
        return {"message": f"Added {len(update.documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_sports_data(url: str) -> Dict:
    """Generic sports data fetcher"""
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