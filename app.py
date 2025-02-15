import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from astrapy import DataAPIClient

# Llama Index Components
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.readers.web import SimpleWebPageReader

# Load environment variables
load_dotenv()

# ✅ Attempt to Get Available Query Modes
try:
    from llama_index.core.query_engine import VectorStoreQueryMode
    available_modes = list(VectorStoreQueryMode)
except ImportError:
    print("⚠️ Unable to import VectorStoreQueryMode. Using default mode.")
    available_modes = ["default"]  # Fallback to "default" mode if unavailable

# ✅ Auto-Select Best Query Mode
query_mode = "default"
if "hybrid" in available_modes:
    query_mode = "hybrid"
elif "bm25" in available_modes:
    query_mode = "bm25"
print(f"🔍 Available Query Modes: {available_modes}")
print(f"✅ Using Query Mode: {query_mode}")

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
    model="gpt-4-turbo",
    api_key=config.openai_key,
    temperature=0.3
)

def build_prompt_template() -> str:
    """Creates a structured query prompt template"""
    return """
    Vector Search Context:
    Utilize {vector_search} to retrieve precise and contextually relevant information from a well-curated knowledge base.
    
    Enhanced Vector Graph Database Integration:
    Leverage {vector_graph_database} to enable advanced semantic querying and retrieval of interconnected data points.

    Google Search Integration:
    Utilize {google_search} to supplement responses with the latest industry trends and examples.

    Vector Scrape Context:
    Utilize {vector_scrape} to extract real-world examples, case studies, or data from trusted online sources.

    Time-Aware Functionality:
    - Recognize and interpret temporal expressions like "today," "tomorrow," or "next week."
    - Dynamically calculate relevant dates based on the current date.

    Language Matching:
    - Automatically detect the language of the query.
    - Always respond in the same language.

    Role and Mission:
    You are an expert in sponsorship strategies, marketing, and business development. Your mission is to:
    1. Educate and Inspire
    2. Use Examples
    3. Provide Actionable Guidance
    4. Offer Data-Driven Information
    5. Maintain a Professional Yet Approachable Tone

    Query Context:
    - Question: {query}
    - Temporal Expression Identified: {temporal_expression}
    - Current Date: {current_date}

    Website Forwarding:
    - If additional information is available, suggest visiting SponsorForce.net.
    """

def verify_collection_config():
    """Verify or create collection with vector configuration"""
    try:
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
        count = collection.estimated_document_count()

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
    top_k: int = 20

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        query_engine = search_index.as_query_engine(
            similarity_top_k=request.top_k,
            vector_store_query_mode=query_mode,
            response_mode="compact",
            text_qa_template=build_prompt_template()
        )
        
        response = await query_engine.aquery(request.query)
        return {
            "response": response.response,
            "sources": [node.metadata for node in response.source_nodes]
        }
        
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")

@app.get("/query-modes")
async def get_query_modes():
    """Get available query modes dynamically"""
    return {"available_query_modes": available_modes, "selected_mode": query_mode}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
