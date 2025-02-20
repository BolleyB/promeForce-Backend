import os
import asyncio
import logging
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import httpx
from astrapy import DataAPIClient
from astrapy.exceptions import TooManyDocumentsToCountException
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from starlette.requests import Request

# Try importing redis.asyncio, with fallback if not available
try:
    import redis.asyncio as redis
except ImportError:
    logging.warning("Redis not installed; caching disabled.")
    redis = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Redis setup with fallback
redis_client = None
if redis:
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDISHOST", "localhost"),
            port=int(os.getenv("REDISPORT", "6379")),
            username=os.getenv("REDISUSER", ""),
            password=os.getenv("REDISPASSWORD", ""),
            decode_responses=True
        )
    except Exception as e:
        logging.error(f"Failed to connect to Redis: {e}")
        redis_client = None

# Global Role and Mission Prompt
ROLE_AND_MISSION_PROMPT = """
You are an expert in sponsorship strategies, marketing, and business development. Your role is to provide high-quality, informative responses that help users understand and navigate these fields effectively.

When crafting your response, consider the following guidelines:

1. **Educate and Inspire:** Offer in-depth insights and practical advice that leave the user feeling more informed and confident.

2. **Use Examples:** Incorporate real-world examples, industry trends, or case studies to make the information more relatable and engaging.

3. **Provide Actionable Guidance:** When appropriate, offer step-by-step instructions, best practices, or actionable recommendations that users can implement.

4. **Include Data:** Use data, statistics, or references to back up your statements and establish credibility.

5. **Maintain a Professional Yet Approachable Tone:** Communicate in a way that is both authoritative and easy to understand.

In terms of response structure, aim to present the information in a clear and organized manner that best suits the user's query. You can use various formatting techniques such as headings, bullet points, numbered lists, and paragraphs to make the response more readable and engaging.

For example, if the user asks for a brief overview, provide a concise summary. If the user requests a step-by-step guide, format the response accordingly.

Always ensure that your response is well-formatted and easy to follow, using appropriate formatting elements to highlight important information.
"""

class AstraDBConfig(BaseModel):
    endpoint: str = os.getenv("ASTRA_DB_ENDPOINT")
    token: str = os.getenv("ASTRA_DB_TOKEN")
    collection: str = "sf_data"
    embedding_dim: int = 1536
    namespace: str = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

class APIConfig(BaseModel):
    openai_key: str = os.getenv("OPENAI_API_KEY")
    searchapi_key: str = os.getenv("SEARCHAPI_KEY")
    newsapi_key: str = os.getenv("NEWSAPI_KEY")

app = FastAPI(title="SponsorForce AI Backend")
config = APIConfig()
db_config = AstraDBConfig()

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = DataAPIClient(db_config.token)
db = client.get_database(api_endpoint=db_config.endpoint, namespace=db_config.namespace)

try:
    astra_vector_store = AstraDBVectorStore(
        token=db_config.token,
        api_endpoint=db_config.endpoint,
        collection_name=db_config.collection,
        embedding_dimension=db_config.embedding_dim,
        namespace=db_config.namespace,
    )
    print("✅ Astra DB connection established")
except Exception as e:
    print(f"❌ Astra DB connection failed: {e}")
    raise

embed_model = OpenAIEmbedding(api_key=config.openai_key, model_name="text-embedding-3-small")
llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=config.openai_key, temperature=0.3)
multi_modal_llm = OpenAIMultiModal(model="gpt-4-vision-preview", api_key=config.openai_key)

def verify_collection_config():
    try:
        collection_names = [c.name for c in db.list_collections()]
        if db_config.collection not in collection_names:
            print(f"🆕 Creating collection {db_config.collection}")
            db.create_collection(
                name=db_config.collection,
                options={"vector": {"dimension": db_config.embedding_dim, "metric": "cosine"}}
            )
            print(f"✅ Collection {db_config.collection} created")
        if "sessions" not in collection_names:
            db.create_collection("sessions")
            print("✅ Created sessions collection")
        print(f"🔍 Collection {db_config.collection} exists")
        collection = db.get_collection(db_config.collection)
        print("✅ Collection verification successful")
    except Exception as e:
        print(f"❌ Collection verification failed: {e}")
        raise

def create_index_from_existing() -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        vector_store=astra_vector_store,
        embed_model=embed_model,
        storage_context=StorageContext.from_defaults(vector_store=astra_vector_store)
    )

async def initialize_documents() -> VectorStoreIndex:
    try:
        web_reader = SimpleWebPageReader()
        local_reader = SimpleDirectoryReader("./data/")
        web_docs, local_docs = await asyncio.gather(
            web_reader.load_data_async([
                "https://www.sponsorforce.net/#/portal/home",
                "https://www.sponsorforce.net/#/portal/topics",
                "https://www.sponsorforce.net/#/portal/resource"
            ]),
            local_reader.load_data_async()
        )
        return VectorStoreIndex(
            documents=[*web_docs, *local_docs],
            storage_context=StorageContext.from_defaults(vector_store=astra_vector_store),
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

async def handle_sports_query(query: str) -> Dict:
    search_url = f"https://www.searchapi.io/api/v1/search?engine=google&q={query}&api_key={config.searchapi_key}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(search_url)
            response.raise_for_status()
            search_results = response.json()
            items = search_results.get("organic_results", [])
            results = []
            for item in items[:5]:
                results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet") or item.get("description"),
                    "link": item.get("link")
                })
            return {"response": results}
        except httpx.HTTPStatusError as e:
            logging.error(f"SearchAPI error: {e.response.status_code}, Details: {str(e)}")
            return {"error": f"SearchAPI failed with status {e.response.status_code}"}

async def perform_deep_search(query: str) -> Dict:
    async with httpx.AsyncClient() as client:
        search_task = client.get(f"https://www.searchapi.io/api/v1/search?engine=google&q={query}&api_key={config.searchapi_key}")
        news_task = client.get(f"https://newsapi.org/v2/everything?q={query}&apiKey={config.newsapi_key}")
        
        search_resp, news_resp = await asyncio.gather(search_task, news_task)
        
        synthesized_response = "## Deep Analysis\n\n"
        sources = []
        
        if search_resp.status_code == 200:
            organic_results = search_resp.json().get("organic_results", [])
            synthesized_response += "### Web Results\n"
            for idx, item in enumerate(organic_results[:3], 1):
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No description")
                link = item.get("link", "No link")
                synthesized_response += f"#### Source {idx}: {title}\n- **Summary:** {snippet}\n- **Link:** {link}\n\n"
                sources.append({"title": title, "link": link})
        
        if news_resp.status_code == 200:
            articles = news_resp.json().get("articles", [])
            synthesized_response += "### News Results\n"
            for idx, article in enumerate(articles[:2], 1):
                title = article.get("title", "No title")
                description = article.get("description", "No description")
                url = article.get("url", "No url")
                synthesized_response += f"#### Source {idx}: {title}\n- **Summary:** {description}\n- **Link:** {url}\n\n"
                sources.append({"title": title, "link": url})
        
        synthesized_response += "## Synthesis\nThis analysis combines web and news insights for a comprehensive overview."
        return {"response": synthesized_response, "sources": sources}

def detect_intent(query: str) -> str:
    query_lower = query.lower()
    if "overview" in query_lower:
        return "compact"
    elif "step-by-step" in query_lower or "guide" in query_lower:
        return "tree_summarize"
    return "refine"

def create_custom_query_engine(index: VectorStoreIndex, similarity_top_k: int = 12, response_mode: str = "refine") -> RetrieverQueryEngine:
    retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
    response_synthesizer = get_response_synthesizer(response_mode=response_mode)
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

class QueryRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = None
    top_k: int = 5
    deep_search: bool = False
    session_id: str = None

@app.post("/query")
@limiter.limit("10/minute")
async def handle_query(request: Request, query_request: QueryRequest, user_id: str = Header(default=None)):
    try:
        cache_key = f"query:{query_request.query}:{query_request.top_k}:{query_request.deep_search}:{query_request.session_id}"
        cached_response = await redis_client.get(cache_key) if redis_client else None
        if cached_response:
            return {"response": cached_response, "sources": []}

        query_lower = query_request.query.lower()
        if any(keyword in query_lower for keyword in ["live scores", "fixture", "match", "score", "result"]):
            sports_response = await handle_sports_query(query_request.query)
            if sports_response:
                if redis_client:
                    await redis_client.setex(cache_key, 3600, str(sports_response["response"]))
                return sports_response

        context = ""
        if query_request.session_id:
            session_collection = db.get_collection("sessions")
            past_queries = await session_collection.find({"session_id": query_request.session_id}).to_list(length=5)
            context = "Previous queries:\n" + "\n".join(q["query"] for q in past_queries) + "\n\n"
            await session_collection.insert_one({"session_id": query_request.session_id, "query": query_request.query, "timestamp": int(time.time())})

        final_query = f"{context}{ROLE_AND_MISSION_PROMPT}\n\n{query_request.query}"
        response_mode = detect_intent(query_request.query)
        
        if query_request.deep_search:
            deep_task = perform_deep_search(query_request.query)
            query_engine = create_custom_query_engine(index=search_index, similarity_top_k=query_request.top_k, response_mode=response_mode)
            vector_task = query_engine.aquery(final_query)
            deep_response, vector_response = await asyncio.gather(deep_task, vector_task)
            if "error" not in deep_response:
                deep_response["response"] += f"\n\n### Additional Insights from Internal Data\n{vector_response.response}"
                deep_response["sources"].extend([node.metadata for node in vector_response.source_nodes])
            if redis_client:
                await redis_client.setex(cache_key, 3600, deep_response["response"])
            return deep_response
        else:
            query_engine = create_custom_query_engine(index=search_index, similarity_top_k=query_request.top_k, response_mode=response_mode)
            response = await query_engine.aquery(final_query)
            formatted_response = {
                "response": f"## Response\n\n{response.response}",
                "sources": [node.metadata for node in response.source_nodes]
            }
            if redis_client:
                await redis_client.setex(cache_key, 3600, formatted_response["response"])
            return formatted_response
    except Exception as e:
        logging.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")

@app.get("/collection-info")
async def get_collection_info():
    try:
        collection = db.get_collection(db_config.collection)
        count_result = collection.estimated_document_count()
        count = count_result["status"]["count"] if isinstance(count_result, dict) else count_result
        sample = collection.find_one({})["data"]["document"]
        return {"total_documents": count, "sample_document": sample}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DocumentUpdate(BaseModel):
    documents: List[Dict]

@app.post("/update-documents")
async def update_documents(update: DocumentUpdate):
    try:
        for doc in update.documents:
            text = doc["text"]
            metadata = doc.get("metadata", {})
            embedding = embed_model.get_text_embedding(text)
            astra_vector_store.add([embedding], documents=[text], metadatas=[metadata])
        global search_index
        search_index = create_index_from_existing()
        if redis_client:
            await redis_client.flushdb()  # Clear cache on update
        return {"message": f"Added {len(update.documents)} documents and updated the index"}
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
        return {"status": "unhealthy", "error": str(e)}

@app.post("/query-with-image")
async def query_with_image(query: str = Form(...), image: UploadFile = File(...)):
    try:
        image_content = await image.read()
        response = multi_modal_llm.complete(
            prompt=query,
            image_documents=[SimpleDirectoryReader.load_image(image_content)]
        )
        return {"response": response.text}
    except Exception as e:
        logging.error(f"Image query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.getenv("SSL_KEY_PATH"),
        ssl_certfile=os.getenv("SSL_CERT_PATH")
    )