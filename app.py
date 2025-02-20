import os
import asyncio
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

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

While a typical response might include an introduction, main points, action steps, and a conclusion, feel free to adapt this structure as needed based on the specific query and the information you're conveying.

For example, if the user asks for a brief overview, you can provide a concise summary without detailed steps. If the user requests a step-by-step guide, you can format the response accordingly.

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

app = FastAPI(title="SponsorForce AI Backend")
config = APIConfig()
db_config = AstraDBConfig()

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
        else:
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
                "https://www.sponsorforce.net/#/portal/topics",  # Fixed typo from 'topic'
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
    search_url = f"https://www.searchapi.io/api/v1/search?engine=google&q={query}&api_key={config.searchapi_key}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(search_url)
            response.raise_for_status()
            search_results = response.json()
            organic_results = search_results.get("organic_results", [])
            synthesized_response = "Here’s a deep analysis based on the latest available data:\n\n"
            sources = []
            for idx, item in enumerate(organic_results[:5], 1):
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No description")
                link = item.get("link", "No link")
                synthesized_response += f"### Source {idx}: {title}\n"
                synthesized_response += f"- **Summary:** {snippet}\n"
                synthesized_response += f"- **Link:** {link}\n\n"
                sources.append({"title": title, "link": link})
            synthesized_response += "### Synthesis\n"
            synthesized_response += "Based on the gathered data, this analysis provides a comprehensive overview of the topic. For further details, refer to the cited sources."
            return {"response": synthesized_response, "sources": sources}
        except httpx.HTTPStatusError as e:
            logging.error(f"Deep search failed: {e.response.status_code}, Details: {str(e)}")
            return {"error": f"Deep search failed with status {e.response.status_code}"}

def create_custom_query_engine(index: VectorStoreIndex, similarity_top_k: int = 12, response_mode: str = "refine") -> RetrieverQueryEngine:
    retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
    response_synthesizer = get_response_synthesizer(response_mode=response_mode)
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

class QueryRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = None
    top_k: int = 5
    deep_search: bool = False  # Added deep_search flag

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        query_lower = request.query.lower()
        if any(keyword in query_lower for keyword in ["live scores", "fixture", "match", "score", "result"]):
            sports_response = await handle_sports_query(request.query)
            if sports_response:
                return sports_response
        final_query = ROLE_AND_MISSION_PROMPT + "\n\n" + request.query
        if request.deep_search:
            deep_response = await perform_deep_search(request.query)
            if "error" not in deep_response:
                query_engine = create_custom_query_engine(
                    index=search_index,
                    similarity_top_k=request.top_k,
                    response_mode="tree_summarize"
                )
                vector_response = await query_engine.aquery(final_query)
                deep_response["response"] += f"\n\n### Additional Insights from Internal Data\n{vector_response.response}"
                deep_response["sources"].extend([node.metadata for node in vector_response.source_nodes])
            return deep_response
        else:
            query_engine = create_custom_query_engine(
                index=search_index,
                similarity_top_k=request.top_k,
                response_mode="tree_summarize"
            )
            response = await query_engine.aquery(final_query)
            return {
                "response": response.response,
                "sources": [node.metadata for node in response.source_nodes]
            }
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.getenv("SSL_KEY_PATH"),
        ssl_certfile=os.getenv("SSL_CERT_PATH")
    )