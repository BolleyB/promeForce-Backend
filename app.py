import os
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import httpx
import tweepy
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
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Updated Role and Mission Prompt (broadened for sports)
ROLE_AND_MISSION_PROMPT = """
You are an expert in sponsorship strategies, marketing, business development, and sports (including football, basketball, cricket, tennis, and motorsport). Your role is to provide high-quality, informative responses that empower users to understand and excel in these fields, delivering precise, specific answers tailored to the query.

When crafting your response, adhere to these guidelines:

1. **Educate and Inspire:** Provide in-depth insights and practical advice that boost the user’s knowledge and confidence. Always use specific, named examples (e.g., “Nike’s $1.5B NFL deal” or “Arsenal’s 7-1 win over PSV”) instead of vague placeholders like “[Company X]” or “[Player A].”
2. **Leverage Real-World Context:** Incorporate current examples, industry trends, or case studies to make responses engaging and relevant. For time-sensitive queries (e.g., “latest headlines”), draw from real-time sources like web articles, X posts, or official updates, ensuring data aligns with the specified timeframe (e.g., last 24 hours).
3. **Provide Actionable Guidance:** Include clear, step-by-step instructions, best practices, or recommendations by default, unless a brief overview is requested. Ensure advice is practical and implementable.
4. **Back Up with Data:** Support claims with credible data, statistics, or references (e.g., “Per Statista 2024, 65% of brands increased sponsorship budgets” or “X post by @FutballNews_, March 4, 2025”). Include timestamps for real-time sources and note the current date (e.g., March 5, 2025).
5. **Maintain a Professional Yet Approachable Tone:** Communicate authoritatively in an accessible, conversational style, avoiding jargon unless explained.

**Response Structure:**
- Use a clear, organized format: an introduction (context or timeframe, e.g., “Last 24 hours as of March 5, 2025”), main points (details with examples), actionable steps (if applicable), and a conclusion (summary or next steps).
- Enhance readability with headings, bullet points, numbered lists, or bold text to highlight key details.
- Adapt depth to the query: concise summaries for overviews, detailed guides with specifics for “how-to” or “list” requests.

**Additional Instructions:**
- For time-sensitive queries, filter data to the exact timeframe (e.g., last 24 hours from March 4, 07:24 AM PST to March 5, 07:24 AM PST) and fetch real-time insights from X, the web, or sports databases.
- Prioritize current, verifiable information over hypothetical or outdated examples. If specifics are unavailable, estimate based on trends and flag assumptions (e.g., “Likely £40M based on 2024 market rates”).
- Cite sources clearly (e.g., “Per @YohaigNG, March 4, 2025”) to build credibility.

Ensure responses are well-formatted, specific, and leave the user with clear, actionable takeaways or solutions, regardless of the topic.
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
    sportsdb_key: str = os.getenv("SPORTSDB_API_KEY")
    x_bearer_token: str = os.getenv("X_BEARER_TOKEN")

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

async def update_sports_data():
    try:
        web_reader = SimpleWebPageReader()
        sports_docs = await web_reader.load_data_async([
            "https://www.bbc.com/sport/football",
            "https://www.skysports.com/premier-league-news"
        ])
        embeddings = [embed_model.get_text_embedding(doc.text) for doc in sports_docs]
        astra_vector_store.add(embeddings, documents=[d.text for d in sports_docs], metadatas=[d.metadata for d in sports_docs])
        global search_index
        search_index = create_index_from_existing()
        print("✅ Sports data updated in vector store")
    except Exception as e:
        print(f"❌ Sports data update failed: {e}")

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
        
        # Start background sports data updates
        scheduler = AsyncIOScheduler()
        scheduler.add_job(update_sports_data, "interval", hours=6)
        scheduler.start()
        print("✅ Background scheduler started")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

async def fetch_sports_data(query: str, timeframe: str = None) -> Dict:
    base_url = f"https://www.thesportsdb.com/api/v1/json/{config.sportsdb_key}"
    async with httpx.AsyncClient() as client:
        try:
            if "fixture" in query.lower() or "match" in query.lower():
                date_match = re.search(r"March \d{1,2}, 2025", query)
                date = date_match.group().replace("March ", "").replace(", 2025", "") if date_match else "04"
                url = f"{base_url}/eventsday.php?d=2025-03-{date}"
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                if data["events"]:
                    return {"response": [{"team1": e["strHomeTeam"], "team2": e["strAwayTeam"], "time": e["strTime"]} for e in data["events"]]}
            elif "standing" in query.lower():
                url = f"{base_url}/lookuptable.php?l=4328&s=2024-2025"  # Premier League
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                if data["table"]:
                    return {"response": [{"team": t["name"], "points": t["points"]} for t in data["table"][:10]]}
            url = f"https://www.searchapi.io/api/v1/search?engine=google&q={query}&api_key={config.searchapi_key}&tbs=qdr:{timeframe[0] if timeframe else 'h'}"
            response = await client.get(url)
            response.raise_for_status()
            search_results = response.json()
            return {"response": [{"title": r["title"], "snippet": r["snippet"], "link": r["link"]} for r in search_results.get("organic_results", [])[:5]]}
        except httpx.HTTPStatusError as e:
            logging.error(f"Sports API error: {e.response.status_code}")
            return {"error": str(e)}

async def fetch_x_posts(query: str, timeframe_hours: int = 24) -> List[Dict]:
    try:
        auth = tweepy.OAuth2BearerHandler(config.x_bearer_token)
        api = tweepy.API(auth)
        since_time = (datetime.now() - timedelta(hours=timeframe_hours)).strftime("%Y-%m-%d")
        tweets = api.search_tweets(q=query, count=10, since=since_time, tweet_mode="extended")
        return [{"text": t.full_text, "user": t.user.screen_name, "created_at": str(t.created_at)} for t in tweets]
    except Exception as e:
        logging.error(f"X fetch error: {e}")
        return []

async def perform_deep_search(query: str, timeframe: str = None) -> Dict:
    search_url = f"https://www.searchapi.io/api/v1/search?engine=google&q={query}&api_key={config.searchapi_key}&tbs=qdr:{timeframe[0] if timeframe else 'h'}"
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
            synthesized_response += "This analysis compiles the most relevant insights from recent sources."
            return {"response": synthesized_response, "sources": sources}
        except httpx.HTTPStatusError as e:
            logging.error(f"Deep search failed: {e.response.status_code}")
            return {"error": f"Deep search failed with status {e.response.status_code}"}

def create_custom_query_engine(index: VectorStoreIndex, similarity_top_k: int = 12, response_mode: str = "tree_summarize") -> RetrieverQueryEngine:
    retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
    response_synthesizer = get_response_synthesizer(response_mode=response_mode)
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

def format_response(raw_response: str, query: str) -> str:
    if "list" in query.lower():
        items = raw_response.split("\n")
        return "\n".join([f"- {item.strip()}" for item in items if item.strip()])
    elif "latest" in query.lower():
        return f"### Latest Updates\n{raw_response}\n### Conclusion\nSee cited sources for more details."
    return raw_response

class QueryRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = None
    top_k: int = 5
    deep_search: bool = False

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        query_lower = request.query.lower()
        timeframe = "24h" if "last 24 hours" in query_lower else "48h" if "last 48 hours" in query_lower else None
        
        # Handle sports-specific queries with API
        if any(keyword in query_lower for keyword in ["fixture", "match", "score", "standing", "result"]):
            sports_response = await fetch_sports_data(request.query, timeframe)
            if "error" not in sports_response:
                return sports_response
        
        # Construct query with prompt and context
        structured_query = f"{ROLE_AND_MISSION_PROMPT}\n\nCurrent date: {datetime.now().strftime('%Y-%m-%d %H:%M PST')}\nQuery: {request.query}"
        if timeframe:
            structured_query += f"\nFilter to {timeframe} timeframe."
        
        # Fetch X posts for real-time context
        x_posts = await fetch_x_posts(request.query, 24 if timeframe == "24h" else 48 if timeframe == "48h" else 24)
        if x_posts:
            structured_query += "\n\nRecent X Posts:\n" + "\n".join([f"- @{p['user']}: {p['text']} ({p['created_at']})" for p in x_posts])

        # Query engine
        query_engine = create_custom_query_engine(search_index, request.top_k, "tree_summarize")
        response = await query_engine.aquery(structured_query)
        
        # Post-process to ensure specificity
        formatted_response = format_response(response.response, request.query)
        if "[Company X]" in formatted_response or "[Player A]" in formatted_response:
            deep_response = await perform_deep_search(request.query, timeframe)
            formatted_response += f"\n\n### Additional Insights\n{deep_response['response']}"
        
        return {
            "response": formatted_response,
            "sources": [node.metadata for node in response.source_nodes] + [{"title": f"X: @{p['user']}", "link": p["created_at"]} for p in x_posts]
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