import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import requests

# Llama Index & Readers imports
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentWorkflow,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.workflow import Context

# Import a web reader to load website content
from llama_index.readers.web import SimpleWebPageReader

# SerpAPI for web search functionality
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

# Environment variables for credentials and API keys
API_ENDPOINT = "https://d7b29760-fb5b-4e81-8c0c-d49e7b604ed3-us-east-2.apps.astra.datastax.com"
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Add your SerpAPI key to .env
COLLECTION_NAME = "sf_data"
SPORTS_DB_API_KEY = "392246"  # Your The Sports DB API key

# ---------------------------
# Initialize OpenAI Model and Embedding
# ---------------------------
embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002")
llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

# ---------------------------
# Initialize Astra DB Vector Store (Query-only mode)
# ---------------------------
astra_db_store = AstraDBVectorStore(
    token=ASTRA_DB_TOKEN,
    api_endpoint=API_ENDPOINT,
    collection_name=COLLECTION_NAME,
    embedding_dimension=1536,  # Ensure this matches your embedding model's dimension
)

# ---------------------------
# Function to load website content using a web reader.
# ---------------------------
def load_website_documents():
    """
    Load website content such as posts and sponsorship pages.
    Replace the URLs below with your actual website pages.
    """
    reader = SimpleWebPageReader()
    urls = [
        "https://www.sponsorforce.net/#/portal/home",
        "https://www.sponsorforce.net/#/portal/topic",
        "https://www.sponsorforce.net/#/portal/resource",
        "https://www.sponsorforce.net/#/portal/perspective",
        # Add additional URLs as needed.
    ]
    documents = reader.load_data(urls=urls)
    print(f"✅ Loaded {len(documents)} documents from the website.")
    return documents

# Load website documents.
website_docs = load_website_documents()

# ---------------------------
# Create a Storage Context and Build the Index
# ---------------------------
storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

# Build the index using website content.
# (If you want to add more documents later, you'll need to update or rebuild the index accordingly.)
index = VectorStoreIndex(website_docs, storage_context=storage_context, embed_model=embed_model)

# ---------------------------
# Centralized Prompt for the agents
# ---------------------------
ROLE_AND_MISSION_PROMPT = (
    "You are an expert in sponsorship strategies, marketing, and business development. Your mission is to:\n"
    "1. **Educate and Inspire**: Provide in-depth insights, practical advice, and actionable recommendations that leave the user feeling informed and empowered.\n"
    "2. **Use Examples**: Incorporate real-world examples, case studies, and industry trends to illustrate your points.\n"
    "3. **Data-Driven Insights**: Include relevant data, statistics, and references to support your claims and establish credibility.\n"
    "4. **Step-by-Step Guidance**: Break down complex concepts into clear, actionable steps that the user can implement immediately.\n"
    "5. **Professional Yet Approachable Tone**: Maintain a professional tone while ensuring the response is engaging and easy to understand.\n\n"
    "**Query Context**:\n"
    "Question: {question}\n"
    "Temporal Expression Identified: {temporal_expression}\n"
    "Current Date: {current_date}\n\n"
    "**Website Forwarding Functionality**:\n"
    "If the user’s query can be better addressed by visiting SponsorForce.net, provide a direct and professional redirection.\n"
    "Example Phrasing:\n"
    "- 'For more detailed resources and tools, please visit our website at SponsorForce.net.'\n"
    "- 'You can explore additional insights and services on our website: SponsorForce.net.'\n\n"
    "**Answer Guidelines**:\n"
    "1. Always start with a direct response to the user's question, addressing the specific query and timeframe.\n"
    "2. Provide detailed explanations, including background information, key features, and benefits.\n"
    "3. Use bullet points or numbered lists to organize information for clarity.\n"
    "4. Include relevant examples, case studies, or data to support your response.\n"
    "5. When appropriate, encourage users to visit SponsorForce.net for more comprehensive information, tools, or resources.\n"
    "6. Avoid generic responses—always tailor your answer to the user's query."
)

# ---------------------------
# Tools: Functions used by the agents.
# ---------------------------
async def search_database(ctx: Context, query: str) -> str:
    """Search the vector database for relevant information using semantic search."""
    print(f"🔍 Searching database for query: {query}")
    try:
        # Use the index to perform a semantic search
        query_engine = index.as_query_engine(
            similarity_top_k=5,  # Retrieve top 5 most relevant documents
            response_mode="tree_summarize",  # Use "tree_summarize" for detailed responses
        )
        response = query_engine.query(query)

        if not response or not response.response:
            print("⚠️ No relevant information found in the database.")
            return "No relevant information found in the database."

        print(f"📄 Retrieved Response: {response.response}")
        return response.response
    except Exception as e:
        print(f"❌ Error during database query: {e}")
        return "Database search failed."

async def search_web(ctx: Context, query: str) -> str:
    """Search the web or fetch sports data for relevant information."""
    if "live scores" in query.lower():
        return await fetch_live_scores()
    elif "upcoming fixtures" in query.lower() or "next matches" in query.lower():
        # Extract team name from the query (e.g., "upcoming fixtures for Arsenal")
        team_name = query.lower().replace("upcoming fixtures for", "").strip()
        return await fetch_upcoming_fixtures(team_name)

    print(f"🌐 Searching the web for query: {query}")
    try:
        if not SERPAPI_KEY:
            raise ValueError("SERPAPI_KEY environment variable is not set.")

        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 20,  # Number of search results to retrieve
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" not in results:
            print("⚠️ No relevant information found on the web.")
            return "No relevant information found on the web."

        # Extract and format search results
        web_results = []
        for result in results["organic_results"]:
            web_results.append(f"{result['title']}: {result['link']}\n{result.get('snippet', '')}")

        formatted_results = "\n\n".join(web_results)
        print(f"🌐 Retrieved Web Results: {formatted_results}")
        return formatted_results
    except Exception as e:
        print(f"❌ Error during web search: {e}")
        return "Web search failed."

async def fetch_live_scores() -> str:
    """Fetch live soccer scores from The Sports DB."""
    url = f"https://www.thesportsdb.com/api/v2/json/{SPORTS_DB_API_KEY}/livescore/soccer"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data.get("events"):
            return "No live matches at the moment."

        live_matches = []
        for event in data["events"]:
            home_team = event["strHomeTeam"]
            away_team = event["strAwayTeam"]
            score = f"{event['intHomeScore']} - {event['intAwayScore']}"
            live_matches.append(f"{home_team} {score} {away_team}")

        return "\n".join(live_matches)
    except Exception as e:
        print(f"❌ Error fetching live scores: {e}")
        return "Failed to fetch live scores."

async def fetch_upcoming_fixtures(team_name: str) -> str:
    """Fetch upcoming fixtures for a specific team."""
    # First, get the team ID
    team_url = f"https://www.thesportsdb.com/api/v1/json/{SPORTS_DB_API_KEY}/searchteams.php?t={team_name}"
    try:
        response = requests.get(team_url)
        response.raise_for_status()
        team_data = response.json()

        if not team_data.get("teams"):
            return f"No team found with the name {team_name}."

        team_id = team_data["teams"][0]["idTeam"]

        # Now, fetch upcoming fixtures for the team
        fixtures_url = f"https://www.thesportsdb.com/api/v1/json/{SPORTS_DB_API_KEY}/eventsnext.php?id={team_id}"
        response = requests.get(fixtures_url)
        response.raise_for_status()
        fixtures_data = response.json()

        if not fixtures_data.get("events"):
            return f"No upcoming fixtures found for {team_name}."

        fixtures = []
        for event in fixtures_data["events"]:
            home_team = event["strHomeTeam"]
            away_team = event["strAwayTeam"]
            date = event["dateEvent"]
            time = event["strTime"]
            fixtures.append(f"{home_team} vs {away_team} on {date} at {time}")

        return "\n".join(fixtures)
    except Exception as e:
        print(f"❌ Error fetching upcoming fixtures: {e}")
        return "Failed to fetch upcoming fixtures."

async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Record notes on a given topic."""
    current_state = await ctx.get("state")
    current_state.setdefault("research_notes", {})[notes_title] = notes
    await ctx.set("state", current_state)
    return "Notes recorded."

async def write_report(ctx: Context, report_content: str) -> str:
    """Write a detailed report on a given topic."""
    current_state = await ctx.get("state")
    research_notes = current_state.get("research_notes", {})

    combined_notes = "\n\n".join(f"### {title}\n{content}" for title, content in research_notes.items())
    full_report = f"{combined_notes}\n\n{report_content}"

    current_state["report_content"] = full_report
    await ctx.set("state", current_state)

    print(f"📝 Generated Report Length: {len(full_report)} characters")
    return full_report

async def review_report(ctx: Context, review: str) -> str:
    """Review a report and provide feedback."""
    current_state = await ctx.get("state")
    current_state["review"] = review
    await ctx.set("state", current_state)
    return "Report reviewed."

# ---------------------------
# Agents Setup
# ---------------------------
research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Search the vector database and the web for information on a given topic.",
    system_prompt=ROLE_AND_MISSION_PROMPT,
    llm=llm,
    tools=[search_web],  # Use the search_web tool
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Write a report using research notes.",
    system_prompt=ROLE_AND_MISSION_PROMPT,
    llm=llm,
    tools=[write_report, record_notes],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Review a report and provide feedback.",
    system_prompt=ROLE_AND_MISSION_PROMPT,
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)

# Define the workflow with initial state
agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent="WriteAgent",
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)

# ---------------------------
# API Server Setup
# ---------------------------
app = FastAPI()

# Add CORS middleware to allow cross-origin requests (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for queries
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Handle user queries from the frontend."""
    try:
        user_message = request.query

        # Fetch live scores for relevant queries
        if "live scores" in user_message.lower():
            live_scores = await fetch_live_scores()
            return {"response": live_scores}

        # Fetch upcoming fixtures for relevant queries
        if "upcoming fixtures" in user_message.lower() or "next matches" in user_message.lower():
            team_name = user_message.lower().replace("upcoming fixtures for", "").strip()
            fixtures = await fetch_upcoming_fixtures(team_name)
            return {"response": fixtures}

        # Use the workflow for other queries
        handler = agent_workflow.run(user_msg=user_message)

        final_response = None
        async for event in handler.stream_events():
            if isinstance(event, AgentOutput):
                if event.response.content:
                    final_response = event.response.content

        if not final_response:
            raise HTTPException(status_code=404, detail="No response generated by the workflow")

        # Enhance the response with additional details from the database.
        # Provide the required 'workflow' argument to Context.
        database_response = await search_database(Context(workflow=agent_workflow), user_message)
        if database_response and database_response != "No relevant information found in the database.":
            final_response = f"{final_response}\n\n**Additional Information from Our Database**:\n{database_response}"

        return {"response": final_response}

    except Exception as e:
        print(f"❌ Error handling query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
def read_root():
    """Root endpoint to verify the backend is running."""
    return {"message": "Backend is running and ready to process queries!"}

# ---------------------------
# Run the FastAPI App
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
