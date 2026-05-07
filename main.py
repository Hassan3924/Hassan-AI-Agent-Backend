# from phi.agent import Agent 
# from phi.model.groq import Groq 
# from phi.tools.yfinance import YFinanceTools
# from phi.tools.duckduckgo import DuckDuckGo
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.team import Team
from pydantic import BaseModel
import openai
import os
import groq, phi
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

model_name = "qwen/qwen3-32b"

load_dotenv()

#Web Search Agent
web_search_agent = Agent(
    name = "Web Search Agent",
    role = "Search the web for the information",
    model = Groq(
        id = model_name),
    tools = [DuckDuckGoTools()],
    instructions=[
        "You are a general web search expert.",
        "Never use any finance tools. Only use DuckDuckGo.",
        "After getting results, summarize clearly and include sources."
    ],    
    markdown=True
)

# Financial Agent
finance_agent = Agent(
    name = "Financial Agent",
    role = "Analyze financial data and provide insights",
    model = Groq(id = model_name),
    tools = [YFinanceTools()],
    instructions=[
        "Only answer questions about stocks, companies, or financial data.",
        "If the query is about a person, athlete, or non-financial topic, reply that you only handle financial information."],
    markdown=True
)

multi_ai_agent = Team(
        members = [web_search_agent, finance_agent],
        model = Groq(id = model_name),
        mode = "route",
        instructions=[
        "Route to Web Search Agent for people, news, sports, or general topics.",
        "Route to Financial Agent ONLY for stock tickers or company financial performance."],
         markdown=True
    )

app = FastAPI(title = "Hassan's Multi Agent System")

# Allowing Netilfy website to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hassanaghauri.netlify.app", 
                   "http://localhost:3000", 
                   "http://localhost:8001",
                   "http://127.0.0.1:8001",
                   "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = multi_ai_agent.run(input=request.message)
        # Safely extract the actual text message
        if hasattr(result, "content") and result.content:
            response = result.content
            print(response)
        else:
            response = str(result)

        print("AI Response:", response)

        return {"response": response}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Health check (optional)
@app.get("/")
async def root():
    return {"status": "AI Agent is running ✅"}