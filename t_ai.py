from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import glob
import random
from dotenv import load_dotenv
from openai import OpenAI

# 1. Setup & Config
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="LPS Support API")

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=API_KEY) if API_KEY else None

# --- DATA MODELS ---

class TicketRequest(BaseModel):
    full_name: str
    email: str
    category: str
    description: str
    priority: str

class ChatStartRequest(BaseModel):
    issue: str

class QuickHelpRequest(BaseModel):
    description: str

class QAPair(BaseModel):
    question: str
    answer: str

class DiagnosisRequest(BaseModel):
    issue: str
    qa_history: List[QAPair]

class EscalationRequest(BaseModel):
    name: str
    email: str
    issue: str
    qa_history: List[QAPair] 

# --- HELPER FUNCTIONS ---

def generate_id():
    return f"TICKET-{random.randint(100000, 999999)}"

def load_knowledge_base():
    """
    NAIVE RAG: Reads all .txt and .md files from the 'knowledge_base' folder
    and combines them into a single context string.
    """
    kb_content = ""
    kb_path = "knowledge_base"
    
    if not os.path.exists(kb_path):
        os.makedirs(kb_path) # Create if it doesn't exist
        return "No internal knowledge base documents found."

    # Read all .txt and .md files
    files = glob.glob(os.path.join(kb_path, "*.txt")) + glob.glob(os.path.join(kb_path, "*.md"))
    
    if not files:
        return "No internal documents found."

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                filename = os.path.basename(file_path)
                kb_content += f"\n--- DOCUMENT: {filename} ---\n"
                kb_content += f.read()
                kb_content += "\n"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return kb_content

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "version": "1.2.0 (RAG Enabled)"}

# 1. Standard Ticket Endpoint (With RAG context for better categorization/reply)
@app.post("/api/ticket/create")
def create_ticket(req: TicketRequest):
    if not client: raise HTTPException(500, "OpenAI API missing")
    
    ticket_id = generate_id()
    # Load RAG Context
    kb_context = load_knowledge_base()

    prompt = f"""
    Write a formal support ticket email response.
    
    INTERNAL KNOWLEDGE BASE (Use this if relevant):
    {kb_context}
    
    TICKET DETAILS:
    Ticket ID: {ticket_id}
    User: {req.full_name} ({req.email})
    Category: {req.category}
    Priority: {req.priority}
    Issue: {req.description}
    
    Task: Write the email response. If the internal knowledge base has a solution, include it.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
        )
        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# 2. Start AI Chat (RAG helps ask better questions)
@app.post("/api/chat/start")
def start_chat(req: ChatStartRequest):
    if not client: raise HTTPException(500, "OpenAI API missing")
    
    kb_context = load_knowledge_base()

    prompt = f"""
    You are a Triage Bot.
    
    INTERNAL DOCS:
    {kb_context}
    
    User Issue: "{req.issue}"
    
    Task: Generate 3-5 clarifying questions. 
    If the internal docs mention specific troubleshooting steps for this issue, ask if they have tried them.
    Format: Single text block, separated by "||". No numbering.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.5
        )
        content = response.choices[0].message.content
        questions = [q.strip() for q in content.split("||") if q.strip()]
        
        return {"success": True, "questions": questions}
    except Exception as e:
        raise HTTPException(500, str(e))

# 3. Final Diagnosis (Critical for RAG)
@app.post("/api/chat/diagnose")
def get_diagnosis(req: DiagnosisRequest):
    if not client: raise HTTPException(500, "OpenAI API missing")
    
    kb_context = load_knowledge_base()
    history_text = "\n".join([f"Q: {p.question}\nA: {p.answer}" for p in req.qa_history])
    
    prompt = f"""
    Tier 2 Support Agent. 
    
    INTERNAL KNOWLEDGE BASE (Priority Source):
    {kb_context}
    
    User Issue: "{req.issue}"
    Diagnostic Q&A:
    {history_text}
    
    Task: Provide a final technical diagnosis and step-by-step solution.
    ALWAYS prioritize solutions found in the INTERNAL KNOWLEDGE BASE over general knowledge.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.3
        )
        return {"success": True, "diagnosis": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(500, str(e))

# 4. Escalate / Stop Halfway
@app.post("/api/chat/escalate")
def escalate_ticket(req: EscalationRequest):
    if not client: raise HTTPException(500, "OpenAI API missing")
    
    ticket_id = generate_id()
    if not req.qa_history:
        history_summary = "User provided no answers."
    else:
        history_summary = "\n".join([f"- Q: {p.question}\n  A: {p.answer}" for p in req.qa_history])
    
    prompt = f"""
    Create an Escalation Ticket.
    User: {req.name} ({req.email})
    Ticket ID: {ticket_id}
    Original Issue: {req.issue}
    Diagnostic Log: {history_summary}
    Task: Summarize what was tried and flag for human review.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": prompt}]
        )
        return {
            "success": True,
            "ticket_id": ticket_id,
            "ticket_text": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# 5. Quick Help (RAG Enabled)
@app.post("/api/ticket/quick-help")
def quick_help(req: QuickHelpRequest):
    if not client: raise HTTPException(500, "OpenAI API missing")
    
    kb_context = load_knowledge_base()

    prompt = f"""
    You are a Helpful IT Assistant.
    
    INTERNAL KNOWLEDGE BASE:
    {kb_context}
    
    The user is about to submit a ticket with this description: "{req.description}"
    
    Task: 
    Provide 3 concise, bullet-pointed "Quick Fix" suggestions based on the Internal Knowledge Base if applicable.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.7
        )
        return {
            "success": True,
            "suggestion": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(500, str(e))