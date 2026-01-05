from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
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

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "version": "1.1.0"}

# 1. Standard Ticket Endpoint
@app.post("/api/ticket/create")
def create_ticket(req: TicketRequest):
    if not client: raise HTTPException(500, "OpenAI API missing")
    
    ticket_id = generate_id()
    prompt = f"""
    Write a formal support ticket email.
    Ticket ID: {ticket_id}
    User: {req.full_name} ({req.email})
    Category: {req.category}
    Priority: {req.priority}
    Issue: {req.description}
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

# 2. Start AI Chat (Get Questions)
@app.post("/api/chat/start")
def start_chat(req: ChatStartRequest):
    if not client: raise HTTPException(500, "OpenAI API missing")
    
    prompt = f"""
    You are a Triage Bot. User Issue: "{req.issue}"
    Generate 3-5 clarifying questions. 
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

# 3. Final Diagnosis (Chat)
@app.post("/api/chat/diagnose")
def get_diagnosis(req: DiagnosisRequest):
    if not client: raise HTTPException(500, "OpenAI API missing")
    
    history_text = "\n".join([f"Q: {p.question}\nA: {p.answer}" for p in req.qa_history])
    
    prompt = f"""
    Tier 2 Support Agent. 
    Issue: "{req.issue}"
    Diagnostic Q&A:
    {history_text}
    Task: Provide a final technical diagnosis and step-by-step solution.
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

# 5. NEW: Quick Help (Description Only)
@app.post("/api/ticket/quick-help")
def quick_help(req: QuickHelpRequest):
    if not client: raise HTTPException(500, "OpenAI API missing")
    
    prompt = f"""
    You are a Helpful IT Assistant.
    The user is about to submit a ticket with this description: "{req.description}"
    
    Task: 
    Provide 3 concise, bullet-pointed "Quick Fix" suggestions they can try immediately.
    Keep it short and encouraging. If the issue is vague, suggest restarting the device.
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