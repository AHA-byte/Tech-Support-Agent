import streamlit as st
import os
import random
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load Environment Variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 2. Configure OpenAI Client
client = None
if API_KEY:
    try:
        client = OpenAI(api_key=API_KEY)
    except Exception as e:
        st.error(f"Error configuring OpenAI Client: {e}")

def generate_ticket_id():
    """Generates a random 6-digit ticket number."""
    return f"TICKET-{random.randint(100000, 999999)}"

def get_ai_solution(name, category, description, ticket_id):
    """
    Sends the user issue to OpenAI (GPT-4o) for a formatted response.
    """
    if not client:
        return "Error: API Key not configured."
        
    try:
        system_prompt = f"""
        You are an expert IT Support Agent (Tier 2 Level).
        
        Context:
        - User Name: {name}
        - Ticket ID: {ticket_id}
        - Category: {category}
        
        Task:
        Draft a highly accurate, technical, and professional response.
        
        Format Requirements:
        1. Header: "Ticket #{ticket_id} - [Precise Technical Summary]"
        2. Greeting: "Dear {name},"
        3. Analysis: Briefly explain probable Root Cause.
        4. Solution: Detailed numbered steps to fix the issue.
        5. Closing: Professional sign-off.
        """

        response = client.chat.completions.create(
            model="gpt-4o",  # High accuracy model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Issue Description: {description}"}
            ],
            temperature=0.3, # Keep it factual and consistent
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error connecting to AI service: {str(e)}"

# --- UI Layout ---
st.set_page_config(page_title="AI Tech Support (OpenAI)", page_icon="logo.webp")

# Create two columns: a small one for the logo, a wide one for the title
# The [1, 5] ratio means the second column is 5x wider than the first
col1, col2 = st.columns([1, 15]) 

with col1:
    # Adjust 'width' to match the size of your text height roughly
    st.image("logo.webp", width=70) 

with col2:
    st.title("LPS Tech Support")
st.markdown("Submit your issue below. Powered by **OpenAI GPT-4o** for high-accuracy resolution.")

with st.form("support_ticket_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        full_name = st.text_input("Full Name", placeholder="Jane Doe")
        email = st.text_input("Email Address", placeholder="jane@example.com")
        priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
        
    with col2:
        subject = st.text_input("Subject", placeholder="e.g., VPN handshake failure")
        category = st.selectbox("Category", ["Hardware", "Software", "Network", "Access/Login", "Other"])
    
    description = st.text_area("Detailed Issue Description", height=150, placeholder="Please provide error codes or steps to reproduce...")
    
    submitted = st.form_submit_button("Generate Support Ticket")

# --- Logic on Submit ---
if submitted:
    if not API_KEY:
        st.error("Please configure your OPENAI_API_KEY in the .env file.")
    elif not description or not full_name:
        st.warning("Please fill in at least your Name and the Description.")
    else:
        with st.spinner("Consulting Knowledge Base (OpenAI)..."):
            ticket_id = generate_ticket_id()
            solution_text = get_ai_solution(full_name, category, description, ticket_id)
            
            if "Error" in solution_text:
                st.error(solution_text)
            else:
                st.success(f"Ticket Created Successfully!")
                st.divider()
                st.subheader("Generated Response")
                st.markdown(solution_text)
                
                with st.expander("Ticket Metadata"):
                    st.json({
                        "ticket_id": ticket_id,
                        "model": "gpt-4o",
                        "priority": priority,
                        "subject": subject
                    })