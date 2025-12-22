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

# --- HELPER FUNCTIONS ---

def generate_ticket_id():
    return f"TICKET-{random.randint(100000, 999999)}"

def generate_escalation_ticket(name, email, initial_issue, qa_history):
    """
    Generates a ticket when AI diagnosis fails or is skipped.
    """
    if not client: return "Error"
    
    ticket_id = generate_ticket_id()
    # Format history from list of tuples
    history_text = "\n".join([f"- Q: {q}\n  A: {a}" for q, a in qa_history])
    
    prompt = f"""
    You are a Support Ticket Generator.
    
    CONTEXT:
    The user was interacting with the AI Agent but decided to escalate to a human.
    
    User: {name} ({email})
    Ticket ID: {ticket_id}
    Original Issue: {initial_issue}
    
    AI CHAT LOG (Partial history of what was asked so far):
    {history_text}
    
    TASK:
    Write a formal Escalation Ticket. 
    - Header: Ticket #{ticket_id} - Escalated to Human Support
    - Section 1: User Details & Original Issue
    - Section 2: Partial Diagnostic Info (Summarize the Q&A log briefly)
    - Section 3: Action Required (Human agent to follow up).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating ticket: {e}"

def get_diagnostic_questions(issue_description):
    """Step 1: Get questions."""
    if not client: return []
    
    prompt = f"""
    You are an IT Support Triage Bot.
    User Issue: "{issue_description}"
    
    Task: Generate 3 to 5 critical clarifying questions to diagnose this specific issue. 
    Do not provide a solution yet. Just ask questions.
    
    FORMATTING RULE: 
    Write the questions in a single block of text. 
    Separate each question with a double pipe symbol "||". 
    Example: Is the light blinking?||Have you tried restarting?||When did this start?
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        content = response.choices[0].message.content
        questions = [q.strip() for q in content.split("||") if q.strip()]
        return questions
    except Exception as e:
        st.error(f"AI Error: {e}")
        return []

def get_final_diagnosis(issue_description, qa_history):
    """Step 2: Get final diagnosis."""
    if not client: return "Error: API not connected."
    
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_history])
    
    prompt = f"""
    You are an Expert Tier 2 Tech Support Agent.
    Original Issue: "{issue_description}"
    Clarifying Q&A:
    {history_text}
    
    Task: 
    Provide a final technical diagnosis and a solution based on these answers.
    Format:
    1. **Diagnosis**: What is likely wrong.
    2. **Solution**: Step-by-step fix.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

def get_ticket_solution(name, category, description, ticket_id):
    """Standard ticket generation."""
    if not client: return "Error"
    prompt = f"""
    Context: User {name}, Ticket {ticket_id}, Category {category}.
    Issue: {description}
    Task: Write a support ticket response.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- PAGE LAYOUT ---
st.set_page_config(page_title="LPS Tech Support", page_icon="💠", layout="wide")

# Header
col1, col2 = st.columns([1, 15])
with col1:
    if os.path.exists("logo.webp"):
        st.image("logo.webp", width=70)
    else:
        st.write("💠")
with col2:
    st.title("LPS Tech Support System")

# Navigation
page = st.radio("Select Mode:", ["📝 Send Support Ticket", "🤖 Direct AI Chat"], horizontal=True)
st.divider()

# --- MODE 1: SEND TICKET ---
if page == "📝 Send Support Ticket":
    st.subheader("Submit a Formal Ticket")
    with st.form("support_ticket_form"):
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
        with c2:
            subject = st.text_input("Subject")
            category = st.selectbox("Category", ["Hardware", "Software", "Network", "Access", "Other"])
        
        description = st.text_area("Description", height=150)
        submitted = st.form_submit_button("Generate Ticket")
    
    if submitted and description:
        with st.spinner("Generating Ticket..."):
            tid = generate_ticket_id()
            res = get_ticket_solution(full_name, category, description, tid)
            st.success("Ticket Generated!")
            st.markdown(res)

# --- MODE 2: DIRECT AI CHAT ---
elif page == "🤖 Direct AI Chat":
    st.subheader("Interactive Triage Agent")
    
    # Initialize Session State
    if "chat_step" not in st.session_state:
        st.session_state.chat_step = "init"
    if "chat_questions" not in st.session_state:
        st.session_state.chat_questions = []
    if "chat_answers" not in st.session_state:
        st.session_state.chat_answers = []
    if "chat_current_q_index" not in st.session_state:
        st.session_state.chat_current_q_index = 0
    if "original_issue" not in st.session_state:
        st.session_state.original_issue = ""
    if "final_diagnosis" not in st.session_state:
        st.session_state.final_diagnosis = None
    if "show_escalation" not in st.session_state:
        st.session_state.show_escalation = False

    # STEP 1: Get the Initial Issue
    if st.session_state.chat_step == "init":
        user_issue = st.text_area("What seems to be the problem?", height=100, placeholder="e.g., My computer screen is black but the fan is spinning.")
        
        if st.button("Start Diagnosis"):
            if not user_issue:
                st.warning("Please describe the issue first.")
            else:
                with st.spinner("Analyzing issue and generating clarifying questions..."):
                    qs = get_diagnostic_questions(user_issue)
                    if qs:
                        st.session_state.original_issue = user_issue
                        st.session_state.chat_questions = qs
                        st.session_state.chat_step = "questioning"
                        st.session_state.final_diagnosis = None 
                        st.session_state.show_escalation = False
                        st.rerun()
                    else:
                        st.error("Could not generate questions. Please try again.")

    # STEP 2: Ask Questions One by One
    elif st.session_state.chat_step == "questioning":
        idx = st.session_state.chat_current_q_index
        total = len(st.session_state.chat_questions)
        
        st.progress((idx / total))
        st.caption(f"Question {idx + 1} of {total}")
        
        current_q = st.session_state.chat_questions[idx]
        st.markdown(f"### 🤖 AI Asks: \n**{current_q}**")
        
        answer = st.text_input("Your Answer:", key=f"ans_{idx}")
        
        col_next, col_stop = st.columns([1, 1])
        
        with col_next:
            if st.button("Next ➡️"):
                if not answer:
                    st.warning("Please provide an answer.")
                else:
                    st.session_state.chat_answers.append((current_q, answer))
                    if idx + 1 < total:
                        st.session_state.chat_current_q_index += 1
                        st.rerun()
                    else:
                        st.session_state.chat_step = "diagnosis"
                        st.rerun()

        # --- NEW: Early Exit Button ---
        with col_stop:
            if st.button("🛑 Stop & Create Ticket"):
                # Save whatever answer was currently typed (optional)
                if answer:
                    st.session_state.chat_answers.append((current_q, answer))
                
                # Force move to end step, but set flags to skip AI diagnosis
                st.session_state.chat_step = "diagnosis"
                st.session_state.show_escalation = True # Force show form
                st.session_state.final_diagnosis = "SKIPPED" # Flag to skip logic
                st.rerun()

    # STEP 3: Final Diagnosis OR Escalation
    elif st.session_state.chat_step == "diagnosis":
        
        # Only run diagnosis if the user didn't skip
        if st.session_state.final_diagnosis != "SKIPPED":
            
            # Run AI Diagnosis if not already cached
            if st.session_state.final_diagnosis is None:
                with st.spinner("Compiling final diagnosis..."):
                    final_res = get_final_diagnosis(
                        st.session_state.original_issue, 
                        st.session_state.chat_answers
                    )
                    st.session_state.final_diagnosis = final_res
            
            # Show Diagnosis UI
            st.success("Analysis Complete.")
            st.markdown("### 🩺 Final Diagnosis & Solution")
            st.markdown(st.session_state.final_diagnosis)
            
            # View History
            with st.expander("📝 View Troubleshooting History"):
                st.write(f"**Issue:** {st.session_state.original_issue}")
                for q, a in st.session_state.chat_answers:
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"**A:** {a}")
                    st.divider()

            st.divider()
            st.subheader("Did this solve your problem?")
            col_y, col_n = st.columns([1, 4])
            
            with col_y:
                if st.button("✅ Yes, Solved!"):
                    st.balloons()
                    st.success("Great! Happy to help.")
                    if st.button("Start New Chat"):
                        st.session_state.chat_step = "init"
                        st.session_state.chat_questions = []
                        st.session_state.chat_answers = []
                        st.session_state.chat_current_q_index = 0
                        st.session_state.original_issue = ""
                        st.session_state.final_diagnosis = None
                        st.session_state.show_escalation = False
                        st.rerun()
            with col_n:
                if st.button("❌ No, Create Ticket"):
                    st.session_state.show_escalation = True
                    st.rerun()

        # --- ESCALATION FORM ---
        # Shows if user clicked "Stop" in Step 2 OR "No" in Step 3
        if st.session_state.show_escalation:
            if st.session_state.final_diagnosis == "SKIPPED":
                st.warning("Diagnostic process stopped by user. Proceeding to ticket creation.")
            else:
                st.warning("We're sorry the AI solution didn't work. Please provide details to escalate.")
            
            with st.form("escalation_form"):
                e_name = st.text_input("Your Name")
                e_email = st.text_input("Your Email")
                e_submitted = st.form_submit_button("Submit Escalation Ticket")
            
            if e_submitted:
                if e_name and e_email:
                    with st.spinner("Generating Escalation Ticket with partial context..."):
                        ticket_text = generate_escalation_ticket(
                            e_name, 
                            e_email, 
                            st.session_state.original_issue, 
                            st.session_state.chat_answers
                        )
                        st.success("Escalation Ticket Created Successfully!")
                        st.markdown(ticket_text)
                else:
                    st.error("Please provide Name and Email.")