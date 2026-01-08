import streamlit as st
import smtplib
from email.mime.text import MIMEText
from io import StringIO
import contextlib

from functools import partial
from langchain_core.tools import Tool

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAIN AI Agent 🤖",
    page_icon="🤖",
    layout="wide"
)

# --- AGENT TOOL DEFINITIONS ---

# We define the tools in the global scope so the agent can access them.
# The actual credentials will be passed during the agent's initialization.

@tool
def send_email(content: str, recipient_email: str = None, sender_email: str = None, email_password: str = None, default_recipient: str = None):
    """
    Sends an email with a message. The subject line will be automatically generated.
    
    Use this tool when a user asks to send an email, a message, a note, or a notification to someone.

    The 'content' parameter should contain the ENTIRE message the user wants to send.
    For example, if the user says "email my manager that the project is done and the report is attached",
    the 'content' argument should be "The project is done and the report is attached.".

    The 'recipient_email' is the address to send the email to. If not provided, a default address will be used.
    """
    try:
        # Determine the recipient
        recipient_to_use = recipient_email if recipient_email else default_recipient
        if not recipient_to_use:
            return "Error: No recipient email provided and no default is set."
        
        if not sender_email or not email_password:
            return "Error: Sender email credentials are not configured."

        # Smart Subject Generation
        first_line = content.strip().split('\n')[0]
        subject = (first_line[:50] + '...') if len(first_line) > 50 else first_line
        if not subject:
            subject = "No Subject"

        body = content

        msg = MIMEText(body)
        msg['From'] = sender_email
        msg['To'] = recipient_to_use
        msg['Subject'] = subject

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, email_password)
        server.send_message(msg)
        server.quit()
        return f"Email sent successfully to {recipient_to_use}!"
    except Exception as e:
        return f"Failed to send email: {e}"

        

@tool
def save_note(filename: str, content: str):
    """
    Saves a text note to a file. Use this to remember information for later.
    The 'filename' should be a simple name like 'my_note.txt'.
    The 'content' is the text you want to save.
    """
    try:
        with open(filename, "w") as f:
            f.write(content)
        return f"Note saved successfully as {filename}."
    except Exception as e:
        return f"Failed to save note: {e}"

# --- AGENT INITIALIZATION FUNCTION ---

def initialize_agent(groq_api_key, sender_email, email_password, default_recipient):
    """Initializes and returns the LangChain agent."""
    
    # Initialize the LLM (brain)
    llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", groq_api_key=groq_api_key)
    
    # --- THIS IS THE CORRECTED SECTION ---

    # 1. Access the raw python function from the decorated tool object
    #    and create a partial function with the credentials "baked in".
    send_email_partial = partial(send_email.func, 
                                 sender_email=sender_email, 
                                 email_password=email_password, 
                                 default_recipient=default_recipient)

    # 2. Create a new Tool object from the partial function. We must provide the
    #    name and description from the original tool so the agent knows how to use it.
    email_tool_configured = Tool(
        name=send_email.name,
        func=send_email_partial,
        description=send_email.description
    )

    # --- END OF CORRECTION ---
    
    # Configure the other tools as before
    search_tool = DuckDuckGoSearchRun()
    
    tools = [search_tool, email_tool_configured, save_note]

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You have access to a set of tools to help you answer questions and perform tasks."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the AgentExecutor
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- STREAMLIT UI ---

st.title("RAIN AI Agent 🤖")
st.info("This is an AI agent that can search the web, send emails, and save notes. Provide your credentials in the sidebar to get started.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR FOR CREDENTIALS ---
with st.sidebar:
    st.header("⚙️ Credentials")
    st.write("Enter your API keys and email info below. Your data is not stored.")
    
    groq_api_key = st.text_input("Groq API Key", type="password")
    sender_email = st.text_input("Your Gmail Address (Sender)")
    email_password = st.text_input("Your Gmail App Password", type="password", help="[How to get a Gmail App Password](https://support.google.com/accounts/answer/185833)")
    default_recipient = st.text_input("Default Recipient Email", help="The email address for alerts if none is specified in the prompt.")
    
    # A button to initialize the agent
    if st.button("Initialize Agent"):
        if all([groq_api_key, sender_email, email_password, default_recipient]):
            with st.spinner("Initializing agent..."):
                st.session_state.agent_executor = initialize_agent(
                    groq_api_key, sender_email, email_password, default_recipient
                )
            st.success("Agent initialized successfully! You can now chat.")
        else:
            st.error("Please fill in all the credential fields.")

# --- MAIN CHAT INTERFACE ---

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if the agent has been initialized
if "agent_executor" not in st.session_state:
    st.warning("Please initialize the agent using the form in the sidebar.")
else:
    # Get user input
    if prompt := st.chat_input("Ask the agent to do something..."):
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the prompt with the agent
        with st.chat_message("assistant"):
            with st.spinner("🤖 The agent is thinking..."):
                try:
                    # Capture the agent's thought process (stdout)
                    string_io = StringIO()
                    with contextlib.redirect_stdout(string_io):
                        response = st.session_state.agent_executor.invoke({"input": prompt})
                    
                    agent_thoughts = string_io.getvalue()
                    final_answer = response.get("output", "I'm sorry, I encountered an error.")

                    # Display the agent's thought process in an expander
                    with st.expander("Show Agent's Thought Process"):
                        st.text_area("Logs", agent_thoughts, height=300, disabled=True)
                    
                    # Display the final answer
                    st.markdown(final_answer)

                    # Add the final answer to the session state
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})