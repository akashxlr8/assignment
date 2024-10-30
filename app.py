import streamlit as st
from dataclasses import dataclass
from rag import TravelAssistant
from main import chat_with_model

@dataclass
class Message:
    actor: str
    payload: str

# Constants
USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

def initialize_session_state():
    """Initialize session state variables"""
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [
            Message(actor=ASSISTANT, 
                   payload="Hi! I'm your Travel Assistant. Ask me anything about your journey details!")
        ]
    if "travel_assistant" not in st.session_state:
        st.session_state.travel_assistant = TravelAssistant()

def main():
    st.title("Travel Assistant RAG")
    
    # Initialize session state
    initialize_session_state()
    
    # Add greeting detection
    greetings = {'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'}
    
    # Display chat messages
    for msg in st.session_state[MESSAGES]:
        st.chat_message(msg.actor).write(msg.payload)

    # Get user input
    if prompt := st.chat_input("What would you like to know about your journey?"):
        # Add user message to chat
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
        st.chat_message(USER).write(prompt)

        # Check if prompt is a greeting
        if prompt.lower().strip() in greetings:
            greeting_response = "Hello! I'm your travel assistant. How can I help you with your travel plans today?"
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=greeting_response))
            st.chat_message(ASSISTANT).write(greeting_response)
            return

        # Show spinner while processing (only for non-greetings)
        with st.spinner("Searching travel documents..."):
            # Get context and generate response
            result = st.session_state.travel_assistant.get_answer(prompt)
            
            # Optional: Display relevant documents (can be toggled)
            with st.expander("View relevant documents"):
                for doc in result['similar_docs']:
                    st.markdown("---")
                    st.json(doc.page_content)

            # Get AI response
            with st.spinner("Generating response..."):
                answer = chat_with_model(result['prompt'])
                
                # Add assistant message to chat
                st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=answer))
                st.chat_message(ASSISTANT).write(answer)

    # Add sidebar options
    with st.sidebar:
        st.title("Options")
        if st.button("Clear Conversation"):
            st.session_state[MESSAGES] = [
                Message(actor=ASSISTANT, 
                       payload="Hi! I'm your Travel Assistant. Ask me anything about your journey details!")
            ]
            st.rerun()

if __name__ == "__main__":
    main()

