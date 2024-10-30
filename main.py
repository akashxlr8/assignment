from huggingface_hub import InferenceClient
from rag import TravelAssistant
import os
import streamlit as st

def chat_with_model(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=1000):
    try:
        client = InferenceClient(st.secrets["HF_TOKEN"])
        response = client.text_generation(
            model=model,
            prompt=prompt,
            max_new_tokens=max_tokens
        )
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    # Initialize the travel assistant
    assistant = TravelAssistant()
    
    print("\nWelcome to your Travel Assistant!")
    print("You can ask questions about your journey details.")
    print("Type 'quit' to exit.\n")

    # Add greeting detection
    greetings = {'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'}

    while True:
        # Get user input
        query = input("\nWhat would you like to know about your journey? ").strip()
        
        # Check for exit condition
        if query.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using Travel Assistant!")
            break
            
        if not query:
            print("Please ask a question!")
            continue

        # Check if query is a greeting
        if query.lower() in greetings:
            print("\nAI Response:")
            print("Hello! I'm your travel assistant. How can I help you with your travel plans today?")
            continue

        # Get context and generate response
        result = assistant.get_answer(query)
        
        # Get AI response
        answer = chat_with_model(result['prompt'])
        
        # Print similar documents (optional - for debugging)
        print("\nRelevant documents found:")
        for doc in result['similar_docs']:
            print("-" * 50)
            print(doc.page_content)
            
        # Print the final answer
        print("\nAI Response:")
        print(answer)

if __name__ == "__main__":
    main()