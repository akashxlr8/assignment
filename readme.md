# Travel Assistant RAG

A conversational AI assistant that helps users query their travel information using RAG (Retrieval Augmented Generation) architecture. The assistant can answer questions about travel details, including flight information, dates, and times.

## Technical Implementation

- **RAG Architecture**: Implemented using Pinecone for vector storage and LLaMA 3 for response generation
- **Embedding Model**: Utilizing Jina AI embeddings for document vectorization
- **Frontend**: Built with Streamlit for an interactive chat interface
- **Backend**: Python-based implementation with modular design patterns
- **Features**: 
  - Natural conversation flow with greeting detection
  - Real-time document similarity search
  - Context-aware response generation
  - Expandable view of relevant documents for transparency

## Live Demo

The application is deployed and accessible at: https://assignment-rag.streamlit.app/
![image](https://github.com/user-attachments/assets/88256d70-5c81-421f-a0d3-13dce21a05d9)



## Project Structure

- `rag.py`: Core RAG implementation with TravelAssistant class
- `main.py`: CLI interface demonstrating backend functionality
- `app.py`: Streamlit web interface for user interaction
- `Journey_Details.json`: Sample travel information (not included)

## Technical Requirements

- Python 3.9+
- Streamlit
- Pinecone Vector Database
- Jina AI Embeddings
- Hugging Face's LLaMA 3

## Local Development Setup

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd travel-assistant-rag
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure API keys in `.streamlit/secrets.toml`:
    ```toml
    PINECONE_API_KEY = "your-pinecone-key"
    JINA_API_KEY = "your-jina-key"
    HF_TOKEN = "your-huggingface-token"
    ```

4. Run the application:
    ```bash
    streamlit run app.py


## Architecture Overview

1. **Data Processing**: JSON travel documents are split and vectorized using Jina AI embeddings
2. **Vector Storage**: Embeddings are stored in Pinecone for efficient similarity search
3. **Query Processing**: User inputs are processed to find relevant context
4. **Response Generation**: LLaMA 3 generates natural language responses using retrieved context
5. **User Interface**: Streamlit provides an interactive chat experience
