# Standard library imports
import json
import os

# Third-party imports
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.embeddings import JinaEmbeddings
from langchain_pinecone import PineconeVectorStore
import streamlit as st

class TravelAssistant:
    PROMPT = """
    You are a helpful travel assistant. Using the JSON travel information provided between <context> tags, answer the question between <question> tags.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    Instructions:
    - If the information contains flight details, include the flight number, departure/arrival times, and airports
    - If the requested information is not found in the context, say so clearly
    - Keep the response concise and focused on the question
    - If dates and times are mentioned, format them clearly so the user can understand them like this: 11 July 2024 at 14:35
    
    Please provide your response in a clear, natural language format.
    """

    def __init__(self):
        load_dotenv()
        self.pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        self.text_embeddings = JinaEmbeddings(
            jina_api_key=st.secrets["JINA_API_KEY"],
            model_name="jina-embeddings-v2-base-en"
        )
        self.index_name = "example-index2"
        self.setup_vector_store()

    def setup_vector_store(self):
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self._create_new_index()
        else:
            print(f"Index '{self.index_name}' already exists, connecting to existing index")
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.text_embeddings,
            )

    def _create_new_index(self):
        print(f"Creating new index: {self.index_name}")
        
        self.pc.create_index(
            name=self.index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Waiting for index to be ready...")

        with open("Journey_Details.json", 'r') as file:
            json_data = json.load(file)
        
        json_splitter = RecursiveJsonSplitter()
        chunks = json_splitter.split_json(json_data)
        texts = [json.dumps(chunk) for chunk in chunks]
        
        self.vector_store = PineconeVectorStore.from_texts(
            texts=texts,
            embedding=self.text_embeddings,
            index_name=self.index_name,
        )
        print(f"Added {len(texts)} documents to new index")

    def get_answer(self, query, k=3, score_threshold=0.7):
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Filter results based on similarity score
        relevant_results = [
            (doc, score) for doc, score in results 
            if score >= score_threshold
        ]
        
        if not relevant_results:
            return {
                'context': '',
                'similar_docs': [],
                'prompt': self.PROMPT.format(
                    context="No relevant information found in the travel documents.",
                    question=query
                )
            }
        
        # Use only relevant documents
        context = "\n".join([doc.page_content for doc, _ in relevant_results])
        return {
            'context': context,
            'similar_docs': [doc for doc, _ in relevant_results],
            'prompt': self.PROMPT.format(context=context, question=query)
        }