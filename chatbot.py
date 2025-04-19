"""
Faculty information chatbot module that uses LangChain, Ollama and ChromaDB
to answer questions about faculty information.
"""

import os
from typing import List, Dict, Any

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage

# Define paths
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

# Define Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "gemma3:4b"  # Model for chat/completions
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # Model for embeddings

class FacultyInfoChatbot:
    """
    Chatbot for answering questions about faculty information.
    Uses Ollama for LLM capabilities and ChromaDB for vector retrieval.
    """
    
    def __init__(self, 
                 chat_model=OLLAMA_CHAT_MODEL,
                 embedding_model=OLLAMA_EMBEDDING_MODEL,
                 base_url=OLLAMA_BASE_URL):
        """
        Initialize the faculty information chatbot with the specified models.
        
        Args:
            chat_model: The Ollama model to use for chat completions
            embedding_model: The Ollama model to use for embeddings
            base_url: The base URL for the Ollama API
        """
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.chat_history = []
        self.setup_retrieval_chain()
    
    def setup_retrieval_chain(self):
        """Set up the retrieval chain for answering questions."""
        # Load the persisted vector store using the embedding model
        print(f"Loading vector store from {VECTORSTORE_DIR}")
        print(f"Using {self.embedding_model} for embeddings")
        embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.base_url
        )
        self.vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
        
        # Create a retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create an LLM using the chat model
        print(f"Using {self.chat_model} for chat completions")
        self.llm = ChatOllama(
            model=self.chat_model,
            temperature=0.2,
            base_url=self.base_url
        )
        
        # Create a prompt template
        template = """
        You are a helpful assistant specializing in answering questions about university faculty members.
        Use the following context to answer the question at the end. If you don't know the answer, just say that 
        you don't know, don't try to make up an answer.
        
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Helpful Answer:"""
        
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain
        self.chain = (
            {
                "context": self.retriever, 
                "question": RunnablePassthrough(),
                "chat_history": lambda _: self.get_chat_history_str()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def get_chat_history_str(self) -> str:
        """Format chat history into a string for the prompt."""
        return "\n".join(
            f"Human: {human}\nAI: {ai}" 
            for human, ai in self.chat_history
        )
    
    def add_message_to_history(self, user_message: str, assistant_response: str):
        """Add the user message and assistant response to chat history."""
        self.chat_history.append((user_message, assistant_response))
        # Keep only the last 5 exchanges to avoid context length issues
        self.chat_history = self.chat_history[-5:]
    
    async def ask(self, question: str) -> str:
        """
        Ask a question about faculty information.
        
        Args:
            question: The user's question about faculty
            
        Returns:
            The chatbot's answer
        """
        # Use ainvoke for async operation
        response = await self.chain.ainvoke(question)
        self.add_message_to_history(question, response)
        return response


# Simple example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        chatbot = FacultyInfoChatbot()
        while True:
            question = input("\nQuestion (type 'exit' to quit): ")
            if question.lower() == "exit":
                break
            response = await chatbot.ask(question)
            print(f"\nAnswer: {response}")
    
    asyncio.run(main())