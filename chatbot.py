"""
Faculty information chatbot that uses LangChain, Ollama and ChromaDB
to answer questions about faculty information.
"""

import os
import logging
from typing import List

# Replace these imports
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma

# With these:
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FacultyInfoChatbot')

# Define paths and settings
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "gemma3:4b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

class FacultyInfoChatbot:
    """Chatbot for answering questions about faculty information."""
    
    def __init__(self, 
                 chat_model=OLLAMA_CHAT_MODEL,
                 embedding_model=OLLAMA_EMBEDDING_MODEL,
                 base_url=OLLAMA_BASE_URL,
                 retriever_k=5,
                 debug_mode=False):
        """Initialize the faculty information chatbot."""
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.retriever_k = retriever_k
        self.debug_mode = debug_mode
        self.chat_history = []
        self.setup_chain()
    
    def setup_chain(self):
        """Set up the retrieval chain for answering questions."""
        try:
            # Set up embeddings
            embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url
            )
            
            # Load vector store
            self.vectorstore = Chroma(
                persist_directory=VECTORSTORE_DIR, 
                embedding_function=embeddings
            )
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.retriever_k}
            )
            
            # Set up LLM
            self.llm = ChatOllama(
                model=self.chat_model,
                temperature=0.2,
                base_url=self.base_url
            )
            
            # Create prompt template
            template = """
            You are a helpful assistant specializing in answering questions about the Faculty of Sciences of Tunis (FST).
            Use the following context to answer the question. If you don't have enough information, clearly state that.
            
            Context information:
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
            
        except Exception as e:
            logger.error(f"Failed to set up chain: {e}")
            raise
    
    def get_chat_history_str(self):
        """Format chat history into a string for the prompt."""
        return "\n".join(
            f"Human: {human}\nAI: {ai}" 
            for human, ai in self.chat_history
        )
    
    def add_message_to_history(self, user_message, assistant_response):
        """Add the conversation to chat history."""
        self.chat_history.append((user_message, assistant_response))
        # Keep only the last 5 exchanges
        self.chat_history = self.chat_history[-5:]
    
    async def get_relevant_documents(self, query):
        """Retrieve relevant documents for a query."""
        try:
            docs = self.retriever.invoke(query)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def ask(self, question):
        """Ask a question about faculty information."""
        try:
            # Log the question
            logger.info(f"Question: {question}")
            
            # For debug mode, print retrieved documents
            if self.debug_mode:
                docs = await self.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs])
                logger.info(f"Retrieved {len(docs)} documents")
                
                # Generate response using the chain
                response = await self.chain.ainvoke(question)
            else:
                # Use the chain directly (it handles retrieval internally)
                response = await self.chain.ainvoke(question)
            
            # Add to chat history
            self.add_message_to_history(question, response)
            return response
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def get_stats(self):
        """Get basic statistics about the vector store."""
        try:
            collection = self.vectorstore._collection
            return {
                "document_count": collection.count(),
                "collection_name": collection.name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        try:
            # Initialize chatbot
            chatbot = FacultyInfoChatbot(debug_mode=True)
            
            # Print stats
            print("\nVector Store Stats:")
            stats = chatbot.get_stats()
            print(f"Documents: {stats.get('document_count', 'unknown')}")
            
            # Interactive query loop
            while True:
                question = input("\nQuestion (type 'exit' to quit): ")
                if question.lower() == "exit":
                    break
                elif question.lower() == "debug":
                    chatbot.debug_mode = not chatbot.debug_mode
                    print(f"Debug mode {'enabled' if chatbot.debug_mode else 'disabled'}")
                    continue
                
                # Get answer
                print("Thinking...")
                response = await chatbot.ask(question)
                print(f"\nAnswer: {response}")
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure Ollama is running and the vector store exists.")
    
    asyncio.run(main())