"""
Faculty information chatbot that uses LangChain, Ollama and multiple language-specific
ChromaDB vector stores to answer questions about faculty information.
"""

import os
import logging
from typing import List, Dict
import langid

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
VECTORSTORE_DIR_AR = os.path.join(os.path.dirname(__file__), "vectorstore_ar")
VECTORSTORE_DIR_FR = os.path.join(os.path.dirname(__file__), "vectorstore_fr")
VECTORSTORE_DIR_ENG = os.path.join(os.path.dirname(__file__), "vectorstore_eng")
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
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url
            )
            
            # Load vector stores for each language
            self.vectorstores = {}
            self.retrievers = {}
            
            # Arabic vector store
            if os.path.exists(VECTORSTORE_DIR_AR):
                self.vectorstores['ar'] = Chroma(
                    persist_directory=VECTORSTORE_DIR_AR, 
                    embedding_function=self.embeddings
                )
                self.retrievers['ar'] = self.vectorstores['ar'].as_retriever(
                    search_kwargs={"k": self.retriever_k}
                )
            
            # French vector store
            if os.path.exists(VECTORSTORE_DIR_FR):
                self.vectorstores['fr'] = Chroma(
                    persist_directory=VECTORSTORE_DIR_FR, 
                    embedding_function=self.embeddings
                )
                self.retrievers['fr'] = self.vectorstores['fr'].as_retriever(
                    search_kwargs={"k": self.retriever_k}
                )
            
            # English vector store
            if os.path.exists(VECTORSTORE_DIR_ENG):
                self.vectorstores['en'] = Chroma(
                    persist_directory=VECTORSTORE_DIR_ENG, 
                    embedding_function=self.embeddings
                )
                self.retrievers['en'] = self.vectorstores['en'].as_retriever(
                    search_kwargs={"k": self.retriever_k}
                )
            
            # Default vector store (fallback)
            if not self.retrievers:
                raise ValueError("No vector stores found. Please run ingest.py first.")
            
            # Set up LLM
            self.llm = ChatOllama(
                model=self.chat_model,
                temperature=0.2,
                base_url=self.base_url
            )
            
            # Create prompt templates for each language
            self.prompts = {
                'fr': ChatPromptTemplate.from_template("""
                Vous êtes CampusIA, l'assistant intelligent de la Faculté des Sciences de Tunis. 🎓
                Vous aidez les étudiants pour:
                1. Trouver des salles et comprendre les procédures administratives 🗺️📋
                2. Retrouver des objets perdus grâce à une analyse intelligente 🔍
                
                Répondez naturellement en utilisant ces informations:
                {context}
                
                Conversation précédente:
                {chat_history}
                
                Élève: {question}
                
                Conseils de réponse:
                - Sois amical et utilise des emojis pertinents (mais pas trop)
                - Pour les localisations: Donne le bâtiment, étage, et points de repères proches
                - Pour l'administratif: Liste les étapes claires avec documents nécessaires
                - Pour les objets perdus: Pose 1-2 questions courtes si besoin de détails
                - Si incertain, propose de contacter le secrétariat ou la scolarité
                - Jamais de mention 'd'après les documents' ou 'dans mes données'
                
                CampusIA:"""),
                
                'en': ChatPromptTemplate.from_template("""
                You are CampusIA, the intelligent assistant of the Faculty of Sciences of Tunis. 🎓
                You help students with:
                1. Finding rooms and understanding administrative procedures 🗺️📋
                2. Finding lost objects through intelligent analysis 🔍
                
                Answer naturally using this information:
                {context}
                
                Previous conversation:
                {chat_history}
                
                Student: {question}
                
                Response tips:
                - Be friendly and use relevant emojis (but not too many)
                - For locations: Give the building, floor, and nearby landmarks
                - For administrative matters: List clear steps with necessary documents
                - For lost objects: Ask 1-2 short questions if details are needed
                - If uncertain, suggest contacting the secretariat or student affairs
                - Never mention 'according to documents' or 'in my data'
                
                CampusIA:"""),
                
                'ar': ChatPromptTemplate.from_template("""
                أنت كامبس آي أيه، المساعد الذكي لكلية العلوم بتونس. 🎓
                أنت تساعد الطلاب في:
                1. العثور على القاعات وفهم الإجراءات الإدارية 🗺️📋
                2. العثور على الأشياء المفقودة من خلال التحليل الذكي 🔍
                
                أجب بشكل طبيعي باستخدام هذه المعلومات:
                {context}
                
                المحادثة السابقة:
                {chat_history}
                
                الطالب: {question}
                
                نصائح للإجابة:
                - كن ودودًا واستخدم الرموز التعبيرية المناسبة (ولكن ليس بكثرة)
                - للمواقع: أعط المبنى والطابق والمعالم القريبة
                - للمسائل الإدارية: اذكر خطوات واضحة مع المستندات اللازمة
                - للأشياء المفقودة: اطرح سؤالاً أو سؤالين قصيرين إذا كانت هناك حاجة لمزيد من التفاصيل
                - إذا كنت غير متأكد، اقترح التواصل مع السكرتارية أو شؤون الطلبة
                - لا تذكر أبدًا عبارات مثل 'وفقًا للوثائق' أو 'في بياناتي'
                
                كامبس آي أيه:""")
            }
            
            # Create chains for each language
            self.chains = {}
            for lang, prompt in self.prompts.items():
                if lang in self.retrievers:
                    self.chains[lang] = (
                        {
                            "context": self.retrievers[lang], 
                            "question": RunnablePassthrough(),
                            "chat_history": lambda _: self.get_chat_history_str()
                        }
                        | prompt
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
    
    async def get_relevant_documents(self, query, language='fr'):
        """Retrieve relevant documents for a query in the specified language."""
        try:
            # If we don't have a retriever for this language, fall back to French or first available
            if language not in self.retrievers:
                language = 'fr' if 'fr' in self.retrievers else next(iter(self.retrievers))
                logger.warning(f"No retriever for language {language}, falling back to {language}")
                
            docs = self.retrievers[language].invoke(query)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def detect_language(self, text):
        """Detect the language of the given text using langid library.
        
        Args:
            text: The text to analyze for language detection
            
        Returns:
            String indicating the detected language code (e.g., 'fr', 'en', 'ar')
        """
        try:
            lang, confidence = langid.classify(text)
            logger.info(f"Detected language: {lang} (confidence: {confidence:.2f})")
            
            # Map the detected language to one of our supported languages
            if lang in self.retrievers:
                return lang
            
            # Default mappings for languages we might get but don't have exact vectorstores for
            language_map = {
                'ar': 'ar',  # Arabic
                'fr': 'fr',  # French
                'en': 'en',  # English
                'fa': 'ar',  # Persian -> Arabic
                'ur': 'ar',  # Urdu -> Arabic
                # Add more mappings as needed
            }
            
            if lang in language_map and language_map[lang] in self.retrievers:
                mapped_lang = language_map[lang]
                logger.info(f"Mapped language {lang} to {mapped_lang}")
                return mapped_lang
            
            # Default to French if available, otherwise first available language
            default_lang = 'fr' if 'fr' in self.retrievers else next(iter(self.retrievers), 'en')
            logger.info(f"Using default language: {default_lang}")
            return default_lang
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            # Default to French if available, otherwise first language
            return 'fr' if 'fr' in self.retrievers else next(iter(self.retrievers), 'en')
    
    async def ask(self, question):
        """Ask a question about faculty information."""
        try:
            # Log the question
            logger.info(f"Question: {question}")
            
            # Detect language of the question
            language = self.detect_language(question)
            logger.info(f"Using language: {language}")
            
            # If we don't have a chain for this language, fall back to French or first available
            if language not in self.chains:
                language = 'fr' if 'fr' in self.chains else next(iter(self.chains))
                logger.warning(f"No chain for language {language}, falling back to {language}")
            
            # For debug mode, print retrieved documents
            if self.debug_mode:
                docs = await self.get_relevant_documents(question, language)
                context = "\n\n".join([doc.page_content for doc in docs])
                logger.info(f"Retrieved {len(docs)} documents from {language} vector store")
                
                # Generate response using the chain
                response = await self.chains[language].ainvoke(question)
            else:
                # Use the chain directly (it handles retrieval internally)
                response = await self.chains[language].ainvoke(question)
                
            # Add to chat history
            self.add_message_to_history(question, response)
            return response
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def get_stats(self):
        """Get basic statistics about the vector stores."""
        try:
            stats = {}
            for lang, vectorstore in self.vectorstores.items():
                collection = vectorstore._collection
                stats[lang] = {
                    "document_count": collection.count(),
                    "collection_name": collection.name
                }
            return stats
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
            for lang, stat in stats.items():
                print(f"Language: {lang}, Documents: {stat.get('document_count', 'unknown')}")
            
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