# import streamlit as st
# import os
# import tempfile
# from typing import List, Tuple
# import re

# # Try to import required packages
# try:
#     from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     from langchain.embeddings import HuggingFaceEmbeddings
#     from langchain.vectorstores import FAISS
#     from langchain.schema import Document
#     from langchain.llms import GPT4All
#     from langchain.chains import RetrievalQA
#     from langchain.prompts import PromptTemplate
#     from langchain.callbacks.manager import CallbackManager
#     from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#     LANGCHAIN_AVAILABLE = True
# except ImportError as e:
#     st.error(f"Missing packages: {e}")
#     LANGCHAIN_AVAILABLE = False

# class ChatGPTStyleBot:
#     def __init__(self):
#         self.vector_store = None
#         self.embeddings = None
#         self.llm = None
#         self.qa_chain = None
        
#     def setup_embeddings(self):
#         """Setup embeddings model"""
#         if not LANGCHAIN_AVAILABLE:
#             return False
#         try:
#             self.embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",
#                 model_kwargs={'device': 'cpu'}
#             )
#             return True
#         except Exception as e:
#             st.error(f"Embeddings error: {e}")
#             return False
    
#     def setup_llm(self):
#         """Setup local LLM for ChatGPT-like responses"""
#         try:
#             # Try to find model file
#             model_path = self._find_model_file()
#             if not model_path:
#                 st.info("🤖 **No LLM model found.** Using smart extraction mode. For ChatGPT-like responses, download a model.")
#                 return False
            
#             callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
#             self.llm = GPT4All(
#                 model=model_path,
#                 callback_manager=callback_manager,
#                 verbose=False,
#                 n_ctx=2048,
#                 temp=0.3,  # Slightly higher temperature for creative responses
#                 top_p=0.9,
#                 n_threads=6
#             )
#             return True
            
#         except Exception as e:
#             st.warning(f"LLM setup failed: {e}. Using smart extraction.")
#             return False
    
#     def _find_model_file(self):
#         """Find LLM model file"""
#         possible_paths = [
#             "orca-mini-3b.ggmlv3.q4_0.bin",
#             "gpt4all-lora-quantized.bin",
#             os.path.expanduser("~/.local_rag_chatbot/models/orca-mini-3b.ggmlv3.q4_0.bin"),
#             "./models/orca-mini-3b.ggmlv3.q4_0.bin",
#             "./models/gpt4all-lora-quantized.bin"
#         ]
        
#         for path in possible_paths:
#             if os.path.exists(path):
#                 return path
#         return None
    
#     def create_vector_store(self, documents):
#         """Create vector store from documents"""
#         if not LANGCHAIN_AVAILABLE:
#             return False
            
#         if not self.embeddings:
#             if not self.setup_embeddings():
#                 return False
        
#         try:
#             self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
#             # Setup LLM after vector store is ready
#             llm_available = self.setup_llm()
            
#             if llm_available and self.llm:
#                 # Create ChatGPT-style QA chain
#                 self._setup_qa_chain()
#                 st.success("🎉 **ChatGPT-style responses enabled!**")
#             else:
#                 st.info("🔍 **Smart extraction mode** - Answers will be synthesized from your documents")
            
#             return True
#         except Exception as e:
#             st.error(f"Vector store error: {e}")
#             return False
    
#     def _setup_qa_chain(self):
#         """Setup QA chain with ChatGPT-style prompting"""
#         # Custom prompt for better responses
#         custom_prompt = PromptTemplate(
#             template="""You are a helpful AI assistant. Use the following context to answer the user's question in a clear, conversational, and helpful manner.

# Context: {context}

# Question: {question}

# Please provide a comprehensive answer based on the context. If the context doesn't contain enough information, say so. Format your response in a natural, conversational way.

# Answer: """,
#             input_variables=["context", "question"]
#         )
        
#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             chain_type="stuff",
#             retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
#             return_source_documents=True,
#             chain_type_kwargs={"prompt": custom_prompt}
#         )
    
#     def get_answer(self, question: str) -> Tuple[str, List]:
#         """Get ChatGPT-style answer"""
#         if not self.vector_store:
#             return "❌ Please process documents first. Upload files and click 'Process Documents'.", []
        
#         try:
#             # If LLM is available, use it for generated responses
#             if self.qa_chain and self.llm:
#                 return self._get_llm_answer(question)
#             else:
#                 # Fallback to smart extraction
#                 return self._get_smart_answer(question)
            
#         except Exception as e:
#             return f"❌ Error: {str(e)}", []

#     def _get_llm_answer(self, question: str) -> Tuple[str, List]:
#         """Get answer using LLM for ChatGPT-style responses"""
#         result = self.qa_chain({"query": question})
        
#         # Clean up the response
#         answer = result["result"].strip()
#         sources = result.get("source_documents", [])
        
#         return answer, sources

#     def _get_smart_answer(self, question: str) -> Tuple[str, List]:
#         """Smart extraction when no LLM is available"""
#         relevant_docs = self.vector_store.similarity_search(question, k=3)
        
#         if not relevant_docs:
#             return "🤔 I couldn't find specific information about this in your documents. Could you try rephrasing your question or ask about a different topic?", []
        
#         # Create a ChatGPT-style response from the documents
#         answer = self._create_conversational_answer(question, relevant_docs)
#         return answer, relevant_docs

#     def _create_conversational_answer(self, question: str, relevant_docs: List) -> str:
#         """Create a conversational answer from document content"""
        
#         # Extract key information
#         key_points = []
#         for doc in relevant_docs:
#             content = doc.page_content.strip()
#             # Extract the most relevant sentence
#             sentences = content.split('. ')
#             if sentences:
#                 # Find sentence most relevant to the question
#                 question_words = set(question.lower().split())
#                 best_sentence = sentences[0]  # Default to first sentence
#                 best_score = 0
                
#                 for sentence in sentences:
#                     sentence_lower = sentence.lower()
#                     score = sum(1 for word in question_words if word in sentence_lower and len(word) > 3)
#                     if score > best_score:
#                         best_score = score
#                         best_sentence = sentence
                
#                 if best_sentence and len(best_sentence) > 20:
#                     key_points.append(best_sentence.strip() + '.')
        
#         # Remove duplicates and limit length
#         unique_points = []
#         seen_points = set()
#         for point in key_points:
#             point_hash = hash(point[:100])  # Simple deduplication
#             if point_hash not in seen_points and len(unique_points) < 3:
#                 unique_points.append(point)
#                 seen_points.add(point_hash)
        
#         if not unique_points:
#             return "🤔 I found some documents but couldn't extract a clear answer. Could you provide more context or ask a more specific question?"
        
#         # Build conversational response
#         if "what is" in question.lower() or "define" in question.lower():
#             response = f"Based on your documents, **{question}** can be understood as:\n\n"
#         elif "how" in question.lower():
#             response = f"Here's how this works according to your documents:\n\n"
#         else:
#             response = f"Here's what I found about **{question}** in your documents:\n\n"
        
#         # Add the key points in a natural way
#         for i, point in enumerate(unique_points):
#             response += f"• {point}\n"
        
#         response += f"\nThis information is synthesized from your uploaded documents."
        
#         return response

# def process_uploaded_files(uploaded_files):
#     """Process uploaded files and return documents"""
#     if not LANGCHAIN_AVAILABLE:
#         return []
    
#     all_documents = []
#     processed_files = []
#     failed_files = []
    
#     # Initialize text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
    
#     for uploaded_file in uploaded_files:
#         try:
#             file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_path = tmp_file.name
            
#             try:
#                 if file_extension == '.pdf':
#                     loader = PyPDFLoader(tmp_path)
#                 elif file_extension == '.txt':
#                     loader = TextLoader(tmp_path, encoding='utf-8')
#                 elif file_extension in ['.docx', '.doc']:
#                     loader = Docx2txtLoader(tmp_path)
#                 else:
#                     failed_files.append(f"{uploaded_file.name}: Unsupported format")
#                     continue
                
#                 # Load and split documents
#                 documents = loader.load()
#                 chunks = text_splitter.split_documents(documents)
#                 all_documents.extend(chunks)
#                 processed_files.append(uploaded_file.name)
                
#             except Exception as e:
#                 failed_files.append(f"{uploaded_file.name}: {str(e)}")
#             finally:
#                 os.unlink(tmp_path)
                
#         except Exception as e:
#             failed_files.append(f"{uploaded_file.name}: {str(e)}")
    
#     # Show processing results
#     if processed_files:
#         st.success(f"✅ Successfully processed: {len(processed_files)} files")
#     if failed_files:
#         for failed in failed_files:
#             st.error(f"❌ {failed}")
    
#     return all_documents

# def download_llm_model():
#     """Download a local LLM model for ChatGPT-like responses"""
#     import requests
#     import urllib.request
    
#     st.info("📥 Downloading LLM model for ChatGPT-like responses...")
    
#     model_url = "https://gpt4all.io/models/ggml-orca-mini-3b.q4_0.bin"
#     model_path = "orca-mini-3b.ggmlv3.q4_0.bin"
    
#     try:
#         # Create models directory
#         os.makedirs("models", exist_ok=True)
#         model_path = os.path.join("models", "orca-mini-3b.ggmlv3.q4_0.bin")
        
#         # Download with progress
#         urllib.request.urlretrieve(model_url, model_path)
#         st.success("✅ Model downloaded successfully! Restart the app to use ChatGPT-style responses.")
#         return True
#     except Exception as e:
#         st.error(f"❌ Download failed: {e}")
#         return False

# def main():
#     st.set_page_config(
#         page_title="ChatGPT-Style RAG Chatbot",
#         page_icon="🤖",
#         layout="wide"
#     )
    
#     st.title("💬 ChatGPT-Style RAG Chatbot")
#     st.markdown("**Ask questions and get intelligent responses from your documents**")
    
#     # Initialize session state
#     if 'chatbot' not in st.session_state:
#         st.session_state.chatbot = ChatGPTStyleBot()
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'documents_processed' not in st.session_state:
#         st.session_state.documents_processed = False
    
#     # Sidebar
#     with st.sidebar:
#         st.header("🤖 Response Mode")
        
#         # Check if LLM is available
#         llm_available = st.session_state.chatbot.llm is not None
        
#         if llm_available:
#             st.success("🎉 **ChatGPT Mode** - Intelligent responses")
#         else:
#             st.info("🔍 **Smart Mode** - Document-based answers")
#             if st.button("📥 Download LLM Model", use_container_width=True):
#                 download_llm_model()
        
#         st.markdown("---")
#         st.header("📁 Upload Documents")
        
#         # Check system status
#         if LANGCHAIN_AVAILABLE:
#             st.success("✅ System Ready")
#         else:
#             st.error("❌ Install packages:")
#             st.code("pip install langchain sentence-transformers faiss-cpu pypdf2 python-docx docx2txt gpt4all")
        
#         uploaded_files = st.file_uploader(
#             "Choose documents",
#             type=['txt', 'pdf', 'docx', 'doc'],
#             accept_multiple_files=True,
#             help="Upload PDF, TXT, or Word documents"
#         )
        
#         if uploaded_files:
#             st.info(f"📄 {len(uploaded_files)} files selected")
            
#             if st.button("🚀 Process Documents", type="primary", use_container_width=True):
#                 if not LANGCHAIN_AVAILABLE:
#                     st.error("Please install required packages first")
#                 else:
#                     with st.spinner("Processing documents..."):
#                         documents = process_uploaded_files(uploaded_files)
                        
#                         if documents:
#                             success = st.session_state.chatbot.create_vector_store(documents)
#                             if success:
#                                 st.session_state.documents_processed = True
#                                 st.session_state.messages = []
#                                 st.success(f"✅ Ready! Processed {len(documents)} text chunks")
#                                 st.balloons()
#                             else:
#                                 st.error("❌ Failed to create knowledge base")
#                         else:
#                             st.error("❌ No documents could be processed")
        
#         st.markdown("---")
#         st.header("⚙️ Controls")
        
#         if st.button("🗑️ Clear Chat", use_container_width=True):
#             st.session_state.messages = []
#             st.rerun()
            
#         if st.button("🔄 Reset All", use_container_width=True):
#             for key in list(st.session_state.keys()):
#                 del st.session_state[key]
#             st.rerun()
    
#     # Main chat area
#     st.header("💬 Conversation")
    
#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
            
#             # Show sources for assistant messages if available
#             if message["role"] == "assistant" and message.get("sources"):
#                 with st.expander("📚 View Source Documents"):
#                     for i, source in enumerate(message["sources"]):
#                         st.markdown(f"**Source {i+1}:**")
#                         st.text(source.page_content[:500] + "..." if len(source.page_content) > 500 else source.page_content)
#                         st.markdown("---")
    
#     # Chat input
#     if prompt := st.chat_input("Ask me anything about your documents..."):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Display user message immediately
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate and display assistant response
#         with st.chat_message("assistant"):
#             if not st.session_state.documents_processed:
#                 st.error("Please upload and process documents first!")
#             else:
#                 with st.spinner("Thinking..."):
#                     try:
#                         # Get answer from chatbot
#                         answer, sources = st.session_state.chatbot.get_answer(prompt)
                        
#                         # Display the answer
#                         st.markdown(answer)
                        
#                         # Add to chat history
#                         st.session_state.messages.append({
#                             "role": "assistant", 
#                             "content": answer,
#                             "sources": sources
#                         })
                        
#                     except Exception as e:
#                         error_msg = f"Error: {str(e)}"
#                         st.error(error_msg)
#                         st.session_state.messages.append({
#                             "role": "assistant", 
#                             "content": error_msg,
#                             "sources": []
#                         })

# if __name__ == "__main__":
#     main()























import streamlit as st
import os
import tempfile
from typing import List, Tuple
import re
import sys

# Try to import required packages
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_community.llms.gpt4all import GPT4All
    from langchain.chains import RetrievalQA
    from langchain.chains import ConversationalRetrievalChain
    from langchain_core.prompts import PromptTemplate
    from langchain_core.callbacks import CallbackManager
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
    from langchain.chains.retrieval_qa import RetrievalQA
    from langchain.chains.conversational_retrieval import ConversationalRetrievalChain


    

    LANGCHAIN_AVAILABLE = True
except ImportError as e:  # Fixed exception
    print(f"Import Error: {e}")
    LANGCHAIN_AVAILABLE = False

class ChatGPTStyleBot:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.qa_chain = None
        
    def setup_embeddings(self):
        """Setup embeddings model"""
        if not LANGCHAIN_AVAILABLE:
            return False
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            return True
        except Exception as e:
            st.error(f"Embeddings error: {e}")
            return False
    
    def setup_llm(self):
        """Setup local LLM for ChatGPT-like responses"""
        try:
            # Try to find model file
            model_path = self._find_model_file()
            if not model_path:
                return False
            
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            self.llm = GPT4All(
                model=model_path,
                callback_manager=callback_manager,
                verbose=False,
                n_ctx=2048,
                temp=0.3,
                top_p=0.9,
                n_threads=6
            )
            return True
            
        except Exception as e:
            return False
    
    def _find_model_file(self):
        """Find LLM model file"""
        possible_paths = [
            "orca-mini-3b.ggmlv3.q4_0.bin",
            "gpt4all-lora-quantized.bin",
            os.path.expanduser("~/.local_rag_chatbot/models/orca-mini-3b.ggmlv3.q4_0.bin"),
            "./models/orca-mini-3b.ggmlv3.q4_0.bin",
            "./models/gpt4all-lora-quantized.bin"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def create_vector_store(self, documents):
        """Create vector store from documents"""
        if not LANGCHAIN_AVAILABLE:
            return False
            
        if not self.embeddings:
            if not self.setup_embeddings():
                return False
        
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Setup LLM after vector store is ready
            llm_available = self.setup_llm()
            
            if llm_available and self.llm:
                self._setup_qa_chain()
                return True
            else:
                return True
            
        except Exception as e:
            st.error(f"Vector store error: {e}")
            return False
    
    def _setup_qa_chain(self):
        """Setup QA chain with ChatGPT-style prompting"""
        custom_prompt = PromptTemplate(
            template="""You are a helpful AI assistant. Use the following context to answer the user's question in a clear, conversational, and helpful manner.

Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context. If the context doesn't contain enough information, say so. Format your response in a natural, conversational way.

Answer: """,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt}
        )
    
    def get_answer(self, question: str) -> Tuple[str, List]:
        """Get ChatGPT-style answer"""
        if not self.vector_store:
            return "❌ Please process documents first. Upload files and click 'Process Documents'.", []
        
        try:
            if self.qa_chain and self.llm:
                return self._get_llm_answer(question)
            else:
                return self._get_smart_answer(question)
            
        except Exception as e:
            return f"❌ Error: {str(e)}", []

    def _get_llm_answer(self, question: str) -> Tuple[str, List]:
        """Get answer using LLM for ChatGPT-style responses"""
        result = self.qa_chain({"query": question})
        answer = result["result"].strip()
        sources = result.get("source_documents", [])
        return answer, sources

    def _get_smart_answer(self, question: str) -> Tuple[str, List]:
        """Smart extraction when no LLM is available"""
        relevant_docs = self.vector_store.similarity_search(question, k=3)
        
        if not relevant_docs:
            return "🤔 I couldn't find specific information about this in your documents. Could you try rephrasing your question or ask about a different topic?", []
        
        answer = self._create_conversational_answer(question, relevant_docs)
        return answer, relevant_docs

    def _create_conversational_answer(self, question: str, relevant_docs: List) -> str:
        """Create a conversational answer from document content"""
        key_points = []
        for doc in relevant_docs:
            content = doc.page_content.strip()
            sentences = content.split('. ')
            if sentences:
                question_words = set(question.lower().split())
                best_sentence = sentences[0]
                best_score = 0
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    score = sum(1 for word in question_words if word in sentence_lower and len(word) > 3)
                    if score > best_score:
                        best_score = score
                        best_sentence = sentence
                
                if best_sentence and len(best_sentence) > 20:
                    key_points.append(best_sentence.strip() + '.')
        
        unique_points = []
        seen_points = set()
        for point in key_points:
            point_hash = hash(point[:100])
            if point_hash not in seen_points and len(unique_points) < 3:
                unique_points.append(point)
                seen_points.add(point_hash)
        
        if not unique_points:
            return "🤔 I found some documents but couldn't extract a clear answer. Could you provide more context or ask a more specific question?"
        
        if "what is" in question.lower() or "define" in question.lower():
            response = f"Based on your documents, **{question}** can be understood as:\n\n"
        elif "how" in question.lower():
            response = f"Here's how this works according to your documents:\n\n"
        else:
            response = f"Here's what I found about **{question}** in your documents:\n\n"
        
        for point in unique_points:
            response += f"• {point}\n"
        
        response += f"\nThis information is synthesized from your uploaded documents."
        return response

def process_uploaded_files(uploaded_files):
    """Process uploaded files and return documents"""
    if not LANGCHAIN_AVAILABLE:
        return []
    
    all_documents = []
    processed_files = []
    failed_files = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    for uploaded_file in uploaded_files:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                if file_extension == '.pdf':
                    loader = PyPDFLoader(tmp_path)
                elif file_extension == '.txt':
                    loader = TextLoader(tmp_path, encoding='utf-8')
                elif file_extension in ['.docx', '.doc']:
                    loader = Docx2txtLoader(tmp_path)
                else:
                    failed_files.append(f"{uploaded_file.name}: Unsupported format")
                    continue
                
                documents = loader.load()
                chunks = text_splitter.split_documents(documents)
                all_documents.extend(chunks)
                processed_files.append(uploaded_file.name)
                
            except Exception as e:
                failed_files.append(f"{uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            failed_files.append(f"{uploaded_file.name}: {str(e)}")
    
    if processed_files:
        st.success(f"✅ Successfully processed: {len(processed_files)} files")
    if failed_files:
        for failed in failed_files:
            st.error(f"❌ {failed}")
    
    return all_documents

def main():
    st.set_page_config(
        page_title="ChatGPT-Style RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for better visibility
    st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Bot message - LEFT side */
    .bot-message {
        background-color: #f0f8ff;
        padding: 16px;
        border-radius: 16px;
        margin: 8px 0;
        border-bottom-left-radius: 4px;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #e1f5fe;
    }
    
    /* User message - RIGHT side */
    .user-message {
        background-color: #e3f2fd;
        padding: 16px;
        border-radius: 16px;
        margin: 8px 0;
        border-bottom-right-radius: 4px;
        max-width: 80%;
        margin-left: auto;
        border: 1px solid #bbdefb;
    }
    
    /* Message containers */
    .message-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin: 10px 0;
    }
    
    .bot-container {
        align-items: flex-start;
    }
    
    .user-container {
        align-items: flex-end;
    }
    
    /* Make sure messages are visible */
    .stChatMessage {
        visibility: visible !important;
    }
    
    /* Chat input styling */
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 70%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("💬 ChatGPT-Style RAG Chatbot")
    st.markdown("**Ask questions and get intelligent responses from your documents**")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatGPTStyleBot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 System Status")
        
        llm_available = st.session_state.chatbot.llm is not None
        if llm_available:
            st.success("🎉 ChatGPT Mode Enabled")
        else:
            st.info("🔍 Smart Extraction Mode")
        
        st.markdown("---")
        st.header("📁 Upload Documents")
        
        if LANGCHAIN_AVAILABLE:
            st.success("✅ System Ready")
        else:
            st.error("❌ Install packages first")
        
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=['txt', 'pdf', 'docx', 'doc'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} files selected")
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                if not LANGCHAIN_AVAILABLE:
                    st.error("Please install required packages first")
                else:
                    with st.spinner("Processing documents..."):
                        documents = process_uploaded_files(uploaded_files)
                        
                        if documents:
                            success = st.session_state.chatbot.create_vector_store(documents)
                            if success:
                                st.session_state.documents_processed = True
                                st.session_state.messages = []
                                st.success(f"✅ Ready! Processed {len(documents)} text chunks")
                                st.balloons()
                            else:
                                st.error("❌ Failed to create knowledge base")
                        else:
                            st.error("❌ No documents could be processed")
        
        st.markdown("---")
        st.header("⚙️ Controls")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main chat area - SIMPLIFIED DISPLAY LOGIC
    st.header("💬 Conversation")
    
    # Display all messages - SIMPLE AND RELIABLE
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            # Bot message on LEFT
            with st.container():
                col1, col2 = st.columns([8, 2])
                with col1:
                    st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
                    
                    # Show sources if available
                    if message.get("sources"):
                        with st.expander("📚 Source Documents"):
                            for j, source in enumerate(message["sources"]):
                                st.markdown(f"**Source {j+1}:**")
                                st.text(source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content)
                                st.markdown("---")
        
        else:
            # User message on RIGHT
            with st.container():
                col1, col2 = st.columns([2, 8])
                with col2:
                    st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input - SIMPLIFIED LOGIC
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate bot response
        if not st.session_state.documents_processed:
            error_msg = "❌ Please upload and process documents first! Go to the sidebar, upload your files, and click 'Process Documents'."
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    answer, sources = st.session_state.chatbot.get_answer(prompt)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "sources": []
                    })
        
        # Force rerun to display new messages
        st.rerun()

if __name__ == "__main__":
    main()