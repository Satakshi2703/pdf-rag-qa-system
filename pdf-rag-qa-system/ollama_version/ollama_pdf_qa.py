import os
import pickle
import requests
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import warnings

warnings.filterwarnings("ignore")

# === ENHANCED CONFIGURATION ===
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MAX_CONTEXT_LENGTH = 2000
SIMILARITY_THRESHOLD = 0.7

class Colors:
    """ANSI color codes for better terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# === IMPROVED OLLAMA CLIENT ===
class EnhancedOllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Check Ollama service status and return detailed info"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    'running': True,
                    'models': [m['name'] for m in models],
                    'model_available': any(self.model in m['name'] for m in models)
                }
            return {'running': False, 'models': [], 'model_available': False}
        except Exception as e:
            return {'running': False, 'error': str(e), 'models': [], 'model_available': False}
    
    def pull_model_with_progress(self) -> bool:
        """Pull model with enhanced progress tracking"""
        print(f"{Colors.OKCYAN}🔄 Pulling {self.model}... This may take several minutes.{Colors.ENDC}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                stream=True,
                timeout=600
            )
            
            last_status = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get('status', '')
                        
                        if 'completed' in data and 'total' in data:
                            progress = (data['completed'] / data['total']) * 100
                            bar_length = 30
                            filled_length = int(bar_length * data['completed'] // data['total'])
                            bar = '█' * filled_length + '-' * (bar_length - filled_length)
                            print(f"\r{Colors.OKBLUE}📥 [{bar}] {progress:.1f}%{Colors.ENDC}", end="", flush=True)
                        
                        elif status != last_status:
                            if status == 'success':
                                print(f"\n{Colors.OKGREEN}✅ Model {self.model} installed successfully!{Colors.ENDC}")
                                return True
                            elif status:
                                print(f"\n{Colors.WARNING}📦 {status}...{Colors.ENDC}")
                                last_status = status
                    except json.JSONDecodeError:
                        continue
            
            return True
            
        except Exception as e:
            print(f"\n{Colors.FAIL}❌ Error pulling model: {e}{Colors.ENDC}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 400) -> Optional[str]:
        """Generate response with improved error handling and retries"""
        if not prompt.strip():
            return None
            
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": max_tokens,
                "stop": ["\n\nQuestion:", "\n\nUser:", "Context:", "\n---"],
                "repeat_penalty": 1.1
            }
        }
        
        for attempt in range(3):
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '').strip()
                    
                    # Log performance metrics
                    duration = time.time() - start_time
                    tokens_per_sec = result.get('eval_count', 0) / duration if duration > 0 else 0
                    print(f"{Colors.OKCYAN}⚡ Response time: {duration:.1f}s | {tokens_per_sec:.1f} tokens/sec{Colors.ENDC}")
                    
                    return answer if answer else None
                    
                elif response.status_code == 404:
                    print(f"{Colors.WARNING}⚠️ Model not found. Attempting to pull...{Colors.ENDC}")
                    if self.pull_model_with_progress():
                        continue
                    return None
                else:
                    print(f"{Colors.FAIL}❌ API Error {response.status_code}: {response.text}{Colors.ENDC}")
                    
            except requests.exceptions.Timeout:
                print(f"{Colors.WARNING}⏰ Request timeout (attempt {attempt + 1}/3){Colors.ENDC}")
                if attempt == 2:
                    return None
                time.sleep(2)
            except Exception as e:
                print(f"{Colors.FAIL}❌ Generation error: {e}{Colors.ENDC}")
                if attempt == 2:
                    return None
                time.sleep(1)
        
        return None

# === ENHANCED DOCUMENT PROCESSING ===
class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True
        )
    
    def load_and_process_pdf(self, pdf_path: str) -> List[Any]:
        """Load PDF and process with enhanced cleaning"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"{Colors.OKBLUE}📄 Loading PDF: {Path(pdf_path).name}{Colors.ENDC}")
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        print(f"{Colors.OKGREEN}✅ Loaded {len(pages)} pages{Colors.ENDC}")
        
        # Enhanced text cleaning
        for page in pages:
            page.page_content = self.clean_text(page.page_content)
        
        documents = self.splitter.split_documents(pages)
        
        # Add metadata and clean chunks
        for i, doc in enumerate(documents):
            doc.page_content = doc.page_content.strip()
            if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                page_num = doc.metadata['page'] + 1
                doc.page_content = f"[Page {page_num}] {doc.page_content}"
            doc.metadata['chunk_id'] = i
        
        print(f"{Colors.OKGREEN}✅ Created {len(documents)} text chunks{Colors.ENDC}")
        return documents
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning for better processing"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('', '')  # Remove null bytes
        text = text.replace('\x00', '')
        
        # Fix common spacing issues
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' :', ':')
        text = text.replace(' ;', ';')
        
        return text

# === SMART VECTOR STORE MANAGER ===
class VectorStoreManager:
    def __init__(self, embed_model: str = EMBED_MODEL):
        self.embed_model = embed_model
        self.embedding_function = None
        self.vector_store = None
    
    def get_or_create_vector_store(self, pdf_path: str) -> FAISS:
        """Get existing vector store or create new one"""
        pdf_name = Path(pdf_path).stem
        db_dir = f"{pdf_name}_vectordb"
        
        if Path(f"{db_dir}.pkl").exists():
            print(f"{Colors.OKBLUE}🔍 Loading existing vector database...{Colors.ENDC}")
            return self.load_vector_store(db_dir)
        else:
            print(f"{Colors.OKCYAN}🔨 Creating new vector database...{Colors.ENDC}")
            return self.create_vector_store(pdf_path, db_dir)
    
    def create_vector_store(self, pdf_path: str, db_dir: str) -> FAISS:
        """Create new vector store from PDF"""
        processor = DocumentProcessor()
        documents = processor.load_and_process_pdf(pdf_path)
        
        print(f"{Colors.OKBLUE}🧮 Creating embeddings...{Colors.ENDC}")
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = FAISS.from_documents(documents, self.embedding_function)
        
        # Save vector store and documents
        self.vector_store.save_local(db_dir)
        with open(f"{db_dir}_docs.pkl", "wb") as f:
            pickle.dump(documents, f)
        
        print(f"{Colors.OKGREEN}✅ Vector database saved to {db_dir}{Colors.ENDC}")
        return self.vector_store
    
    def load_vector_store(self, db_dir: str) -> FAISS:
        """Load existing vector store"""
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = FAISS.load_local(
            db_dir, 
            self.embedding_function, 
            allow_dangerous_deserialization=True
        )
        
        print(f"{Colors.OKGREEN}✅ Vector database loaded{Colors.ENDC}")
        return self.vector_store

# === ADVANCED PROMPT ENGINEERING ===
class PromptEngineer:
    @staticmethod
    def create_qa_prompt(context: str, question: str, doc_name: str = "") -> str:
        """Create optimized prompt for ISTQB testing scenarios"""
        return f"""You are an ISTQB-certified software testing expert. Based on the provided context from the ISTQB Foundation Level certification guide, identify the most appropriate test design technique for the given scenario.

CONTEXT FROM ISTQB GUIDE:
{context}

TESTING SCENARIO: {question}

INSTRUCTIONS:
- Identify the PRIMARY test design technique that best fits this scenario
- Choose from: Equivalence Partitioning, Boundary Value Analysis, Decision Tables, State Transition Testing, Use Case Testing, or other ISTQB techniques
- Explain WHY this technique is most suitable
- Be specific and concise (maximum 150 words)
- If multiple techniques apply, mention the most important one first
- Use proper ISTQB terminology

PRIMARY TECHNIQUE:"""

    @staticmethod
    def optimize_context(docs: List[Any], max_length: int = MAX_CONTEXT_LENGTH) -> str:
        """Optimize context with smart truncation and relevance scoring"""
        if not docs:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs):
            content = doc.page_content
            content_length = len(content)
            
            if current_length + content_length <= max_length:
                context_parts.append(f"[Excerpt {i+1}] {content}")
                current_length += content_length
            else:
                # Smart truncation - try to keep complete sentences
                remaining = max_length - current_length
                if remaining > 200:
                    sentences = content.split('. ')
                    truncated_content = ""
                    for sentence in sentences:
                        if len(truncated_content) + len(sentence) + 2 <= remaining - 20:
                            truncated_content += sentence + ". "
                        else:
                            break
                    
                    if truncated_content:
                        context_parts.append(f"[Excerpt {i+1}] {truncated_content.strip()}...")
                break
        
        return "\n\n".join(context_parts)

# === MAIN APPLICATION CLASS ===
class PDFQASystem:
    def __init__(self):
        self.ollama_client = EnhancedOllamaClient()
        self.vector_manager = VectorStoreManager()
        self.prompt_engineer = PromptEngineer()
        self.pdf_path = None
        self.vector_store = None
    
    def initialize_system(self) -> bool:
        """Initialize the entire system"""
        print(f"{Colors.HEADER}{Colors.BOLD}🚀 Advanced PDF Q&A System with Llama3.2:3b{Colors.ENDC}")
        print("=" * 60)
        
        # Check Ollama status
        print(f"{Colors.OKBLUE}🔍 Checking Ollama status...{Colors.ENDC}")
        status = self.ollama_client.check_ollama_status()
        
        if not status['running']:
            print(f"{Colors.FAIL}❌ Ollama is not running!{Colors.ENDC}")
            print(f"{Colors.WARNING}Please start Ollama first:{Colors.ENDC}")
            print("   • Download from: https://ollama.ai")
            print("   • Run: ollama serve")
            return False
        
        print(f"{Colors.OKGREEN}✅ Ollama is running{Colors.ENDC}")
        
        # Check if model is available
        if not status['model_available']:
            print(f"{Colors.WARNING}⚠️ Model {OLLAMA_MODEL} not found{Colors.ENDC}")
            if not self.ollama_client.pull_model_with_progress():
                return False
        else:
            print(f"{Colors.OKGREEN}✅ Model {OLLAMA_MODEL} is available{Colors.ENDC}")
        
        # Get PDF path
        while True:
            pdf_path = input(f"{Colors.OKCYAN}📂 Enter path to PDF file: {Colors.ENDC}").strip()
            if Path(pdf_path).exists():
                self.pdf_path = pdf_path
                break
            else:
                print(f"{Colors.FAIL}❌ File not found. Please try again.{Colors.ENDC}")
        
        # Setup vector store
        try:
            self.vector_store = self.vector_manager.get_or_create_vector_store(pdf_path)
            return True
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error setting up vector store: {e}{Colors.ENDC}")
            return False
    
    def run_qa_loop(self):
        """Main Q&A interaction loop with improved input handling"""
        print(f"\n{Colors.HEADER}🎯 Ready for questions!{Colors.ENDC}")
        print(f"{Colors.OKBLUE}📚 Document: {Path(self.pdf_path).name}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}💡 Ask detailed questions about the document content{Colors.ENDC}")
        print(f"{Colors.WARNING}Type 'exit', 'quit', or 'q' to exit{Colors.ENDC}\n")
        
        while True:
            try:
                # Clear any previous input buffer
                import sys
                if hasattr(sys.stdin, 'flush'):
                    sys.stdin.flush()
                
                print(f"{Colors.BOLD}❓ Your question: {Colors.ENDC}", end="", flush=True)
                question = input().strip()
                
                # Handle empty input
                if not question:
                    continue
                    
                # Handle exit commands
                if question.lower() in ['exit', 'quit', 'q']:
                    print(f"{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}")
                    break
                
                # Process the question
                self.process_question(question)
                print("-" * 60)
                
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}")
                break
            except Exception as e:
                print(f"{Colors.FAIL}❌ Input error: {e}{Colors.ENDC}")
                continue
    
    def process_question(self, question: str):
        """Process a single question with enhanced error handling"""
        try:
            # Search for relevant documents with better scoring
            print(f"{Colors.OKBLUE}🔍 Searching for relevant information...{Colors.ENDC}")
            
            # Use multiple search strategies for better results
            docs_similarity = self.vector_store.similarity_search(question, k=4)
            
            # Also search for key terms related to testing techniques
            test_keywords = ["technique", "design", "partition", "boundary", "decision", "state", "transition", "equivalence"]
            keyword_queries = [kw for kw in test_keywords if kw.lower() in question.lower()]
            
            if keyword_queries:
                for keyword in keyword_queries[:2]:  # Limit to 2 additional searches
                    additional_docs = self.vector_store.similarity_search(f"{keyword} testing", k=2)
                    docs_similarity.extend(additional_docs)
            
            # Remove duplicates while preserving order
            seen = set()
            docs = []
            for doc in docs_similarity:
                doc_content = doc.page_content[:100]  # Use first 100 chars as identifier
                if doc_content not in seen:
                    seen.add(doc_content)
                    docs.append(doc)
                if len(docs) >= 5:  # Limit to top 5 unique documents
                    break
            
            if not docs:
                print(f"{Colors.WARNING}⚠️ No relevant information found in the document{Colors.ENDC}")
                return
            
            # Show found sections
            print(f"{Colors.OKGREEN}📄 Found {len(docs)} relevant sections{Colors.ENDC}")
            for i, doc in enumerate(docs[:3]):
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"{Colors.OKCYAN}   {i+1}. {preview}...{Colors.ENDC}")
            
            # Optimize context and generate response
            context = self.prompt_engineer.optimize_context(docs)
            prompt = self.prompt_engineer.create_qa_prompt(
                context, question, Path(self.pdf_path).name
            )
            
            print(f"\n{Colors.OKBLUE}🤔 Generating answer...{Colors.ENDC}")
            response = self.ollama_client.generate_response(prompt, max_tokens=200)  # Reduced for more concise answers
            
            if response:
                print(f"\n{Colors.OKGREEN}{Colors.BOLD}🧠 Answer:{Colors.ENDC}")
                print(f"{Colors.ENDC}{response}\n")
            else:
                print(f"{Colors.FAIL}❌ Could not generate response. Please try rephrasing your question.{Colors.ENDC}")
        
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error processing question: {e}{Colors.ENDC}")

# === MAIN EXECUTION ===
def main():
    """Main application entry point"""
    system = PDFQASystem()
    
    if system.initialize_system():
        system.run_qa_loop()
    else:
        print(f"{Colors.FAIL}❌ System initialization failed{Colors.ENDC}")

if __name__ == "__main__":
    main()