import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
PDF_PATH = "C:/Users/test/Desktop/satakshi/New folder/TC_LE910R1_AT_Commands_Reference_Guide_r6 (2).pdf"
DB_DIR = "faiss_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen1.5-0.5B"  # Changed from TinyLlama to Qwen

print("🚀 RAG System with FAISS and Qwen1.5-0.5B")

# Step 1: Load and process PDF
if not os.path.exists(f"{DB_DIR}.pkl"):
    print("📄 Loading and processing PDF...")
    try:
        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
        print(f"✅ Loaded {len(pages)} pages")

        print("✂️ Splitting into chunks...")
        # Use larger chunks to reduce total count
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=100,
            length_function=len
        )
        documents = splitter.split_documents(pages)
        print(f"✅ Created {len(documents)} chunks")

        print("🔍 Creating embeddings...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cpu'}
        )

        print("💾 Creating FAISS vector store...")
        # FAISS is more reliable than Chroma for large datasets
        vector_store = FAISS.from_documents(documents, embedding_model)
        
        print("💾 Saving vector store...")
        vector_store.save_local(DB_DIR)
        
        # Also save the documents for reference
        with open(f"{DB_DIR}_docs.pkl", "wb") as f:
            pickle.dump(documents, f)
            
        print("✅ Vector store created and saved!")
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        exit(1)
else:
    print("✅ Loading existing FAISS vector store...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        vector_store = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        print("🔄 Deleting corrupted files and recreating...")
        # Clean up and recreate
        for file in [f"{DB_DIR}.faiss", f"{DB_DIR}.pkl", f"{DB_DIR}_docs.pkl"]:
            if os.path.exists(file):
                os.remove(file)
        exit(1)

# Step 2: Load Qwen1.5-0.5B (optional - can skip for testing)
use_llm = input("Load Qwen1.5-0.5B model? (y/n, default=n): ").lower().strip() == 'y'

if use_llm:
    print("🤖 Loading Qwen1.5-0.5B...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
        # Qwen models typically have proper pad tokens, but set if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True  # Required for Qwen models
        )
        model.eval()
        print("✅ Qwen1.5-0.5B loaded!")
    except Exception as e:
        print(f"❌ Error loading Qwen1.5-0.5B: {e}")
        use_llm = False
        print("📝 Continuing with retrieval-only mode...")

# Step 3: Interactive query loop
print("\n💬 Ask questions about the AT Commands document (type 'exit' to quit)")
print("🔍 Retrieval-based answers" + (" + AI generation" if use_llm else " only"))
print()

while True:
    try:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break
            
        if not query:
            continue

        print("🔍 Searching document...")
        
        # Search for relevant documents
        try:
            docs = vector_store.similarity_search(query, k=3)
            print(f"✅ Found {len(docs)} relevant sections")
        except Exception as e:
            print(f"❌ Search error: {e}")
            continue

        if not docs:
            print("❌ No relevant information found")
            continue

        # Combine context from retrieved documents
        context = "\n\n".join([f"Section {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        print(f"\n📄 Retrieved Context:")
        print("=" * 50)
        for i, doc in enumerate(docs):
            print(f"Section {i+1}: {doc.page_content[:200]}...")
            print("-" * 30)
        
        if use_llm:
            print("\n🤖 Generating AI response...")
            try:
                # Improved prompt format for Qwen models
                prompt = f"""<|im_start|>system
You are a helpful assistant that answers questions based on provided documentation about AT commands. Be concise and accurate.
<|im_end|>
<|im_start|>user
Based on this documentation:

{context}

Question: {query}
<|im_end|>
<|im_start|>assistant
"""
                
                # Tokenize with proper truncation for context length
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1800  # Leave room for generation
                )
                
                # Generate with optimized parameters for Qwen
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.8,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                # Extract response
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                # Clean up response (remove any remaining special tokens or artifacts)
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0].strip()
                
                print(f"\n🧠 AI Response:\n{response}")
                
            except Exception as e:
                print(f"❌ AI generation error: {e}")
        
        print("\n" + "="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        continue