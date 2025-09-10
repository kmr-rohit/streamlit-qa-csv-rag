import chromadb
import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

import warnings
warnings.filterwarnings('ignore')

# Constants
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "csv_embeddings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:1b"

# Streamlit UI setup
st.title("📊 CSV Q&A Assistant")
st.markdown("Upload a CSV file and ask questions about your data using AI-powered retrieval augmented generation")

class SentenceTransformerEmbedding:
    """
    A wrapper class for using Sentence Transformers embeddings.
    """
    def __init__(self, model):
        self.model = model

    def embed_documents(self, documents):
        """
        Embed a list of documents (batch embedding).
        """
        return self.model.encode(documents, convert_to_tensor=True).tolist()

    def embed_query(self, query):
        """
        Embed a single query.
        """
        return self.model.encode(query, convert_to_tensor=True).tolist()

def initialize_chroma_client() -> chromadb.Client:
    """
    Initializes the Chroma client with persistent storage directory and tenant name.
    """
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIRECTORY))
    return client

def detect_text_column(df):
    """
    Automatically detect the best text column from the CSV.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        str: The name of the text column
    """
    # Common text column names to look for (prioritized order)
    text_columns = ['description', 'transcript', 'body', 'text', 'content', 'title', 'message', 'comment', 'tweet', 'post']
    
    # First, try to find exact matches
    for col in text_columns:
        if col in df.columns:
            return col
    
    # If no exact match, look for columns with mostly text data
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns
            # Check if this column has substantial text content
            sample_text = str(df[col].iloc[0]) if len(df) > 0 else ""
            if len(sample_text) > 10:  # Has substantial content
                return col
    
    # If still no match, return the first string column
    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        return string_cols[0]
    
    return None

def load_data_to_vector_store(uploaded_file, client) -> Chroma:
    """
    Loads CSV data into the Chroma vector store.

    Args:
        uploaded_file: The uploaded CSV file.
        client: The Chroma client instance.

    Returns:
        vector_store: The Chroma vector store instance.
    """
    df = pd.read_csv(uploaded_file)
    
    # Display CSV info
    st.write(f"📊 **CSV Info:** {len(df)} rows, {len(df.columns)} columns")
    st.write(f"📋 **Columns:** {', '.join(df.columns.tolist())}")
    
    # Auto-detect text column
    detected_column = detect_text_column(df)
    
    if detected_column is None:
        st.error("❌ No suitable text column found in the CSV. Please ensure your CSV has at least one column with text content.")
        st.stop()
    
    # Let user choose multiple text columns
    st.write("**📝 Select Text Columns for Q&A:**")
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    # Default selection includes the detected column
    default_selection = [detected_column] if detected_column in text_columns else [text_columns[0]] if text_columns else []
    
    selected_columns = st.multiselect(
        "Choose which columns contain text content for Q&A (you can select multiple):",
        text_columns,
        default=default_selection,
        help="Multiple columns will be combined to provide richer context for Q&A"
    )
    
    if not selected_columns:
        st.error("❌ Please select at least one column for text content.")
        st.stop()
    
    st.info(f"✅ **Selected columns: {', '.join(selected_columns)}**")
    
    # Show preview of the selected columns
    with st.expander(f"👀 Preview of selected columns", expanded=False):
        for col in selected_columns:
            st.write(f"**{col} column samples:**")
            sample_texts = df[col].head(2).tolist()
            for i, text in enumerate(sample_texts, 1):
                st.write(f"Sample {i}: {str(text)[:150]}{'...' if len(str(text)) > 150 else ''}")
            st.write("---")
    
    # Combine selected columns into single text content
    def combine_columns(row, columns):
        """Combine multiple columns into a single text string."""
        combined_parts = []
        for col in columns:
            value = str(row[col]).strip()
            if value and value != 'nan':
                combined_parts.append(f"{col}: {value}")
        return " | ".join(combined_parts)
    
    # Create combined documents
    documents = df.apply(lambda row: combine_columns(row, selected_columns), axis=1).tolist()
    
    # Create metadata from other columns (excluding the selected text columns)
    metadata_columns = [col for col in df.columns if col not in selected_columns]
    if metadata_columns:
        metadatas = df[metadata_columns].to_dict(orient='records')
    else:
        metadatas = [{}] * len(df)
    
    # Create IDs - use existing 'id' column if available, otherwise create them
    if 'id' in df.columns:
        ids = df['id'].astype(str).tolist()
    else:
        ids = [f"doc_{i}" for i in range(len(df))]

    # Convert documents to langchain Document objects
    documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]

    # Initialize the embedding model
    embedder = SentenceTransformerEmbedding(SentenceTransformer(EMBEDDING_MODEL))

    # Create the Chroma vector store with persistent storage
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
    )

    # Add documents to the vector store
    vector_store.add_documents(documents)
    st.success("✅ CSV successfully stored in the vector database!")
    return vector_store, df

def check_ollama_connection():
    """
    Check if Ollama is running and accessible.
    
    Returns:
        bool: True if Ollama is accessible, False otherwise
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_llm():
    """
    Get an LLM instance, with fallback options if Ollama is not available.
    
    Returns:
        LLM instance or None if no LLM is available
    """
    if check_ollama_connection():
        try:
            return Ollama(model=LLM_MODEL)
        except Exception as e:
            st.error(f"❌ Error connecting to Ollama: {str(e)}")
            return None
    else:
        st.error("❌ Ollama is not running. Please start Ollama or use an alternative LLM.")
        return None

def answer_query_with_rag(vector_store: Chroma, query: str) -> str:
    """
    Generates an answer for the provided query using Retrieval Augmented Generation (RAG).

    Args:
        vector_store: The Chroma vector store instance.
        query: The question to be answered.

    Returns:
        answer: The generated answer.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Get LLM with error handling
    llm = get_llm()
    if llm is None:
        return "❌ **LLM Error**: Unable to connect to Ollama. Please ensure Ollama is running on localhost:11434 or configure an alternative LLM."

    try:
        # Define the prompt for the question answering chain
        prompt = """
            Only use the context below.
            If unsure, say "I don't know".
        
            Context: {context}
        
            Question: {question}
        
            Answer:
            """
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
        
        llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
        
        document_chain = create_stuff_documents_chain(llm=llm, prompt=QA_CHAIN_PROMPT)

        # Use the RetrievalQA chain for RAG
        qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff", 
            )

        answer = qa.run(query)
        return answer
        
    except Exception as e:
        return f"❌ **Error generating answer**: {str(e)}"

def render_chat_interface():
    """
    Renders the interactive chat interface for Q&A.
    """
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    # Add a clear chat button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat", help="Clear the conversation history"):
            st.session_state['conversation_history'] = []
            st.rerun()
    
    with col2:
        st.markdown("💡 **Tip:** Ask questions like 'What are the main topics?' or 'Summarize the data'")

    # Display the conversation
    if st.session_state['conversation_history']:
        st.markdown("---")
        for speaker, message in st.session_state['conversation_history']:
            if speaker == "User":
                # User's message in left-aligned chat bubble
                st.chat_message("user").markdown(f"""
                <div style='padding: 12px; background-color: #E3F2FD; border-radius: 15px; max-width: 85%; 
                word-wrap: break-word; border-left: 4px solid #2196F3; margin: 5px 0;'>
                    <strong>You:</strong><br>{message}
                </div>
                """, unsafe_allow_html=True)
            else:
                # AI's message in right-aligned chat bubble
                st.chat_message("assistant").markdown(f"""
                <div style='padding: 12px; background-color: #E8F5E8; border-radius: 15px; max-width: 85%; 
                word-wrap: break-word; border-left: 4px solid #4CAF50; margin: 5px 0; margin-left: auto;'>
                    <strong>AI Assistant:</strong><br>{message}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👋 Start a conversation by asking a question below!")

    # Input text box for questions at the bottom of the page
    query = st.chat_input(placeholder="💬 Ask a question about your data...")

    if query:
        # Show loading spinner while processing the answer
        with st.spinner('🤔 Thinking... Please wait while I analyze your data.'):
            answer = answer_query_with_rag(st.session_state.vector_store, query)

        # Store the question and answer in the session state to keep track of the conversation
        st.session_state['conversation_history'].append(("User", query))
        st.session_state['conversation_history'].append(("AI", answer))

        # Rerun to display the new conversation
        st.rerun()

# Initialize session state for vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'df' not in st.session_state:
    st.session_state.df = None

# CSV Upload Section
st.header("📁 Upload Your CSV File")
st.markdown("""
**CSV Requirements:**
- **Text Column**: Any column with text content (auto-detected from: body, text, content, description, title, message, etc.)
- **ID Column**: Optional - if not present, auto-generated IDs will be used
- **Metadata**: All other columns will be used as metadata for context

The app will automatically detect the best text column and index it for intelligent Q&A.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV file with the required columns")

if uploaded_file is not None:
    with st.spinner("Processing the uploaded file... Please wait."):
        client = initialize_chroma_client()
        vector_store, df = load_data_to_vector_store(uploaded_file, client)
        st.session_state.vector_store = vector_store
        st.session_state.df = df
        
        st.success("✅ CSV successfully processed and indexed!")
        
        # Show data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.write("**Data Preview:**")
        st.dataframe(df.head())
        
        # Add sample questions based on the data type
        st.markdown("**💡 Try asking questions like:**")
        if 'description' in df.columns or 'transcript' in df.columns:
            # TED talks specific questions
            sample_questions = [
                "What are the most popular topics in these talks?",
                "Which speakers have the most views?",
                "What are the main themes discussed?",
                "Tell me about talks related to technology",
                "What insights can you share about the most viewed talks?",
                "Compare the titles and descriptions to find patterns",
                "What can you tell me about the speakers and their topics?"
            ]
        else:
            # Generic questions for multi-column data
            sample_questions = [
                "What are the main topics in this data?",
                "Can you summarize the key insights?",
                "What patterns do you see across the different columns?",
                "Tell me about the most interesting entries",
                "How do the different text fields relate to each other?",
                "What themes emerge when you combine all the text content?"
            ]
        
        for question in sample_questions:
            st.markdown(f"• {question}")

# Q&A Section
st.header("💬 Ask Questions About Your Data")

if st.session_state.vector_store is not None:
    render_chat_interface()
else:
    st.info("👆 Please upload a CSV file first to start asking questions!")

# Sidebar (moved to end to avoid function definition order issues)
with st.sidebar:
    st.header("🔧 Settings")
    st.markdown("**Model Configuration:**")
    
    # Check LLM connection status
    if check_ollama_connection():
        st.success(f"✅ LLM: {LLM_MODEL} (Connected)")
    else:
        st.error(f"❌ LLM: {LLM_MODEL} (Not Connected)")
        st.markdown("**To fix this:**")
        st.markdown("""
        1. Install Ollama: https://ollama.ai
        2. Run: `ollama pull deepseek-r1:14b`
        3. Start Ollama service
        """)
    
    st.info(f"🧠 Embedding: {EMBEDDING_MODEL}")
    
    st.markdown("---")
    st.markdown("**📋 How it works:**")
    st.markdown("""
    1. Upload a CSV file with text data
    2. The app creates vector embeddings
    3. Ask questions about your data
    4. Get AI-powered answers using RAG
    """)
    
    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("""
    - Ask specific questions
    - Use natural language
    - Try different phrasings
    - Ask for summaries or patterns
    """)

