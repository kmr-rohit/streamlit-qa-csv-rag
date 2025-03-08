# Streamlit Q&A on CSV using Retrieval Augmented Generation (RAG)

![demo](./st-rag-qa.png)

## Description
This application is a Streamlit-based interactive Q&A system that uses Retrieval Augmented Generation (RAG) to answer questions based on the contents of a CSV file uploaded by the user. The uploaded CSV file is indexed into a Chroma vector database, where the content is embedded using the `SentenceTransformers` model for efficient and scalable retrieval. Once the data is indexed, users can interact with the system by querying the dataset and receive responses powered by a language model.

## Key Features
- **Upload CSV Files**: Upload CSV files containing the Twitter/X data that you want to query.
- **Data Storage with VectorDB**: CSV data is converted into embeddings and stored in a vector database (ChromaDB) for fast retrieval.
- **Interactive Question-Answering**: Users can ask questions about the CSV data via an interactive chat interface, and the AI provides relevant answers based on the stored embeddings.
- **Seamless User Experience**: The chat interface is designed to feel like a natural conversation, with message bubbles for both user and AI responses.
- **Persistent Chat History**: The app keeps track of the entire conversation for a continuous and dynamic user experience.

## Installation

To run the project locally, follow these steps:

### Prerequisites
- Python 3.10+
- Streamlit
- LangChain
- Ollama
- DeepSeek-R1 (served via Ollama)
- ChromaDB

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/vishukla/streamlit-qa-csv-rag.git
   cd streamlit-qa-csv-rag
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the application
Once the environment is set up, you can run the Streamlit app:
 ```bash
 streamlit run app.py
 ```
This will launch a local Streamlit server, and the app will be accessible in your web browser at `http://localhost:8501`.

## How to use

### Upload CSV
- Upload a CSV file containing the tweets by clicking the Upload CSV to Vector DB tab.
- The CSV file should have the following columns:
  - `body`: The textual content to be indexed.
  - `id`: A unique identifier for each row (e.g., tweet ID, document ID).
  - `hashtags`, `author`, `mentions`: Metadata associated with each row.
The data will be indexed into the Chroma vector store, and you’ll see a preview of the CSV data.

### Step 2: Interactive Q&A
- Go to the "Question Answering" tab to ask questions related to the data you uploaded.
- The app uses a Retrieval-Augmented Generation (RAG) approach to find the most relevant information in the CSV and generate an AI-powered answer.

### Chat Interface
- Once the CSV is uploaded and the database is created, the user can engage with an interactive chat interface where they can type queries.
- Each user query and AI response is displayed as a message bubble.
- The interface supports continuous conversation, keeping track of the chat history.


## Configuration

### Embedding Model
This app uses the `all-MiniLM-L6-v2` model from `Sentence-Transformers` to create embeddings. You can change the embedding model by modifying the `EMBEDDING_MODEL` variable in the `app.py` script.
  ```bash
  EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Change to another model if desired
  ```

### Persistent Directory
The vector store is persisted in the `./chroma_db` directory. You can change this directory by modifying the `PERSIST_DIRECTORY` variable in the script.
  ```bash
  PERSIST_DIRECTORY = "./chroma_db"  # Modify path if desired
  ```

### Language Model
The current setup uses DeepSeek-R1's `deepseek-r1:14b` served via Ollama. Change this by updating `LLM_MODEL` variable in the script.
  ```bash
  LLM_MODEL = "deepseek-r1:14b"  # Update as needed
  ```

## Architecture

### 1. Vector Database:
- ChromaDB stores the embeddings of the CSV data.
- SentenceTransformer (`all-MiniLM-L6-v2`) is used to create embeddings for the textual data in the CSV.

### 2. Retrieval-Augmented Generation (RAG):
- The app uses the RAG approach to combine the information retrieval from the vector database with generative language models to answer user queries.
- DeepSeek-R1 model (served via Ollama for specific use case) is used for generating answers based on retrieved context.

### 3. Chat Interface:
- The user interacts with the application through a chat interface in Streamlit, which updates dynamically as the conversation progresses.

## Troubleshooting
1. Issue with CSV Format: Ensure your CSV contains the correct columns (e.g., `body`, `id`, `hashtags`, `author`, `mentions`).
2. Model Errors: If you’re using an external language model such OpenAPI, ensure your API keys and environment variables are correctly set and LangChain model initialization in the script. Alternatively if using other models served by Ollama, make sure to update `LLM_MODEL` variable in the script.
3. Chroma Errors: If you encounter issues with Chroma, ensure the persistent directory is accessible, and try clearing the `./chroma_db` folder.
