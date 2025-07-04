from flask import Flask, render_template, request
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

load_dotenv()

app = Flask(__name__)

# Load API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """You are Steven Ndacyarikumukiza, a helpful student at Kepler College, Business Analytics. You should answer general questions yourself; you should be polite.
    You shouldn't let them know you are a bot.
    Answer the questions on the provided context only. Please provide the most accurate response based on the question.
    <context>{context}</context>
    Questions: {input}
    """
)

# Custom embedding class for LangChain compatibility
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, query):
        return self.model.encode([query])[0]

def handle_common_questions(question):
    common_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! What can I help you with?",
    }
    return common_responses.get(question.lower())

def vector_embedding():
    if "vectors" not in app.config:
        app.config['embeddings'] = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        app.config['loader'] = PyPDFDirectoryLoader("./data")
        app.config['docs'] = app.config['loader'].load()
        app.config['text_splitter'] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        app.config['final_documents'] = app.config['text_splitter'].split_documents(app.config['docs'][:20])
        
        document_texts = [doc.page_content for doc in app.config['final_documents']]
        embeddings = app.config['embeddings'].embed_documents(document_texts)
        
        for i, doc in enumerate(app.config['final_documents']):
            doc.metadata["embedding"] = embeddings[i]
        
        app.config['vectors'] = FAISS.from_documents(app.config['final_documents'], app.config['embeddings'])

@app.route('/', methods=['GET', 'POST'])
def index():
    vector_embedding()
    if 'messages' not in app.config:
        app.config['messages'] = []

    if request.method == 'POST':
        question = request.form.get('question')
        common_response = handle_common_questions(question)
        if common_response:
            response = common_response
        else:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = app.config["vectors"].as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({'input': question})['answer']
        app.config['messages'].append((question, response))
    return render_template('index.html', messages=app.config.get('messages', []))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 9999))
    app.run(debug=False, host='0.0.0.0', port=port)
