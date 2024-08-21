from flask import Flask, render_template, request, redirect, url_for
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
load_dotenv()

app=Flask(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")


llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(

  """You are a HR. you should answere general question yourself,you should be polite.
  You shouldn't let them know you are a bot.
  Answer the questions on the provided context only. Please provide the most accurate response based on the question]
<context>
{context}
<context>
Questions: {input}

"""
)

# Define a function for handling common questions
def handle_common_questions(question):
    common_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! What can I help you with?",
        # Add more common questions and responses here
    }
    return common_responses.get(question.lower())

def vector_embedding():
    if "vectors" not in app.config:
        app.config['embeddings'] = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        app.config['loader'] = PyPDFDirectoryLoader("./data")
        app.config['docs'] = app.config['loader'].load()
        app.config['text_splitter'] = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
        app.config['final_documents'] = app.config['text_splitter'].split_documents(app.config['docs'][:20])
        app.config['vectors'] = FAISS.from_documents(app.config['final_documents'],app.config['embeddings'])

def format_response(response):
    # Format the response with HTML line breaks
    formatted_response = response.replace("\n", "<br>")
    return formatted_response

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    vector_embedding()  # Initialize embeddings

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
            start = time.process_time()  # Start timing the response generation
            response = retrieval_chain.invoke({'input': question})['answer']
            response_time = time.process_time() - start
            response += f" (Response Time: {response_time:.2f} seconds)"

        app.config['messages'].append((question, response))

    return render_template('index.html', messages=app.config.get('messages', []))

if __name__ == '__main__':
    app.run(debug=True)