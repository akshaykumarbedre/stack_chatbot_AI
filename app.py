# %% Import Required Libraries
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify
from slackeventsapi import SlackEventAdapter
import slack
from dotenv import load_dotenv
import os

# %% Load Environment Variables
load_dotenv()

# API Keys and Tokens
gemini_api_key = os.environ['GEMINI_API_KEY']
SLACK_TOKEN = os.environ['SLACK_TOKEN']
SIGNING_SECRET = os.environ['SIGNING_SECRET']

# %% Initialize Flask App and Slack Adapter
app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, '/slack/events', app)

# %% Initialize Slack Client
client = slack.WebClient(token=SLACK_TOKEN)

# %% Load Document (Zepto PDF)
document = PyMuPDFLoader("zepto.pdf").load()

# %% Split Document into Chunks
test_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_doc = test_splitter.split_documents(document)

# %% Generate Embeddings using Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)

# %% Create FAISS Vector Store and Save Locally
vector_store = FAISS.from_documents(split_doc, embeddings)
FAISS.save_local(vector_store, "vector_store")

# %% Load FAISS Vector Store
vector = FAISS.load_local(
    "vector_store",
    allow_dangerous_deserialization=True,
    embeddings=embeddings
)

# %% Initialize Retriever
retriever = vector.as_retriever(search_kwargs={'k': 6})

# %% Custom Prompt Template
custom_prompt_template = """
You are an AI assistant that answers user queries based on the provided context. 
If the context is insufficient, respond by saying you don't have enough information. 
Be concise and accurate with freindly way .

Context:
{context}

Question:
{question}

Provide a short, on-point answer.
"""

# Create PromptTemplate
custom_prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# %% Initialize Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key=gemini_api_key
)

# %% Create RetrievalQA Chain with Custom Prompt
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# %% Function to Query RAG and Get Response
def get_rag_response(query):
    response = qa_chain.run(query)
    return response

# %% Slack Message Event Handling
@ slack_event_adapter.on('message')
def message(payload):
    print(payload)
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    print(text,user_id)

    # Skip bot messages
    if user_id is None or 'bot_id' in event:
        return

    # Get response from RAG
    response = get_rag_response(text)

    # Send RAG response to Slack channel
    client.chat_postMessage(channel=channel_id, text=f" {response}")

# %% Send Initial Message on Bot Start
client.chat_postMessage(channel='#first_chatbot', text='Hello World! The bot is up and running! 🎉')

# %% Run Flask App
if __name__ == "__main__":
    app.run(debug=True, port=5000)
