from flask import Flask, request, jsonify, render_template, send_from_directory
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import re

app = Flask(__name__, static_folder='genie', template_folder='genie')

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Please set your GOOGLE_API_KEY in the .env file")

# Configure Google Generative AI
genai.configure(api_key=api_key)

# Initialize Gemini model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048
}
gemini_model = genai.GenerativeModel(model_name="gemini-1.0-pro", generation_config=generation_config)
gemini_convo = gemini_model.start_chat(history=[])

# Global variable to store the PDF text
pdf_text = ""

# PDF processing functions
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Use the following conversation history and context to answer the question. 
    If the answer is not in the provided context, just say "Answer is not available in the context."
    
    Chat History: {chat_history}
    Context: {context}
    Question: {question}
    
    Please provide a detailed answer based on the context and previous conversation:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def count_words(text):
    return len(text.split())

def sum_numbers_in_column(text):
    numbers = re.findall(r'\b\d+\b', text)
    return sum(map(int, numbers))

@app.route('/')
def home():
    return render_template('Ghome.html')

@app.route('/eg.html')
def eg():
    return render_template('eg.html')

@app.route('/Gsignin.html')
def gsignin():
    return render_template('Gsignin.html')

@app.route('/Test.html')
def test():
    return render_template('Test.html')

@app.route('/project.html')
def project():
    return render_template('project.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global pdf_text
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and file.filename.lower().endswith('.pdf'):
        try:
            pdf_text = get_pdf_text(file)
            text_chunks = get_text_chunks(pdf_text)
            get_vector_store(text_chunks)
            return jsonify({'message': 'PDF processed successfully'})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'})

@app.route('/chat', methods=['POST'])
def chat():
    global pdf_text
    data = request.get_json()
    user_message = data['message']
    chat_type = data.get('type', 'gemini')

    if chat_type == 'gemini':
        gemini_convo.send_message(user_message)
        bot_response = gemini_convo.last.text
        return jsonify({'response': bot_response})
    elif chat_type == 'pdf':
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_message)
            
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_message, "chat_history": []})
            
            output_text = response["output_text"]
            
            if "word count" in user_message.lower():
                word_count = count_words(pdf_text)
                output_text += f"\n\nWord count: {word_count}"
            
            if "sum of numbers" in user_message.lower():
                sum_of_numbers = sum_numbers_in_column(pdf_text)
                output_text += f"\n\nSum of numbers: {sum_of_numbers}"
            
            return jsonify({'response': output_text})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Invalid chat type'})

# Serve static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)