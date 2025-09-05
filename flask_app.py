import os
import io
from flask import Flask, render_template_string, request, redirect, url_for, flash
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import PyPDF2

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for flashing messages

HTML_TEMPLATE = """
<!doctype html>
<title>Chat Your PDFs (Flask)</title>
<h2>Chat Your PDFs</h2>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul>
    {% for message in messages %}
      <li style="color:red;">{{ message }}</li>
    {% endfor %}
    </ul>
  {% endif %}
{% endwith %}
<form method=post enctype=multipart/form-data>
  <label>Upload a PDF file:</label>
  <input type=file name=pdf_file accept="application/pdf"><br><br>
  <label>Ask a Question:</label>
  <input type=text name=question style="width:400px;"><br><br>
  <input type=submit value="Get Answer">
</form>
{% if answer %}
  <h3>Answer:</h3>
  <div style="white-space: pre-wrap;">{{ answer }}</div>
{% endif %}
"""

def process_pdf(pdf_data):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
    pdf_pages = pdf_reader.pages
    context = "\n\n".join(page.extract_text() for page in pdf_pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(context)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(
        texts,
        embeddings,
        persist_directory="./chroma_db"
    ).as_retriever()
    return vector_index

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        if not google_api_key:
            flash("API key not found. Please set the GOOGLE_API_KEY environment variable.")
            return render_template_string(HTML_TEMPLATE, answer=None)
        pdf_file = request.files.get("pdf_file")
        question = request.form.get("question", "").strip()
        if not pdf_file or not pdf_file.filename.endswith(".pdf"):
            flash("Please upload a valid PDF file.")
            return render_template_string(HTML_TEMPLATE, answer=None)
        if not question:
            flash("Please enter a question.")
            return render_template_string(HTML_TEMPLATE, answer=None)
        pdf_data = pdf_file.read()
        vector_index = process_pdf(pdf_data)
        docs = vector_index.get_relevant_documents(question)
        prompt_template = """
        Answer the question as detailed as possible from the provided context,
        make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context",
        don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, api_key=google_api_key)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        answer = response.get('output_text', 'No answer found.')
    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == "__main__":
    app.run(debug=True, port=5050)