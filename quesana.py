from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import fitz  # PyMuPDF
import streamlit as st

def get_pdf_text(uploaded_file):
    text = ""
    
    with uploaded_file as file:
        file_contents = file.read()
    
    pdf_document = fitz.open(stream=file_contents, filetype="pdf")
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()

    pdf_document.close()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def retrieval_qa_chain(db, return_source_documents):
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6, "max_length": 500, "max_new_tokens": 700})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db,
                                           return_source_documents=return_source_documents,
                                           )
    return qa_chain

def main():
    st.title("PDF Question Answering Chat Interface")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Read PDF and create text chunks
        raw_text = get_pdf_text(uploaded_file)
        text_chunks = get_text_chunks(raw_text)

        # Create a vector store
        vectorstore = get_vectorstore(text_chunks)

        # Create a retriever database
        db = vectorstore.as_retriever(search_kwargs={'k': 3})

        # Initialize QA bot
        bot = retrieval_qa_chain(db, True)

        # User chat interface
        st.subheader("Chat Interface")
        user_question = st.text_input("Ask a question:")
        
        if st.button("Ask"):
            # Get answer from the language model
            answer = bot(user_question)["result"]
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
