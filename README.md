# PDF Question Answering App

This is a streamlit app for extracting text from PDF documents and building a question answering system on top of the extracted text.

## Features

- Upload a PDF document
- Extract text from PDF 
- Split text into chunks
- Create vector embeddings for text  
- Build a FAISS vector index
- Initialize a RetrievalQA agent using the vector index
- Ask questions about the PDF content and get answers from the QA system

## Usage

To run the app:

```
streamlit run app.py
```

Then upload a PDF file to extract the text. Once a file is uploaded, you can enter questions in the text box and hit Ask to get an answer generated from the PDF content.

Uses the `tiiuae/falcon-7b-instruct` model from HuggingFace for answering.

The app handles splitting the PDF text into appropriate chunks, building embeddings, vector search index using FAISS and configuring the RetrievalQA agent.

## Libraries

- streamlit
- PyPDF2
- fitz
- langchain
- transformers

## Contributing

Contributions to improve the app are welcome! Some ideas:

- Support other file formats beyond PDF
- Add user session management 
- Improve answer relevance through better embeddings
- Containerize the app
