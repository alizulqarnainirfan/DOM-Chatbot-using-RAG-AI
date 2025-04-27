AI Chatbot for Directorate of Monitoring (DOM)
Project Overview
This project is an AI-powered chatbot developed to assist the Directorate of Monitoring (DOM) by providing intelligent, context-aware responses based on uploaded documents. The chatbot leverages modern natural language processing (NLP) techniques and advanced AI technologies to efficiently understand and respond to user queries about the DOM's operations, guidelines, and other related information. This Project is Built by ALI ZULQARNAIN (AI/ML Developer) -2025 @alizulqarnainirfan
# GitHub:alizulqarnainirfan
# LinkedIn:alizulqarnainirfan
# mail:aliirfanawan13@gmail.com

The chatbot integrates several AI components, including LangChain, Hugging Face models, and FAISS for vector-based document search. The FastAPI framework is used to serve the chatbot as a web API, providing a user-friendly interface for interacting with the system.

Key Features
Document Ingestion: The system allows users to upload .docx files that contain important information about the Directorate of Monitoring (DOM). These documents are ingested, processed, and stored for later retrieval.

Contextual Responses: Using FAISS (a vector store) and Hugging Face embeddings, the chatbot generates accurate responses based on the document content. It can provide meaningful insights and explanations about the DOM and related topics.

Advanced AI Models: The chatbot uses advanced Hugging Face transformer models (e.g., Google’s FLAN-T5) for text generation and query understanding, ensuring that responses are natural, relevant, and contextually accurate.

Vector Search: The chatbot leverages FAISS for efficient semantic search. By transforming document text into high-dimensional embeddings, FAISS allows for quick and accurate retrieval of the most relevant sections of the document based on user queries.

FastAPI Integration: The chatbot is deployed using FastAPI, enabling it to interact with users via a RESTful API. This ensures fast, scalable, and easy access to the chatbot’s functionalities.

How It Works
Document Upload: Users upload .docx files containing information about the DOM to the system.

Document Processing: The content of the documents is processed using LangChain to break them into smaller chunks, and embeddings are created using Hugging Face’s sentence-transformers/all-MiniLM-L6-v2 model.

Embedding and Indexing: The processed chunks are stored in a FAISS vector store for efficient retrieval.

Query Handling: When a user submits a query, the chatbot uses FAISS and the Hugging Face model to search the document for the most relevant information and provide an accurate, context-based response.

Web API: FastAPI serves as the backend, allowing users to interact with the chatbot via HTTP requests.

Technologies Used
LangChain: For document ingestion, text splitting, and managing the workflow for querying large-scale documents.

Hugging Face Transformers: For embedding generation and text-to-text generation models to handle user queries.

FAISS: A library for fast similarity search and clustering of dense vectors. FAISS is used to store the embeddings and retrieve relevant document sections based on user queries.

FastAPI: A modern, fast (high-performance) web framework for building APIs with Python 3.6+.

Python: The core programming language used to implement the project.

Installation
Prerequisites
Ensure that you have Python 3.6+ installed. You also need to install pip for managing Python packages.

Steps to Run the Project
Clone the repository:

bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
Set up a virtual environment:

bash
python -m venv venv
Activate the virtual environment:

For Windows:

bash
venv\Scripts\activate
For macOS/Linux:

bash
source venv/bin/activate
Install the required dependencies:

bash
pip install -r requirements.txt
Run the FastAPI server:

bash
uvicorn main:app --reload
The server will be accessible at http://127.0.0.1:8000.

You can interact with the chatbot through the /chat endpoint using POST requests, sending a query field in the JSON body.

Future Improvements
Multilingual Support: Expand the chatbot’s capabilities to handle multiple languages and make it more inclusive.

More Advanced Query Handling: Improve the query processing and response generation for more complex questions.

UI Integration: Build a front-end user interface to make interacting with the chatbot more intuitive and user-friendly.

Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests. If you have any suggestions or find bugs, please open an issue, and we will address it as soon as possible.