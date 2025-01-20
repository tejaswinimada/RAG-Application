# from langchain_community.document_loaders import PyPDFLoader       this is just a normla chatbot ,u gives que and it generates the ans by upoadinh only one pdf
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import streamlit as st
# load_dotenv()
# st.title("Nichebrains Chatbot")
# loaders=PyPDFLoader("A_Brief_Introduction_To_AI.pdf")
# data=loaders.load()
# text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000)
# docs=text_splitter.split_documents(data)
# # embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# # vector=embeddings.embed_query("hello world")
# # vector[:5]

# vectorstore=Chroma.from_documents(documents=docs,embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
# retriever=vectorstore.as_retriever(search_type="similarity" ,search_kwargs={"k":10})
# # retrieved_docs=retriever.invoke("what is ai")
# # print(retrieved_docs[2].page_content)
# llm= ChatGoogleGenerativeAI(model="gemini-1.5-pro" ,temperature=0.3,max_tokens=500)
# query = st.chat_input("Enter your query here: ") 
# prompt = query

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# if query:
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     response = rag_chain.invoke({"input": query})
#     #print(response["answer"])

# #     st.write(response["answer"])

#you can update more than one pdf but does not display the user query
# import os                     
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import streamlit as st

# # Load environment variables
# load_dotenv()

# st.title("Nichebrains Chatbot")

# # Allow users to upload multiple PDFs
# uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# if uploaded_files:
#     docs = []  # List to hold all documents

#     # Process each uploaded PDF
#     for uploaded_file in uploaded_files:
#         with st.spinner(f"Processing {uploaded_file.name}..."):
#             # Save the uploaded file temporarily
#             temp_file_path = os.path.join("temp", uploaded_file.name)
#             os.makedirs("temp", exist_ok=True)  # Create temp directory if it doesn't exist
#             with open(temp_file_path, "wb") as f:
#                 f.write(uploaded_file.read())
            
#             # Load and process the PDF
#             loader = PyPDFLoader(temp_file_path)
#             data = loader.load()
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
#             docs.extend(text_splitter.split_documents(data))
            
#             # Clean up temporary file
#             os.remove(temp_file_path)

#     # Create vector store
#     vectorstore = Chroma.from_documents(
#         documents=docs,
#         embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     )

#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

#     # Initialize the LLM
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

#     query = st.chat_input("Enter your query here:")
#     if query:
#         # Define the system prompt
#         system_prompt = (
#             "You are an assistant for question-answering tasks. "
#             "Use the following pieces of retrieved context to answer "
#             "the question. If you don't know the answer, say that you "
#             "don't know. Use three sentences maximum and keep the "
#             "answer concise."
#             "\n\n"
#             "{context}"
#         )

#         # Create prompt
#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", system_prompt),
#                 ("human", "{input}"),
#             ]
#         )

#         # Create chains
#         question_answer_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#         # Generate response
#         response = rag_chain.invoke({"input": query})
#         st.write(response["answer"])


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st
import base64
# Load environment variables
load_dotenv()
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("nichebrains.jpg")


page_bg_img = f"""
<style>
[data-testid="stApp"] > .main {{
background-image: url("nichebrains.jpg");
background-size: 200%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
</style>
"""
    

st.markdown(page_bg_img,unsafe_allow_html=True)


st.title("Nichebrains Chatbot")

# Allow users to upload multiple PDFs
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    docs = []  # List to hold all documents

    # Process each uploaded PDF
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save the uploaded file temporarily
            temp_file_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)  # Create temp directory if it doesn't exist
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Load and process the PDF
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            docs.extend(text_splitter.split_documents(data))
            
            # Clean up temporary file
            os.remove(temp_file_path)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("Enter your query here:")
    if query:
        # Append user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Define the system prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Generate response
        response = rag_chain.invoke({"input": query})
        bot_response = response["answer"]

        # Append bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

    # Display the chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
