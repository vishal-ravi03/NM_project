import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.messages import AIMessage,HumanMessage,SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import PyPDF2 as pdf
import numpy
import tempfile
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

doc=st.sidebar.file_uploader("uploader Your PDF", type=["pdf"])


LLM=ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

def get_text(file):
    if file is not None:
        reader = pdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
     


prompt = f"""
Summarize the following text in two versions:
1. A concise summary in approximately 100 words.
2. A detailed summary in approximately 1000 words.

Text:
{{text}}
"""

if doc is not None:
    # Process the file
    try: 
        text = get_text(doc)
        text1=text[:5000]
        promt=prompt.format(text=text1)
        with st.spinner("Processing..."):
            response= LLM.invoke(promt)
            result=response.content
            st.write(result)
        st.success("Processing complete!")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        text=""
else:
    pass
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a conversational AI that uses retrieved context and prior user interactions to provide consistent and helpful answers.")
    ]
 
store_data="datastore"


if not os.path.exists(store_data):
    if doc:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(doc.read())
            temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            data = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
            text_splitter = splitter.split_documents(data)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
            vector_store = FAISS.from_documents(text_splitter, embedding=embeddings)
            vector_store.save_local(store_data)
    else:
        st.warning("Please upload a PDF to start.")
        st.stop()
else:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

    vector_store = FAISS.load_local(store_data, embeddings=embeddings,allow_dangerous_deserialization=True)
 
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

 


prompt=ChatPromptTemplate.from_template("""



Use only the information from the context to answer the user's question. If the information is missing, say "I don't know."
                          
context:{context}
                          
question:{question}


""")

def format(doc):
    return " ".join(docs.page_content for docs in doc)

chain=(
    {"context":retriever | format , "question":RunnablePassthrough()}
    |prompt
    |LLM
    |StrOutputParser()
)

question = st.chat_input("Ask your question:")

st.markdown("""
    <style>
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        .chat-message {
            max-width: 70%;
            padding: 5px 10px;
            border-radius: 12px;
            margin: 8px;
        }
        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, #E3FDF5, #FFE6FA);
            color: #333;
            border: 1px solid #A6DCEF;  
            text-align: right;
        }
        .ai-message {
            align-self: flex-start;
            background: linear-gradient(135deg, #FFFAFA, #F5DEB3);
            color: #000;
            border: 1px solid #FFC26F; 
        }
        div.stButton > button:first-child {
        background-color: #ff4d4d ; 
        color: white;
        }
    </style>
""", unsafe_allow_html=True)


if "chat_history" not  in st.session_state:

        st.session_state.chat_history = [
        SystemMessage(content="""
You are a helpful, knowledgeable AI assistant designed to answer user questions using both your internal knowledge and external documents retrieved via a Retrieval-Augmented Generation (RAG) system.

Your responsibilities include:
- Providing accurate, grounded answers based on the retrieved context.
- Citing relevant sources when applicable.
- Admitting when the answer is not known or outside the retrieved context.
- Maintaining a professional, concise, and user-friendly tone.

Guidelines:
- Use only the retrieved documents to answer questions when context is provided.
- If no context is provided or the context is insufficient, fall back on your general knowledge but clearly indicate this.
- Do not fabricate citations or facts.
- If a document contains conflicting information, indicate uncertainty and prefer the more reliable or recent source.
- Do not answer questions unrelated to the domain unless specifically instructed.

Assume the user may upload documents such as PDFs or ask questions about specific content. Always aim to be accurate, transparent, and grounded in the data.

""")
    ]

#process the input 
if question:
    try:
        human_message = HumanMessage(content=question)
        st.session_state.chat_history.append(human_message)
        with st.spinner("🧠 Generating........"):
            result = chain.invoke(question)

        if result:
            with st.spinner("🧠 Generating........"):
                ai_message = AIMessage(content=result)
                st.session_state.chat_history.append(ai_message)

        # Display all chat messages
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(f"<div class='chat-message user-message'>👤 You: {message.content}</div>", unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.markdown(f"<div class='chat-message ai-message'>🤖  AI: {message.content}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Limit message history
        st.session_state.chat_history = st.session_state.chat_history[-100:]

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.markdown("<div class='chat-message ai-message'> Oops! Something went wrong. Try again 💬</div>", unsafe_allow_html=True)

# Clear button
if st.button("Clear Chat",type="primary"):
    st.session_state.chat_history = [
        SystemMessage(content="""
You are a helpful, knowledgeable AI assistant designed to answer user questions using both your internal knowledge and external documents retrieved via a Retrieval-Augmented Generation (RAG) system.

Your responsibilities include:
- Providing accurate, grounded answers based on the retrieved context.
- Citing relevant sources when applicable.
- Admitting when the answer is not known or outside the retrieved context.
- Maintaining a professional, concise, and user-friendly tone.

Guidelines:
- Use only the retrieved documents to answer questions when context is provided.
- If no context is provided or the context is insufficient, fall back on your general knowledge but clearly indicate this.
- Do not fabricate citations or facts.
- If a document contains conflicting information, indicate uncertainty and prefer the more reliable or recent source.
- Do not answer questions unrelated to the domain unless specifically instructed.

Assume the user may upload documents such as PDFs or ask questions about specific content. Always aim to be accurate, transparent, and grounded in the data.

""")
    ]
    st.rerun()
 

            

