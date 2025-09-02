from dotenv import load_dotenv
import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load .env file
load_dotenv()

# Load existing Chroma DB
def load_chroma_db(persist_dir="./data/chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectordb

def create_rag_pipeline():
    vectordb = load_chroma_db()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    #Groq LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = PromptTemplate(
        template="""You are a helpful assistant that answers questions about Bank of Maharashtra loan products.
        Use only the given context to answer. If the answer is not in the context, say "I don’t know."

        Context:
        {context}

        Question: {question}
        Answer:""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

# Streamlit Chat UI
st.set_page_config(page_title="🏦 Loan Product Assistant", page_icon="💰", layout="centered")

st.title("🏦 Bank of Maharashtra Loan Assistant")
st.markdown("🤖 Your smart chatbot for loan queries!")

# Initialize chat history properly
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

qa = create_rag_pipeline()

# Input box at bottom
query = st.chat_input("Ask me about Bank of Maharashtra loans...")

if query:
    with st.spinner("Thinking... 🤔"):
        result = qa(query)

    if isinstance(result, dict):
        answer = result.get("result", "No answer returned")
        context_chunks = [doc.page_content for doc in result.get("source_documents", [])]
    else:
        answer = str(result)
        context_chunks = []

    # Always append dict with proper keys
    st.session_state.chat_history.append({
        "user": query,
        "bot": answer,
        "context": context_chunks
    })

# Display chat history
for chat in st.session_state.chat_history:
    if isinstance(chat, dict):
        # User message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(chat.get("user", ""))

        # Bot response
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(chat.get("bot", ""))
            if chat.get("context"):
                with st.expander("📖 See Retrieved Context"):
                    for j, chunk in enumerate(chat["context"], 1):
                        st.markdown(f"**Chunk {j}:** {chunk}")
    else:
        # Fallback in case old chat entries are just strings
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(str(chat))

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
