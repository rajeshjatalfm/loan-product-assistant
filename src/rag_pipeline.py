import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # 👈 Groq integration

def build_chroma_db(chunks):
    texts = [c["text"] for c in chunks]
    # Initialize embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory="../loan/data/chroma_db"
    )
    vectordb.persist()
    return vectordb

def create_rag_pipeline(chunks):
    vectordb = build_chroma_db(chunks)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 👇 Use Groq LLM
    llm = ChatGroq(
        groq_api_key="",  
        model="llama-3.3-70b-versatile",  # example model (you can also use mixtral, gemma etc.)
        temperature=1
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
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

if __name__ == "__main__":
    # Load processed knowledge base
    with open("../loan/data_scraped/rec_knowledge_base.json") as f:
        chunks = json.load(f)
    print(chunks)
    qa = create_rag_pipeline(chunks)

    # Example query
    query = "What is the maximum tenure for a personal loan?"
    result = qa.run(query)
    print("Q:", query)
    print("A:", result)
