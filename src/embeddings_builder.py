import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def build_chroma_db(input_file="../loan/data_scraped/rec_knowledge_base.json",
                    persist_dir="../loan/data/chroma_db"):
    # Load processed chunks
    with open(input_file) as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Create and persist ChromaDB
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"✅ Embeddings built and saved to {persist_dir}")

if __name__ == "__main__":
    build_chroma_db()
