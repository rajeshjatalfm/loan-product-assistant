# import json
# import re,os

# def clean_text(text):
#     text = re.sub(r"\s+", " ", text)  # remove extra spaces/newlines
#     return text.strip()

# def chunk_text(text, chunk_size=500):
#     words = text.split()
#     for i in range(0, len(words), chunk_size):
#         yield " ".join(words[i:i+chunk_size])

# def main():
#     with open("../loan/data_scraped/raw_data.json") as f:
#         raw_data = json.load(f)
    
#     chunks = []
#     for url, content in raw_data.items():
#         clean = clean_text(content)
#         chunks.append({"url": url, "text": clean})
#         # Uncomment below to enable chunking
#         # for chunk in chunk_text(clean, 500):
#         #     chunks.append({"url": url, "text": chunk})
            
#     output_file = "../loan/data_scraped/knowledge_base.json"

#     if os.path.exists(output_file):
#         os.remove(output_file)
#     with open(output_file, "w") as f:
#         json.dump(chunks, f, indent=2)

# if __name__ == "__main__":
#     main()

import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    # Load scraped raw data
    output_file = "../loan/data_scraped/rec_knowledge_base.json"

    with open("../loan/data_scraped/raw_data.json") as f:
        raw_data = json.load(f)

    all_texts = []
    for url, content in raw_data.items():
        all_texts.append(url+content)

    # Recursive chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # max tokens/chars per chunk
        chunk_overlap=50,    # keep some overlap for context
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for text in all_texts:
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk})

    # Save processed knowledge base
    with open(output_file, "w") as f:
        json.dump(chunks, f, indent=2)

if __name__ == "__main__":
    main()
