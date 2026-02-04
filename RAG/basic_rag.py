import os
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import OpenAIEmbeddings  # veya başka
from langchain_community.vectorstores import Chroma

model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

#-----------------------------------------------------------
# 2. LOAD DOCUMENTS
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
print(f"Loaded {len(docs)} document(s)")
print(f"Total characters: {len(docs[0].page_content)}")


#-----------------------------------------------------------
# 3. SPLIT DOCUMENTS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
splits = text_splitter.split_documents(docs)
print(f"Split into {len(splits)} chunks")


#-----------------------------------------------------------
# 4. CREATE VECTOR STORE
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector store created!")


#-----------------------------------------------------------
# 5. QUERY
"""
5. Query adımı:
- Soruyu vektöre çeviriyor
- En benzer 2 chunk'ı buluyor. 
--> (query, k=2) kısmındaki k paremetresi, en yakın kaç adet chunk'ı bulacağını belirtir. 
"""
query = "What is task decomposition?"
results = vectorstore.similarity_search(query, k=2)

print("\n--- Retrieved Context ---")
for i, doc in enumerate(results):
    print(f"\nChunk {i+1}:")
    print(doc.page_content[:200])


#-----------------------------------------------------------
# 6. RAG CHAIN (Simple)
"""
Ne yapıyor?

- Bulunan chunk'ları birleştiriyor
- Prompt oluşturuyor (context + soru)
- Claude'a gönderiyor
- Yanıt alıyor

"""
from langchain_core.prompts import ChatPromptTemplate

template = """Use the following context to answer the question:

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Combine context
context = "\n\n".join([doc.page_content for doc in results])

# Generate answer
chain = prompt | model
response = chain.invoke({"context": context, "question": query})
print("\n--- Answer ---")
print(response.content)