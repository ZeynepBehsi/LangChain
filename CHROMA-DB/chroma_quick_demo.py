# chroma_quick_demo.py - Hızlı Chroma DB Erişim Demo

print("🔄 Chroma DB'ye bağlanıyor...")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Aynı embedding modelini kullan
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

print("✅ Embedding model yüklendi")

# Mevcut database'e bağlan
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print("✅ Chroma DB'ye bağlandı\n")

# Basit bir arama yap
query = "What are the types of memory?"
print(f"🔍 Soru: {query}\n")

results = vectorstore.similarity_search(query, k=2)

print(f"📚 {len(results)} sonuç bulundu:\n")
for i, doc in enumerate(results, 1):
    print(f"{'='*60}")
    print(f"SONUÇ {i}:")
    print(f"{'='*60}")
    print(doc.page_content[:300])
    print("...\n")

# Database bilgisi
collection = vectorstore._collection
print(f"📊 Toplam döküman sayısı: {collection.count()}")
