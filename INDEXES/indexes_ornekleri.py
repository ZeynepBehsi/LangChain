"""
🎯 LANGCHAIN INDEXES NEDİR?
===========================

Index = Dökümanların organize edilmiş ve aranabilir hali

Analoji:
- Kitap indeksi → Kelime ara → Sayfa numarası bul
- LangChain indeksi → Soru sor → İlgili dökümanları bul
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os

print("="*70)
print("📚 LANGCHAIN INDEXES - KAPSAMLI REHBER")
print("="*70)


# ============================================================================
# 1. INDEX NEDİR?
# ============================================================================
print("\n" + "="*70)
print("1️⃣ INDEX NEDİR?")
print("="*70)

print("""
Index 3 şeyden oluşur:

1. DOCUMENTS (Dökümanlar)
   ├─ Text içeriği
   ├─ Metadata (kaynak, tarih, vs)
   └─ Unique ID

2. EMBEDDINGS (Vektör Temsilleri)
   ├─ Her döküman → sayısal vektör
   ├─ Anlamsal benzerlik için
   └─ 768, 1536 boyutlu vektörler

3. VECTOR STORE (Depolama)
   ├─ Vektörleri saklar
   ├─ Similarity search yapar
   └─ Chroma, Pinecone, FAISS, vs.

┌──────────────────────────────────────┐
│  Index = Documents + Embeddings      │
│                    + Vector Store     │
└──────────────────────────────────────┘
""")


# ============================================================================
# 2. TEMEL INDEX OLUŞTURMA
# ============================================================================
print("\n" + "="*70)
print("2️⃣ TEMEL INDEX OLUŞTURMA")
print("="*70)

# Örnek dökümanlar oluştur
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Python programlama dili 1991'de Guido van Rossum tarafından geliştirildi.",
        metadata={"source": "python_history", "topic": "programming"}
    ),
    Document(
        page_content="Machine Learning veri ile öğrenen algoritmalar geliştirme bilimidir.",
        metadata={"source": "ml_basics", "topic": "ai"}
    ),
    Document(
        page_content="Vector database'ler embedding'leri saklar ve similarity search yapar.",
        metadata={"source": "vector_db", "topic": "database"}
    ),
    Document(
        page_content="LangChain LLM uygulamaları geliştirmek için bir framework'tür.",
        metadata={"source": "langchain_intro", "topic": "framework"}
    )
]

print(f"📄 {len(documents)} döküman oluşturuldu")
print("\nÖrnek Döküman:")
print(f"  Content: {documents[0].page_content[:50]}...")
print(f"  Metadata: {documents[0].metadata}")

# Embedding modeli
print("\n🔢 Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# INDEX OLUŞTUR!
print("\n📊 Index oluşturuluyor (Vector Store)...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./demo_index"
)

print("✅ Index oluşturuldu!")
print(f"   Lokasyon: ./demo_index")
print(f"   Döküman sayısı: {len(documents)}")


# ============================================================================
# 3. INDEX'TEN ARAMA YAPMA
# ============================================================================
print("\n" + "="*70)
print("3️⃣ INDEX'TEN ARAMA YAPMA")
print("="*70)

print("\n🔍 Arama Yöntem 1: Similarity Search")
print("-" * 70)

query = "Python ne zaman yapıldı?"
results = vectorstore.similarity_search(query, k=2)

print(f"Soru: {query}")
print(f"Bulunan: {len(results)} döküman\n")

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
    print(f"   Kaynak: {doc.metadata['source']}\n")


print("\n🔍 Arama Yöntem 2: Similarity Search with Score")
print("-" * 70)

query = "Machine learning nedir?"
results_with_scores = vectorstore.similarity_search_with_score(query, k=2)

print(f"Soru: {query}\n")

for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"{i}. Score: {score:.4f}")
    print(f"   {doc.page_content}")
    print(f"   Metadata: {doc.metadata}\n")


print("\n🔍 Arama Yöntem 3: MMR (Maximum Marginal Relevance)")
print("-" * 70)
print("MMR = Hem alakalı hem de birbirinden farklı sonuçlar")

query = "database"
results_mmr = vectorstore.max_marginal_relevance_search(query, k=2)

print(f"Soru: {query}\n")
for i, doc in enumerate(results_mmr, 1):
    print(f"{i}. {doc.page_content[:60]}...")


# ============================================================================
# 4. RETRIEVER KULLANIMI
# ============================================================================
print("\n" + "="*70)
print("4️⃣ RETRIEVER: INDEX'İ SORGULAMAK İÇİN ARAÇ")
print("="*70)

print("""
Retriever = Index'ten döküman getiren araç

Vector Store vs Retriever:
├─ Vector Store: Ham depolama + arama
└─ Retriever: Standartlaştırılmış arayüz (LangChain chain'lerde kullanılır)
""")

# Retriever oluştur
retriever = vectorstore.as_retriever(
    search_type="similarity",  # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 2}     # En iyi 2 sonuç
)

print("📡 Retriever oluşturuldu")
print(f"   Search Type: similarity")
print(f"   K: 2 (en iyi 2 sonuç)")

# Retriever ile arama
print("\n🔍 Retriever ile arama:")
query = "LangChain nedir?"
docs = retriever.invoke(query)

print(f"Soru: {query}")
print(f"Sonuç: {len(docs)} döküman\n")
for doc in docs:
    print(f"  • {doc.page_content[:60]}...")


# ============================================================================
# 5. FARKLI RETRIEVER TİPLERİ
# ============================================================================
print("\n" + "="*70)
print("5️⃣ FARKLI RETRIEVER TİPLERİ")
print("="*70)

print("\n📌 Tip 1: Similarity (Varsayılan)")
retriever_similarity = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)
print("✅ Benzerlik bazlı arama")

print("\n📌 Tip 2: MMR (Çeşitlilik)")
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 4,      # 4 aday getir
        "lambda_mult": 0.5 # 0=çeşitlilik, 1=benzerlik
    }
)
print("✅ Çeşitli sonuçlar için MMR")

print("\n📌 Tip 3: Score Threshold (Eşik değer)")
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # 0.8'den yüksek skorlar
        "k": 3
    }
)
print("✅ Sadece yüksek skorlu sonuçlar")


# Test edelim
print("\n" + "-"*70)
print("🧪 KARŞILAŞTIRMA TESTİ")
print("-"*70)

test_query = "programlama"

print(f"\nSoru: '{test_query}'\n")

print("1. Similarity Retriever:")
results_1 = retriever_similarity.invoke(test_query)
for doc in results_1:
    print(f"   • {doc.page_content[:50]}...")

print("\n2. MMR Retriever (çeşitlilik):")
results_2 = retriever_mmr.invoke(test_query)
for doc in results_2:
    print(f"   • {doc.page_content[:50]}...")


# ============================================================================
# 6. INDEX GÜNCELLEME
# ============================================================================
print("\n" + "="*70)
print("6️⃣ INDEX GÜNCELLEME")
print("="*70)

print("\n➕ Yeni döküman ekleme:")
new_docs = [
    Document(
        page_content="React modern web uygulamaları için JavaScript kütüphanesidir.",
        metadata={"source": "react_intro", "topic": "frontend"}
    )
]

# Mevcut index'e ekle
vectorstore.add_documents(new_docs)
print("✅ 1 yeni döküman eklendi")

# Kontrol et
print("\n🔍 Kontrol arama:")
results = vectorstore.similarity_search("JavaScript", k=1)
print(f"Bulunan: {results[0].page_content}")


# ============================================================================
# 7. METADATA FİLTRELEME
# ============================================================================
print("\n" + "="*70)
print("7️⃣ METADATA İLE FİLTRELEME")
print("="*70)

print("\n🎯 Sadece 'topic=ai' olan dökümanları ara:")

# Metadata filtresi ile retriever
retriever_filtered = vectorstore.as_retriever(
    search_kwargs={
        "k": 2,
        "filter": {"topic": "ai"}
    }
)

results = retriever_filtered.invoke("algoritma")
print(f"Sonuç sayısı: {len(results)}")
for doc in results:
    print(f"  • {doc.page_content[:60]}...")
    print(f"    Topic: {doc.metadata['topic']}")


# ============================================================================
# 8. INDEX KAYDETME VE YÜKLEME
# ============================================================================
print("\n" + "="*70)
print("8️⃣ INDEX KAYDETME VE YÜKLEME")
print("="*70)

print("""
Chroma otomatik olarak kaydeder (persist_directory belirttiyseniz)

Yükleme:
vectorstore = Chroma(
    persist_directory="./demo_index",
    embedding_function=embeddings
)
""")

# Mevcut index'i yükle
loaded_vectorstore = Chroma(
    persist_directory="./demo_index",
    embedding_function=embeddings
)

print("✅ Index yüklendi (disk'ten)")
print("\n🔍 Test arama:")
results = loaded_vectorstore.similarity_search("Python", k=1)
print(f"Sonuç: {results[0].page_content}")


# ============================================================================
# 9. INDEX İSTATİSTİKLERİ
# ============================================================================
print("\n" + "="*70)
print("9️⃣ INDEX İSTATİSTİKLERİ")
print("="*70)

collection = loaded_vectorstore._collection

print(f"""
📊 Index Bilgileri:
   • Collection: {collection.name}
   • Toplam döküman: {collection.count()}
   • Lokasyon: ./demo_index
   • Embedding boyutu: 768 (all-mpnet-base-v2)
""")


# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "="*70)
print("📚 ÖZET: LANGCHAIN INDEXES")
print("="*70)

print("""
✅ ÖĞRENDİKLERİMİZ:

1. Index = Documents + Embeddings + Vector Store
2. Oluşturma: Chroma.from_documents()
3. Arama: similarity_search(), similarity_search_with_score()
4. Retriever: Standart arayüz (chain'lerde kullanılır)
5. Retriever tipleri: similarity, mmr, score_threshold
6. Güncelleme: add_documents()
7. Filtreleme: metadata ile
8. Persistence: Otomatik kaydedilir

🎯 KULLANIM ALANLARI:

• RAG Sistemleri (zaten yaptık!)
• Question Answering
• Semantic Search
• Document Retrieval
• Knowledge Bases

📝 SONRAKI ADIMLAR:

1. ✅ indexes_ornekleri.py çalıştırın
2. 📖 INDEXES_REHBER.md okuyun
3. 🛠️ Kendi index'inizi oluşturun
""")

print("\n" + "="*70)
print("✅ DEMO TAMAMLANDI!")
print("="*70)
