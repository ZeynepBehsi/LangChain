# chroma_explorer.py - Chroma DB'yi Keşfetme Rehberi

"""
CHROMA DB NEDİR?
- Vector (sayısal vektör) database
- Varsayılan olarak grafiksel arayüzü YOK
- Python kodu ile erişilir
- ./chroma_db klasöründe SQLite olarak saklanır
"""

print("🔍 CHROMA DB KEŞİF ARACI")
print("="*60)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import json

# ADIM 1: Embedding modelini yükle (aynısını kullanmalıyız!)
print("\n📦 Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
print("✅ Model yüklendi")

# ADIM 2: Chroma DB'ye bağlan
print("\n🔌 Chroma DB'ye bağlanıyorum...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
print("✅ Bağlantı başarılı")

# ADIM 3: Database bilgilerini al
print("\n" + "="*60)
print("📊 DATABASE BİLGİLERİ")
print("="*60)

collection = vectorstore._collection
total_docs = collection.count()

print(f"📁 Konum: ./chroma_db/")
print(f"📚 Toplam Döküman: {total_docs}")
print(f"🔢 Collection Adı: {collection.name}")

# ADIM 4: Tüm dökümanları listele (ilk 10)
print("\n" + "="*60)
print("📋 İLK 10 DÖKÜMAN")
print("="*60)

# Tüm verileri çek
all_data = collection.get(
    limit=10,
    include=['documents', 'metadatas', 'embeddings']
)

for i, (doc, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas']), 1):
    print(f"\n{'─'*60}")
    print(f"📄 DÖKÜMAN {i}")
    print(f"{'─'*60}")
    print(f"📝 İçerik (ilk 200 karakter):")
    print(f"   {doc[:200]}...")
    print(f"\n📌 Metadata:")
    print(f"   Kaynak: {metadata.get('source', 'N/A')}")
    print(f"   Başlangıç: {metadata.get('start_index', 'N/A')}")

# ADIM 5: Örnek aramalar
print("\n" + "="*60)
print("🔍 ÖRNEK ARAMALAR")
print("="*60)

test_queries = [
    "What is task decomposition?",
    "Types of memory",
    "What is ReAct?"
]

for query in test_queries:
    print(f"\n❓ Soru: '{query}'")
    results = vectorstore.similarity_search(query, k=1)
    if results:
        print(f"✅ Bulunan en yakın chunk (ilk 150 karakter):")
        print(f"   {results[0].page_content[:150]}...")

# ADIM 6: Skorlu arama (benzerlik skorları ile)
print("\n" + "="*60)
print("📈 SKORLU ARAMA (Benzerlik Skorları)")
print("="*60)

query = "What is agent?"
print(f"\n❓ Soru: '{query}'")
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\n{i}. Sonuç (Skor: {score:.4f}) - Düşük skor = Daha iyi")
    print(f"   {doc.page_content[:150]}...")

# ADIM 7: Collection metadata
print("\n" + "="*60)
print("🔧 COLLECTION DETAYLARI")
print("="*60)

metadata = collection.metadata
print(f"📋 Metadata: {json.dumps(metadata, indent=2)}")

# ADIM 8: İstatistikler
print("\n" + "="*60)
print("📊 İSTATİSTİKLER")
print("="*60)

# Tüm dökümanları çek
all_docs = collection.get(include=['documents'])
all_contents = all_docs['documents']

total_chars = sum(len(doc) for doc in all_contents)
avg_chars = total_chars / len(all_contents) if all_contents else 0

print(f"📏 Toplam karakter: {total_chars:,}")
print(f"📊 Ortalama chunk boyutu: {avg_chars:.0f} karakter")
print(f"📦 En uzun chunk: {max(len(doc) for doc in all_contents)} karakter")
print(f"📦 En kısa chunk: {min(len(doc) for doc in all_contents)} karakter")

print("\n" + "="*60)
print("✅ KEŞİF TAMAMLANDI!")
print("="*60)
