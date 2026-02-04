# chroma_interactive.py - İnteraktif Chroma DB Arayüzü

"""
İNTERAKTİF CHROMA DB ARAYÜZÜ
Bu script size bir "terminal arayüzü" sağlar
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("🚀 İNTERAKTİF CHROMA DB ARAYÜZÜ")
print("="*60)

# Embedding ve vectorstore yükle
print("\n⏳ Yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

collection = vectorstore._collection
print(f"✅ Hazır! ({collection.count()} döküman yüklendi)")

print("\n" + "="*60)
print("KOMUTLAR:")
print("  - Bir soru yazın → Arama yapar")
print("  - 'list' → İlk 5 dökümanı listeler")
print("  - 'stats' → İstatistikleri gösterir")
print("  - 'search:5 [soru]' → 5 sonuçla arama")
print("  - 'exit' → Çıkış")
print("="*60)

while True:
    try:
        user_input = input("\n💬 > ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() == 'exit':
            print("👋 Görüşmek üzere!")
            break
            
        elif user_input.lower() == 'stats':
            total = collection.count()
            all_docs = collection.get(include=['documents'])
            total_chars = sum(len(doc) for doc in all_docs['documents'])
            avg = total_chars / total if total > 0 else 0
            
            print(f"\n📊 İSTATİSTİKLER:")
            print(f"   Toplam döküman: {total}")
            print(f"   Toplam karakter: {total_chars:,}")
            print(f"   Ortalama boyut: {avg:.0f} karakter")
            
        elif user_input.lower() == 'list':
            data = collection.get(limit=5, include=['documents', 'metadatas'])
            print(f"\n📋 İLK 5 DÖKÜMAN:")
            for i, (doc, meta) in enumerate(zip(data['documents'], data['metadatas']), 1):
                print(f"\n{i}. {doc[:100]}...")
                print(f"   Kaynak: {meta.get('source', 'N/A')}")
                
        elif user_input.lower().startswith('search:'):
            # Format: search:5 soru metni
            parts = user_input.split(' ', 1)
            k = int(parts[0].split(':')[1])
            query = parts[1] if len(parts) > 1 else ""
            
            if query:
                results = vectorstore.similarity_search_with_score(query, k=k)
                print(f"\n🔍 '{query}' için {len(results)} sonuç:")
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\n{i}. (Skor: {score:.4f})")
                    print(f"   {doc.page_content[:200]}...")
            else:
                print("❌ Soru yazmalısınız!")
                
        else:
            # Normal arama
            results = vectorstore.similarity_search_with_score(user_input, k=3)
            print(f"\n🔍 Arama sonuçları ({len(results)} bulundu):")
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n{'─'*60}")
                print(f"SONUÇ {i} (Benzerlik: {score:.4f})")
                print(f"{'─'*60}")
                print(doc.page_content[:300])
                print("...")
                
    except KeyboardInterrupt:
        print("\n\n👋 Çıkış yapılıyor...")
        break
    except Exception as e:
        print(f"❌ Hata: {e}")
        print("Lütfen tekrar deneyin.")
