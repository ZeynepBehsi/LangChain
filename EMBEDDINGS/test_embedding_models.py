# test_embedding_models.py - Farklı Embedding Modellerini Test Et

"""
Farklı embedding modellerini karşılaştırma
"""

import time
from langchain_community.embeddings import HuggingFaceEmbeddings

# Test metni
test_text = "What is task decomposition in AI agents?"

print("🔬 EMBEDDING MODEL KARŞILAŞTIRMASI\n")
print("="*60)

# MODEL 1: all-mpnet-base-v2 (Şu anki modeliniz)
print("\n1️⃣ all-mpnet-base-v2 (Mevcut)")
print("-"*60)
start = time.time()
embeddings1 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
vector1 = embeddings1.embed_query(test_text)
elapsed1 = time.time() - start

print(f"✅ Vektör boyutu: {len(vector1)}")
print(f"⏱️  Süre: {elapsed1:.2f} saniye")
print(f"🔢 İlk 5 değer: {vector1[:5]}")

# MODEL 2: all-MiniLM-L6-v2 (Daha küçük, hızlı)
print("\n2️⃣ all-MiniLM-L6-v2 (Hızlı)")
print("-"*60)
start = time.time()
embeddings2 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector2 = embeddings2.embed_query(test_text)
elapsed2 = time.time() - start

print(f"✅ Vektör boyutu: {len(vector2)}")
print(f"⏱️  Süre: {elapsed2:.2f} saniye")
print(f"🔢 İlk 5 değer: {vector2[:5]}")

# KARŞILAŞTIRMA
print("\n" + "="*60)
print("📊 KARŞILAŞTIRMA")
print("="*60)
print(f"Model 1 boyut: {len(vector1)} | Model 2 boyut: {len(vector2)}")
print(f"Model 1 süre: {elapsed1:.2f}s | Model 2 süre: {elapsed2:.2f}s")
print(f"Hız farkı: {elapsed1/elapsed2:.2f}x")

print("\n💡 Sonuç:")
if elapsed2 < elapsed1:
    print(f"   Model 2, {elapsed1/elapsed2:.1f}x daha hızlı!")
print(f"   Ama Model 1 daha büyük vektör = daha iyi doğruluk")
print(f"   Tercih sizin: Hız mı? Doğruluk mu?")
