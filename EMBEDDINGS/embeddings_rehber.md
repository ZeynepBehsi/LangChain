# EMBEDDINGS REHBERİ 🧠

## 📚 İÇİNDEKİLER
1. [Embedding Nedir?](#embedding-nedir)
2. [Nasıl Çalışır?](#nasıl-çalışır)
3. [Neden Önemli?](#neden-önemli)
4. [Mevcut Modeller](#mevcut-modeller)
5. [Model Seçimi](#model-seçimi)
6. [Kullanım Örnekleri](#kullanım-örnekleri)
7. [Önemli Notlar](#önemli-notlar)

---

## 🤔 EMBEDDING NEDİR?

**Embedding** = Metni sayısal vektörlere çevirme işlemi

```python
# Örnek
"kedi"      → [0.2, 0.8, 0.1, 0.5, 0.3, ...] (768 boyutlu vektör)
"köpek"     → [0.3, 0.7, 0.2, 0.4, 0.4, ...]
"bilgisayar"→ [0.9, 0.1, 0.8, 0.2, 0.1, ...]
```

### 🎯 Anlamsal Benzerlik

Vektörler arası mesafe = Anlamsal yakınlık

```
Mesafe(kedi, köpek)      = 0.15  ✅ Yakın (her ikisi de hayvan)
Mesafe(kedi, bilgisayar) = 0.87  ❌ Uzak (ilgisiz)
```

---

## ⚙️ NASIL ÇALIŞIR?

### Adım 1: Model Yükleme
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

### Adım 2: Metin → Vektör Dönüşümü
```python
text = "What is artificial intelligence?"
vector = embeddings.embed_query(text)

print(len(vector))  # 768 (vektör boyutu)
print(vector[:5])   # [0.234, -0.123, 0.456, ...]
```

### Adım 3: Benzerlik Hesaplama
```python
# Cosine similarity ile karşılaştırma
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity([vector1], [vector2])
```

---

## 🎯 NEDEN ÖNEMLİ?

### 1️⃣ **Anlamsal Arama**
Kelime eşleşmesi değil, anlam eşleşmesi!

```
Soru: "How to make AI agents?"

❌ Klasik Arama: "make" kelimesini arar
✅ Embedding Arama: "create", "build", "develop" de bulur
```

### 2️⃣ **RAG Sistemlerinin Temeli**
```
Döküman → Embedding → Vector DB → Arama → İlgili Chunk
```

### 3️⃣ **Çok Dilli Destek**
```python
# Türkçe → İngilizce eşleşme
"yapay zeka" ≈ "artificial intelligence"
```

### 4️⃣ **Verimli Depolama**
```
43,000 karakter metin → 63 chunk → 63 × 768 sayı
```

---

## 📦 MEVCUT MODELLER

Bu klasördeki [embedding_models.py](embedding_models.py) dosyasında 10 farklı model var:

### 🥇 EN POPÜLER MODELLER

#### 1. HuggingFace Embeddings (Ücretsiz) ⭐
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

**Avantajlar:**
- ✅ Tamamen ücretsiz
- ✅ Yerel çalışır (gizlilik)
- ✅ İnternet gerekmez (model indirildikten sonra)

**Dezavantajlar:**
- ❌ İlk kullanımda model indirme süresi
- ❌ RAM kullanır (~1-2GB)

**Popüler Modeller:**
- `all-mpnet-base-v2` → Dengeli (768 boyut)
- `all-MiniLM-L6-v2` → Hızlı (384 boyut)
- `paraphrase-multilingual-MiniLM-L12-v2` → Türkçe destekli
- `BAAI/bge-large-en-v1.5` → En iyi doğruluk (1024 boyut)

---

#### 2. OpenAI Embeddings (Ücretli)
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # veya "text-embedding-3-large"
    api_key="your-api-key"
)
```

**Avantajlar:**
- ✅ En iyi kalite
- ✅ Çok hızlı
- ✅ RAM kullanmaz

**Dezavantajlar:**
- ❌ Ücretli (~$0.02 per 1M token)
- ❌ İnternet gerekir
- ❌ Gizlilik endişeleri

---

#### 3. Cohere Embeddings (Ücretli)
```python
from langchain_community.embeddings import CohereEmbeddings

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",  # Türkçe destekli!
    cohere_api_key="your-api-key"
)
```

**Öne Çıkan Özellik:** Çok dilli destek (100+ dil, Türkçe dahil!)

---

#### 4. Ollama Embeddings (Yerel/Ücretsiz)
```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="llama2",
    base_url="http://localhost:11434"
)
```

**Önce Ollama kurulumu gerekir:**
```bash
# Mac
brew install ollama

# Model indirme
ollama pull llama2
```

---

## 🎯 MODEL SEÇİMİ

### Kullanım Senaryolarına Göre:

| Senaryo | Önerilen Model | Neden? |
|---------|---------------|---------|
| 🎓 **Öğrenme/Prototip** | HuggingFace `all-mpnet-base-v2` | Ücretsiz, kolay başlangıç |
| 🚀 **Production (Kalite)** | OpenAI `text-embedding-3-large` | En iyi doğruluk |
| 💰 **Production (Bütçe)** | HuggingFace `BAAI/bge-large-en-v1.5` | Ücretsiz + iyi kalite |
| ⚡ **Hız Öncelikli** | HuggingFace `all-MiniLM-L6-v2` | Küçük, hızlı |
| 🌍 **Türkçe Destek** | Cohere `embed-multilingual-v3.0` | 100+ dil |
| 🔒 **Gizlilik/GDPR** | HuggingFace veya Ollama | Yerel çalışır |
| ☁️ **Cloud Native** | OpenAI veya Cohere | Managed service |

---

### Karşılaştırma Tablosu:

| Özellik | HuggingFace | OpenAI | Cohere | Ollama |
|---------|-------------|---------|---------|---------|
| **Ücret** | Ücretsiz | Ücretli | Ücretli | Ücretsiz |
| **Kalite** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Hız** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Kurulum** | Kolay | Çok Kolay | Çok Kolay | Orta |
| **İnternet** | İlk kez | Her zaman | Her zaman | Hayır |
| **Gizlilik** | Yüksek | Düşük | Düşük | Yüksek |
| **RAM Kullanımı** | 1-2GB | 0GB | 0GB | 1-2GB |
| **Türkçe** | Bazı modeller | Var | Mükemmel | Model'e göre |

---

## 💻 KULLANIM ÖRNEKLERİ

### Örnek 1: Tek Metin Embedding
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Tek metin
text = "What is machine learning?"
vector = embeddings.embed_query(text)

print(f"Vektör boyutu: {len(vector)}")
print(f"İlk 5 değer: {vector[:5]}")
```

**Çıktı:**
```
Vektör boyutu: 768
İlk 5 değer: [0.234, -0.123, 0.456, 0.789, -0.321]
```

---

### Örnek 2: Çoklu Metin Embedding
```python
texts = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Python is a programming language"
]

vectors = embeddings.embed_documents(texts)

print(f"Toplam vektör sayısı: {len(vectors)}")
print(f"Her vektör boyutu: {len(vectors[0])}")
```

---

### Örnek 3: Benzerlik Hesaplama
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

text1 = "artificial intelligence"
text2 = "machine learning"
text3 = "cooking recipe"

v1 = embeddings.embed_query(text1)
v2 = embeddings.embed_query(text2)
v3 = embeddings.embed_query(text3)

sim_1_2 = cosine_similarity([v1], [v2])[0][0]
sim_1_3 = cosine_similarity([v1], [v3])[0][0]

print(f"AI ↔ ML: {sim_1_2:.3f}")      # Yüksek (ilgili)
print(f"AI ↔ Cooking: {sim_1_3:.3f}") # Düşük (ilgisiz)
```

**Çıktı:**
```
AI ↔ ML: 0.856      ✅ Yüksek benzerlik
AI ↔ Cooking: 0.234 ❌ Düşük benzerlik
```

---

### Örnek 4: RAG'de Kullanım
```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Vector store oluşturma
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Arama
results = vectorstore.similarity_search("What is task decomposition?", k=3)
```

---

## ⚠️ ÖNEMLİ NOTLAR

### 1️⃣ Model Değiştirirken Dikkat!

❌ **YANLIŞ:**
```python
# İlk kez
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vectorstore = Chroma.from_documents(docs, embeddings)

# Sonra model değişti
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# ❌ HATA! Boyutlar uyumsuz (768 vs 384)
```

✅ **DOĞRU:**
```python
# Model değiştirdiyseniz database'i yeniden oluşturun
import shutil
shutil.rmtree("./chroma_db")  # Eski DB'yi sil

# Yeni model ile yeniden oluştur
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
```

---

### 2️⃣ Vektör Boyutları

Farklı modeller farklı boyutlar üretir:

| Model | Boyut |
|-------|-------|
| `all-mpnet-base-v2` | 768 |
| `all-MiniLM-L6-v2` | 384 |
| `BAAI/bge-large-en-v1.5` | 1024 |
| `text-embedding-3-small` (OpenAI) | 1536 |
| `text-embedding-3-large` (OpenAI) | 3072 |

**Kural:** Aynı database için hep aynı modeli kullanın!

---

### 3️⃣ İlk Çalıştırma Yavaş Olabilir

```python
# İlk kez
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
# ⏳ Model indiriliyor... (~400MB, 1-2 dakika)

# İkinci kez
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
# ⚡ Hızlı! (cache'den yükleniyor)
```

**Model cache konumu:**
```
~/.cache/huggingface/hub/
```

---

### 4️⃣ RAM Kullanımı

```python
# Küçük model (az RAM)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# ~500MB RAM

# Büyük model (fazla RAM)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
# ~2GB RAM
```

**Sınırlı RAM varsa:** OpenAI veya Cohere (cloud) kullanın

---

### 5️⃣ Türkçe Desteği

**Türkçe için önerilen modeller:**

```python
# Seçenek 1: Multilingual HuggingFace (Ücretsiz)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Seçenek 2: Cohere Multilingual (Ücretli, En İyi)
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key="your-key"
)
```

---

## 🧪 TEST DOSYALARI

Bu klasörde test dosyaları:

### [test_embedding_models.py](test_embedding_models.py)
2 farklı modeli karşılaştırır:
```bash
python EMBEDDINGS/test_embedding_models.py
```

**Çıktı:**
- Vektör boyutları
- İşlem süreleri
- Performans karşılaştırması

---

## 📚 DAHA FAZLA BİLGİ

### Kaynaklar:
- **HuggingFace Model Hub:** https://huggingface.co/models?pipeline_tag=sentence-similarity
- **Sentence Transformers:** https://www.sbert.net/
- **OpenAI Embeddings:** https://platform.openai.com/docs/guides/embeddings
- **Cohere Embeddings:** https://docs.cohere.com/docs/embeddings

### İlgili Dosyalar:
- [embedding_models.py](embedding_models.py) - 10 farklı model örneği
- [test_embedding_models.py](test_embedding_models.py) - Model karşılaştırma testi
- `../RAG/basic_rag.py` - RAG implementasyonunda kullanım

---

## 🎓 ÖZET

✅ **Embedding** = Metin → Sayısal vektör dönüşümü
✅ **Amaç** = Anlamsal benzerlik ölçmek
✅ **Kullanım** = RAG sistemlerinde arama
✅ **Seçim** = İhtiyaca göre (hız/kalite/maliyet)
✅ **Dikkat** = Model değişince DB yenile!

---

## 💡 HIZLI BAŞLANGIÇ

**En basit kurulum:**
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Model oluştur
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Kullan
text = "Hello, world!"
vector = embeddings.embed_query(text)

print(f"✅ Embedding oluşturuldu! Boyut: {len(vector)}")
```

**İyi öğrenmeler!** 🚀
