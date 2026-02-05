# 🎯 LangChain INDEXES - Detaylı Rehber

## İçindekiler
1. [Index Nedir?](#index-nedir)
2. [Index Bileşenleri](#index-bileşenleri)
3. [Index Oluşturma](#index-oluşturma)
4. [Arama Yöntemleri](#arama-yöntemleri)
5. [Retriever Kullanımı](#retriever-kullanımı)
6. [Farklı Vector Store'lar](#farklı-vector-storelar)
7. [İleri Seviye](#ileri-seviye)
8. [Best Practices](#best-practices)

---

## Index Nedir?

**Index**, dökümanların organize edilmiş ve hızlı arama yapılabilir halidir.

### Günlük Hayattan Analoji

```
📖 Kitap İndeksi:
   "Python" kelimesini ara
   → İndekse bak
   → Sayfa 42, 78, 156
   → Sayfaya git

🔍 LangChain İndeksi:
   "Python nedir?" sorusunu sor
   → Index'te ara
   → İlgili 3 dökümanı bul
   → Dökümanları döndür
```

### Neden Index?

```python
# ❌ Index OLMADAN (her seferinde baştan tara)
for doc in 1_000_000_documents:
    if doc.contains(query):
        results.append(doc)
# ⏱️ Çok yavaş! O(n)

# ✅ Index İLE (önceden hazırlanmış)
results = index.search(query)
# ⚡ Çok hızlı! O(log n) veya O(1)
```

---

## Index Bileşenleri

LangChain Index 3 ana bileşenden oluşur:

### 1. Documents (Dökümanlar)

```python
from langchain.schema import Document

doc = Document(
    page_content="LangChain bir LLM framework'üdür.",
    metadata={
        "source": "langchain.com",
        "author": "Harrison Chase",
        "date": "2024-01-15",
        "category": "tutorial"
    }
)

# Document yapısı:
# ├─ page_content: str (döküman içeriği)
# └─ metadata: dict (ek bilgiler)
```

### 2. Embeddings (Vektör Temsilleri)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Text → Vector dönüşümü
text = "LangChain bir framework'tür"
vector = embeddings.embed_query(text)

# vector = [0.123, -0.456, 0.789, ..., 0.234]
# 768 boyutlu sayısal vektör
```

**Embedding Boyutları:**
```
Model                        Boyut    Kullanım
─────────────────────────────────────────────
all-mpnet-base-v2            768     Genel amaçlı
text-embedding-ada-002       1536    OpenAI (ücretli)
all-MiniLM-L6-v2             384     Hızlı, hafif
instructor-xl                768     Özelleştirilebilir
```

### 3. Vector Store (Depolama)

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma(
    persist_directory="./my_index",
    embedding_function=embeddings
)

# Vector Store işlevleri:
# ├─ Vektörleri saklar
# ├─ Similarity search yapar
# ├─ Metadata filtreler
# └─ Persistent (kalıcı) depolama
```

---

## Index Oluşturma

### Yöntem 1: from_documents() - En Yaygın

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 1. Dökümanlar oluştur
documents = [
    Document(page_content="Python bir programlama dilidir."),
    Document(page_content="JavaScript web için kullanılır."),
    Document(page_content="Machine Learning AI'ın bir dalıdır.")
]

# 2. Embedding modeli
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 3. Index oluştur (tek satır!)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./my_index"
)

# ✅ Index hazır!
```

### Yöntem 2: from_texts() - Text Listesinden

```python
texts = [
    "Python bir programlama dilidir.",
    "JavaScript web için kullanılır.",
    "Machine Learning AI'ın bir dalıdır."
]

metadatas = [
    {"source": "python.txt"},
    {"source": "javascript.txt"},
    {"source": "ml.txt"}
]

vectorstore = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embeddings
)
```

### Yöntem 3: add_documents() - Mevcut Index'e Ekle

```python
# Mevcut index'i yükle
vectorstore = Chroma(
    persist_directory="./my_index",
    embedding_function=embeddings
)

# Yeni dökümanlar ekle
new_docs = [
    Document(page_content="React bir UI kütüphanesidir.")
]

vectorstore.add_documents(new_docs)
```

### Yöntem 4: Büyük Veri - Batch İşleme

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Klasördeki tüm dosyaları yükle
loader = DirectoryLoader('./data/', glob="**/*.txt")
raw_documents = loader.load()

# 2. Dökümanları parçala
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.split_documents(raw_documents)

# 3. Batch'ler halinde index'e ekle
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vectorstore.add_documents(batch)
    print(f"✅ {i+len(batch)}/{len(documents)} döküman eklendi")
```

---

## Arama Yöntemleri

### 1. similarity_search() - Temel Arama

```python
# En basit kullanım
results = vectorstore.similarity_search(
    query="Python nedir?",
    k=3  # En iyi 3 sonuç
)

for doc in results:
    print(doc.page_content)

# Çıktı:
# Python bir programlama dilidir.
# Machine Learning AI'ın bir dalıdır.
# JavaScript web için kullanılır.
```

**Parametreler:**
- `query`: str - Arama sorgusu
- `k`: int - Kaç sonuç döndürülsün (varsayılan: 4)
- `filter`: dict - Metadata filtresi (opsiyonel)

### 2. similarity_search_with_score() - Skorlu Arama

```python
results = vectorstore.similarity_search_with_score(
    query="Machine Learning nedir?",
    k=2
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
    print()

# Çıktı:
# Score: 0.2341  ← Düşük = daha benzer
# Content: Machine Learning AI'ın bir dalıdır.
#
# Score: 0.8765  ← Yüksek = daha farklı
# Content: Python bir programlama dilidir.
```

**Not:** Chroma'da score = distance (uzaklık)
- 0'a yakın = çok benzer ✅
- 1'e yakın = çok farklı ❌

### 3. max_marginal_relevance_search() - MMR

```python
# MMR = Maximum Marginal Relevance
# Hem alakalı hem de birbirinden farklı sonuçlar

results = vectorstore.max_marginal_relevance_search(
    query="programlama dilleri",
    k=3,
    fetch_k=10,      # 10 aday getir
    lambda_mult=0.5  # 0.5 = denge (alakalılık + çeşitlilik)
)

# lambda_mult:
# 0.0 = Maksimum çeşitlilik (birbirinden farklı)
# 1.0 = Maksimum alakalılık (similarity'ye eşit)
# 0.5 = Denge
```

**Ne zaman kullanılır?**
- Çeşitli bakış açıları istediğinizde
- Tekrar eden bilgilerden kaçınmak için
- Özet çıkarma için

### 4. similarity_search_by_vector() - Vektör ile Arama

```python
# Önce bir text'i vektöre çevir
query_vector = embeddings.embed_query("Python nedir?")

# Vektör ile ara
results = vectorstore.similarity_search_by_vector(
    embedding=query_vector,
    k=2
)
```

---

## Retriever Kullanımı

### Retriever Nedir?

```
Vector Store vs Retriever:

Vector Store:
├─ Ham depolama
├─ Birçok farklı arama metodu
└─ Esnek ama tutarlı değil

Retriever:
├─ Standart arayüz
├─ LangChain chain'lerle uyumlu
└─ Tutarlı API
```

### Temel Retriever

```python
# Vector store'dan retriever oluştur
retriever = vectorstore.as_retriever()

# Kullanımı çok basit
docs = retriever.invoke("Python nedir?")

# veya
docs = retriever.get_relevant_documents("Python nedir?")
```

### Retriever Konfigürasyonu

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",  # Arama tipi
    search_kwargs={            # Arama parametreleri
        "k": 4,               # En iyi 4 sonuç
        "score_threshold": 0.5 # Minimum skor
    }
)
```

### Retriever Tipleri

#### 1. Similarity Retriever (Varsayılan)

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Basit benzerlik araması
```

#### 2. MMR Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,              # 4 sonuç döndür
        "fetch_k": 20,       # 20 aday getir
        "lambda_mult": 0.5   # Denge
    }
)

# Çeşitli sonuçlar için
```

#### 3. Score Threshold Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # 0.7'den düşük skorlar (daha benzer)
        "k": 5
    }
)

# Sadece yeterince benzer olanları döndür
# Eşiği geçemezse boş liste döner
```

### Retriever ile Chain Kullanımı

```python
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic

# LLM
llm = ChatAnthropic(model="claude-3-haiku-20240307")

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Chain oluştur
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Kullan
answer = qa_chain.invoke("Python ne zaman geliştirildi?")
print(answer)
```

---

## Farklı Vector Store'lar

### 1. Chroma (Yerel, Ücretsiz)

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

**Özellikler:**
- ✅ Tamamen ücretsiz
- ✅ Yerel çalışır (internet gerektirmez)
- ✅ Persistent (kalıcı) depolama
- ✅ Metadata filtreleme
- ❌ Büyük ölçek için yavaş
- ❌ Dağıtık mimari yok

### 2. FAISS (Facebook AI, Hızlı)

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

# Kaydet
vectorstore.save_local("./faiss_index")

# Yükle
vectorstore = FAISS.load_local(
    "./faiss_index",
    embeddings=embeddings
)
```

**Özellikler:**
- ✅ Çok hızlı
- ✅ Milyonlarca vektör destekler
- ✅ GPU desteği
- ❌ Metadata filtreleme sınırlı
- ❌ Real-time güncelleme zor

### 3. Pinecone (Cloud, Ücretli)

```python
from langchain_community.vectorstores import Pinecone
import pinecone

# Initialize
pinecone.init(
    api_key="your-api-key",
    environment="us-west1-gcp"
)

# Index oluştur
vectorstore = Pinecone.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name="my-index"
)
```

**Özellikler:**
- ✅ Dağıtık mimari
- ✅ Otomatik ölçekleme
- ✅ Real-time güncelleme
- ✅ Metadata filtreleme
- ❌ Ücretli
- ❌ İnternet gerektirir

### 4. Weaviate (Hybrid Search)

```python
from langchain_community.vectorstores import Weaviate
import weaviate

client = weaviate.Client("http://localhost:8080")

vectorstore = Weaviate(
    client=client,
    index_name="MyIndex",
    text_key="text",
    embedding=embeddings
)
```

**Özellikler:**
- ✅ Hybrid search (vector + keyword)
- ✅ GraphQL API
- ✅ Schema tanımlama
- ❌ Kurulum karmaşık

### Vector Store Karşılaştırma

| Store      | Ücretsiz | Hız    | Ölçek    | Metadata | Use Case        |
|------------|----------|--------|----------|----------|-----------------|
| Chroma     | ✅       | Orta   | Küçük    | ✅       | Prototip, Demo  |
| FAISS      | ✅       | ⚡Hızlı | Büyük    | ⚠️       | Yerel, Hızlı    |
| Pinecone   | ❌       | ⚡Hızlı | Çok Büyük| ✅       | Production      |
| Weaviate   | ✅       | Hızlı  | Büyük    | ✅       | Hybrid Search   |
| Qdrant     | ✅       | Hızlı  | Büyük    | ✅       | Production      |

---

## İleri Seviye

### 1. Metadata Filtreleme

```python
# Index oluştururken metadata ekle
documents = [
    Document(
        page_content="Python güçlü bir dildir",
        metadata={"language": "python", "level": "beginner", "year": 2024}
    ),
    Document(
        page_content="JavaScript async çalışır",
        metadata={"language": "javascript", "level": "intermediate", "year": 2024}
    ),
    Document(
        page_content="Rust memory-safe'tir",
        metadata={"language": "rust", "level": "advanced", "year": 2023}
    )
]

vectorstore = Chroma.from_documents(documents, embeddings)

# Filtreyle ara
results = vectorstore.similarity_search(
    query="güçlü dil",
    k=2,
    filter={"language": "python"}  # Sadece Python dökümanları
)

# Birden fazla filtre
results = vectorstore.similarity_search(
    query="modern diller",
    k=3,
    filter={
        "year": 2024,
        "level": {"$in": ["beginner", "intermediate"]}
    }
)
```

**Filter Operatörleri:**
```python
# Eşitlik
filter={"language": "python"}

# IN operatörü (Chroma)
filter={"language": {"$in": ["python", "javascript"]}}

# Sayısal karşılaştırma
filter={"year": {"$gte": 2023}}  # >= 2023

# AND (birden fazla field)
filter={"language": "python", "level": "beginner"}

# OR (Chroma - $or)
filter={"$or": [
    {"language": "python"},
    {"language": "rust"}
]}
```

### 2. Özel Embedding Fonksiyonu

```python
from langchain.embeddings.base import Embeddings
from typing import List

class CustomEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Toplu döküman embedding'i
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        # Tek query embedding'i
        # Kendi modelinizi buraya
        return custom_model.encode(text)

# Kullan
custom_embeddings = CustomEmbedding()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=custom_embeddings
)
```

### 3. Index Merge (Birleştirme)

```python
# İki farklı index'i birleştir
vectorstore1 = Chroma(persist_directory="./index1", ...)
vectorstore2 = Chroma(persist_directory="./index2", ...)

# Index2'deki dökümanları index1'e ekle
docs = vectorstore2.similarity_search("", k=1000)  # Tüm dökümanları al
vectorstore1.add_documents(docs)
```

### 4. Index Güncelleme (Update)

```python
# Chroma'da direkt update yok, silip tekrar eklemek gerek

# 1. Önce sil
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])

# 2. Güncellenmiş versiyonu ekle
updated_docs = [
    Document(
        page_content="Güncellenmiş içerik",
        metadata={"id": "doc_id_1", "version": 2}
    )
]
vectorstore.add_documents(updated_docs)
```

### 5. Batch Processing (Büyük Veri)

```python
import time

def index_large_dataset(documents, batch_size=100):
    """
    Büyük veri setlerini batch'ler halinde index'le
    """
    vectorstore = Chroma(
        persist_directory="./large_index",
        embedding_function=embeddings
    )
    
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i:i+batch_size]
        
        # Batch ekle
        vectorstore.add_documents(batch)
        
        # İlerleme göster
        progress = min(i + batch_size, total)
        print(f"✅ {progress}/{total} ({100*progress/total:.1f}%)")
        
        # Rate limiting (gerekirse)
        time.sleep(0.1)
    
    return vectorstore

# Kullan
docs = load_million_documents()  # 1M döküman
vectorstore = index_large_dataset(docs, batch_size=500)
```

---

## Best Practices

### 1. Chunk Size Optimizasyonu

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ❌ KÖTÜ: Çok büyük chunk'lar
splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,  # Çok büyük!
    chunk_overlap=0
)
# Sorun: Alakasız bilgiler dahil olur

# ✅ İYİ: Optimum boyut
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # LLM context'ine uygun
    chunk_overlap=50  # Bilgi kaybını önler
)
```

**Chunk Size Rehberi:**
```
Döküman Tipi          Chunk Size    Overlap
────────────────────────────────────────────
Kısa cevaplar (FAQ)   200-300      20-30
Genel metinler        500-1000     50-100
Teknik dökümanlar     1000-1500    100-150
Kodlar                300-500      50
```

### 2. Metadata Stratejisi

```python
# ✅ İYİ: Zengin metadata
Document(
    page_content="...",
    metadata={
        "source": "docs/api.md",
        "title": "API Reference",
        "section": "Authentication",
        "category": "backend",
        "tags": ["api", "auth", "security"],
        "created_at": "2024-01-15",
        "version": "2.0",
        "author": "John Doe"
    }
)

# Artık çok detaylı filtreleme yapabilirsiniz:
results = vectorstore.similarity_search(
    query="authentication",
    filter={
        "category": "backend",
        "version": "2.0",
        "tags": {"$in": ["auth", "security"]}
    }
)
```

### 3. Index Bakımı

```python
# Periyodik olarak index'i optimize edin

def maintain_index(vectorstore):
    """Index bakım rutini"""
    
    # 1. Eski dökümanları temizle
    old_doc_ids = get_old_document_ids()
    vectorstore.delete(ids=old_doc_ids)
    
    # 2. Duplicate'leri kaldır
    remove_duplicates(vectorstore)
    
    # 3. Index istatistikleri
    stats = get_index_stats(vectorstore)
    print(f"Index size: {stats['count']} documents")
    
    # 4. Persist
    vectorstore.persist()

# Haftada bir çalıştır
maintain_index(vectorstore)
```

### 4. Error Handling

```python
from langchain_community.vectorstores import Chroma

def safe_index_creation(documents, embeddings):
    """Hata yönetimli index oluşturma"""
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./safe_index"
        )
        print("✅ Index oluşturuldu")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        
        # Fallback: Batch'ler halinde dene
        print("🔄 Batch işleme deneniyor...")
        vectorstore = Chroma(
            persist_directory="./safe_index",
            embedding_function=embeddings
        )
        
        for i in range(0, len(documents), 100):
            try:
                batch = documents[i:i+100]
                vectorstore.add_documents(batch)
                print(f"✅ Batch {i//100 + 1} eklendi")
            except Exception as batch_error:
                print(f"❌ Batch {i//100 + 1} hatası: {batch_error}")
                continue
        
        return vectorstore
```

### 5. Monitoring & Logging

```python
import logging
from datetime import datetime

# Logger ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoredVectorStore:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.search_count = 0
        self.avg_results = []
    
    def search(self, query, k=4):
        """Monitörlü arama"""
        start_time = datetime.now()
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        # Metrikleri kaydet
        elapsed = (datetime.now() - start_time).total_seconds()
        self.search_count += 1
        self.avg_results.append(len(results))
        
        logger.info(f"Search #{self.search_count}: '{query}' - "
                   f"{len(results)} results in {elapsed:.3f}s")
        
        return results
    
    def get_stats(self):
        """İstatistikleri göster"""
        return {
            "total_searches": self.search_count,
            "avg_results": sum(self.avg_results) / len(self.avg_results)
        }

# Kullan
monitored = MonitoredVectorStore(vectorstore)
monitored.search("Python")
print(monitored.get_stats())
```

---

## Özet ve Karşılaştırma

### Index vs Raw Text

```
Zaman Karmaşıklığı:

Raw Text Arama:
├─ O(n) - Her döküman taranır
├─ 1M döküman = 1M işlem
└─ Çok yavaş ❌

Index Arama:
├─ O(log n) - Binary search benzeri
├─ 1M döküman = ~20 işlem
└─ Çok hızlı ✅
```

### Index Kullanım Senaryoları

```python
# ✅ Index KULLAN
scenarios = [
    "RAG (Retrieval Augmented Generation)",
    "Question Answering sistemi",
    "Semantic Search",
    "Document Retrieval",
    "Knowledge Base arama",
    "Chatbot context yönetimi"
]

# ❌ Index GEREKMİYOR
scenarios = [
    "Tek bir kısa döküman",
    "Static cevaplar (hardcoded)",
    "Basit keyword matching",
    "Tam metin aramaya gerek yok"
]
```

### Hangi Vector Store?

```python
# 🏠 Prototip/Development
use_case = "Hızlı prototip"
solution = "Chroma (ücretsiz, kolay)"

# 🚀 Production (Küçük/Orta Ölçek)
use_case = "Production app, <1M döküman"
solution = "Chroma veya FAISS (self-hosted)"

# 🌐 Production (Büyük Ölçek)
use_case = "Enterprise, >1M döküman, dağıtık"
solution = "Pinecone, Weaviate, veya Qdrant"

# ⚡ Maksimum Hız
use_case = "Latency kritik, GPU var"
solution = "FAISS (GPU mode)"

# 🔍 Hybrid Search
use_case = "Hem vector hem keyword search"
solution = "Weaviate veya Qdrant"
```

---

## Sonraki Adımlar

1. ✅ `indexes_ornekleri.py` çalıştırın
2. 📖 Kendi dökümanlarınızla index oluşturun
3. 🔍 Farklı retriever tiplerini deneyin
4. 📊 Metadata filtreleme kullanın
5. 🚀 Production'a hazırlanın

**Başarılar! 🎉**
