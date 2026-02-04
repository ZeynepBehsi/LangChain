# CHROMA DB ERİŞİM REHBERİ

## 🎯 3 FARKLI ERİŞİM YÖNTEMİ

Chroma DB'nize 3 farklı şekilde erişebilirsiniz:

---

## 1️⃣ PYTHON KODU İLE (En Basit)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Embedding model yükle
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Chroma DB'ye bağlan
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Arama yap
results = vectorstore.similarity_search("What is task decomposition?", k=3)

# Sonuçları yazdır
for doc in results:
    print(doc.page_content)
```

**Dosya:** `RAG/chroma_explorer.py`
**Çalıştırma:** `python RAG/chroma_explorer.py`

---

## 2️⃣ İNTERAKTİF TERMINAL (Komut Satırı)

Terminal'de interaktif arayüz!

**Dosya:** `RAG/chroma_interactive.py`

**Çalıştırma:**
```bash
python RAG/chroma_interactive.py
```

**Komutlar:**
- Soru yazın → Arama yapar
- `list` → İlk 5 dökümanı gösterir
- `stats` → İstatistikler
- `search:5 [soru]` → 5 sonuçla arama
- `exit` → Çıkış

**Örnek:**
```
💬 > What is task decomposition?
💬 > stats
💬 > list
💬 > search:5 What is memory?
💬 > exit
```

---

## 3️⃣ WEB ARAYÜZÜ (En Görsel)

Tarayıcıda güzel bir arayüz!

**Dosya:** `RAG/chroma_web_viewer.py`

**Çalıştırma:**
```bash
python RAG/chroma_web_viewer.py
```

**Sonra:**
1. Tarayıcınızda açın: http://localhost:5000
2. Arama kutusuna soru yazın
3. Sonuçları görün!

**Özellikler:**
✅ Güzel görsel arayüz
✅ Canlı arama
✅ Benzerlik skorları
✅ Database istatistikleri

---

## 📊 DATABASE YAPISI

```
chroma_db/
├── chroma.sqlite3          # Ana veritabanı
└── [UUID-klasörleri]/      # Vektör verileri
```

**Önemli:** Bu dosyaları direkt açamazsınız! Python ile erişmelisiniz.

---

## 🛠️ TEMEL İŞLEMLER

### ✅ Arama Yapmak
```python
results = vectorstore.similarity_search("soru", k=3)
```

### ✅ Skorlu Arama
```python
results = vectorstore.similarity_search_with_score("soru", k=3)
for doc, score in results:
    print(f"Skor: {score}, İçerik: {doc.page_content}")
```

### ✅ Toplam Döküman Sayısı
```python
total = vectorstore._collection.count()
print(f"Toplam: {total}")
```

### ✅ Tüm Dökümanları Listeleme
```python
all_data = vectorstore._collection.get(limit=10)
for doc in all_data['documents']:
    print(doc)
```

### ✅ Retriever Olarak Kullanma (RAG için)
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

docs = retriever.invoke("soru")
```

---

## 🚀 HIZLI BAŞLANGIÇ

### 1. Explorer ile keşfet:
```bash
python RAG/chroma_explorer.py
```

### 2. İnteraktif mod ile dene:
```bash
python RAG/chroma_interactive.py
```

### 3. Web arayüzünü başlat:
```bash
python RAG/chroma_web_viewer.py
# Tarayıcıda: http://localhost:5000
```

---

## ❓ SSS (Sık Sorulan Sorular)

**S: Grafiksel arayüz var mı?**
C: Varsayılan olarak yok, ama web arayüzü hazırladım! (`chroma_web_viewer.py`)

**S: Verileri nasıl görebilirim?**
C: Python kodu ile! Explorer veya interactive script'leri kullanın.

**S: Database'i silebilir miyim?**
C: Evet! `rm -rf ./chroma_db` (dikkatli olun!)

**S: Yeni veri ekleyebilir miyim?**
C: Evet! `basic_rag.py` tekrar çalıştırın veya yeni dökümanlar ekleyin.

**S: Embedding model değiştirebilir miyim?**
C: Hayır! Aynı modeli kullanmalısınız, yoksa vektörler uyumsuz olur.

---

## 🎓 ÖĞRENDİKLERİNİZ

✅ Chroma DB bir vector database
✅ Python ile erişilir
✅ 3 farklı arayüz: kod, terminal, web
✅ Anlamsal arama yapabilir
✅ RAG sistemlerinde kullanılır

---

## 📚 KAYNAKLAR

- **Explorer:** `RAG/chroma_explorer.py`
- **Interactive:** `RAG/chroma_interactive.py`
- **Web:** `RAG/chroma_web_viewer.py`
- **Temel RAG:** `RAG/basic_rag.py`

Hangisini denemek istersiniz? 🚀
