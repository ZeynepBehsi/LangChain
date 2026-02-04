# vector_databases.py - LangChain Community Vector Database'leri

"""
LangChain Community'de mevcut VECTOR DATABASE'LER:
"""

# ============================================
# 1️⃣ CHROMA (En Popüler - Ücretsiz)
# ============================================
from langchain_community.vectorstores import Chroma

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
# ✅ Avantajlar: Kolay kurulum, ücretsiz, yerel
# ❌ Dezavantajlar: Büyük ölçekte yavaş olabilir

# ============================================
# 2️⃣ FAISS (Facebook AI - Hızlı)
# ============================================
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
# Yükleme:
vectorstore = FAISS.load_local("faiss_index", embeddings)
# ✅ Avantajlar: Çok hızlı, büyük veri için iyi
# ❌ Dezavantajlar: Sadece arama, metadata desteği sınırlı

# ============================================
# 3️⃣ PINECONE (Ücretli SaaS)
# ============================================
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    docs,
    embeddings,
    index_name="my-index"
)
# ✅ Avantajlar: Yönetilmiş, ölçeklenebilir, hızlı
# ❌ Dezavantajlar: Ücretli, internet gerekir

# ============================================
# 4️⃣ QDRANT (Açık Kaynak)
# ============================================
from langchain_community.vectorstores import Qdrant

vectorstore = Qdrant.from_documents(
    docs,
    embeddings,
    path="./qdrant_db",  # Yerel
    collection_name="my_collection"
)
# Veya cloud:
# url="https://xyz.qdrant.io", api_key="..."
# ✅ Avantajlar: Güçlü filtreleme, ölçeklenebilir
# ❌ Dezavantajlar: Kurulum gerekebilir

# ============================================
# 5️⃣ WEAVIATE (Açık Kaynak + Cloud)
# ============================================
from langchain_community.vectorstores import Weaviate

vectorstore = Weaviate.from_documents(
    docs,
    embeddings,
    weaviate_url="http://localhost:8080"
)
# ✅ Avantajlar: GraphQL desteği, güçlü
# ❌ Dezavantajlar: Docker ile kurulum

# ============================================
# 6️⃣ MILVUS (Açık Kaynak - Enterprise)
# ============================================
from langchain_community.vectorstores import Milvus

vectorstore = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"host": "localhost", "port": "19530"}
)
# ✅ Avantajlar: Çok büyük ölçek için
# ❌ Dezavantajlar: Karmaşık kurulum

# ============================================
# 7️⃣ REDIS (Cache + Vector)
# ============================================
from langchain_community.vectorstores import Redis

vectorstore = Redis.from_documents(
    docs,
    embeddings,
    redis_url="redis://localhost:6379"
)
# ✅ Avantajlar: Hızlı, cache ile birlikte
# ❌ Dezavantajlar: Redis kurulumu gerekli

# ============================================
# 8️⃣ ELASTICSEARCH
# ============================================
from langchain_community.vectorstores import ElasticsearchStore

vectorstore = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="my_index",
    embedding=embeddings
)
# ✅ Avantajlar: Text + vector arama
# ❌ Dezavantajlar: Elasticsearch kurulumu

# ============================================
# 9️⃣ PGVECTOR (PostgreSQL)
# ============================================
from langchain_community.vectorstores import PGVector

vectorstore = PGVector.from_documents(
    docs,
    embeddings,
    connection_string="postgresql://user:pass@localhost/db"
)
# ✅ Avantajlar: PostgreSQL kullanıyorsanız ideal
# ❌ Dezavantajlar: PG extension gerekli

# ============================================
# 🔟 MONGODB ATLAS
# ============================================
from langchain_mongodb import MongoDBAtlasVectorSearch

vectorstore = MongoDBAtlasVectorSearch.from_documents(
    docs,
    embeddings,
    connection_string="mongodb+srv://...",
    database_name="mydb",
    collection_name="mycoll"
)
# ✅ Avantajlar: MongoDB kullanıyorsanız
# ❌ Dezavantajlar: Atlas gerekli

# ============================================
# 1️⃣1️⃣ SUPABASE (PostgreSQL + Cloud)
# ============================================
from langchain_community.vectorstores import SupabaseVectorStore

vectorstore = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase_client,
    table_name="documents"
)
# ✅ Avantajlar: Kolay cloud setup
# ❌ Dezavantajlar: Supabase hesabı gerekli

# ============================================
# 1️⃣2️⃣ DOCARRAY (InMemory)
# ============================================
from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)
# ✅ Avantajlar: Çok hızlı, kurulum yok
# ❌ Dezavantajlar: Sadece bellekte, kalıcı değil

# ============================================
# 1️⃣3️⃣ LANCE DB (Açık Kaynak)
# ============================================
from langchain_community.vectorstores import LanceDB

vectorstore = LanceDB.from_documents(
    docs,
    embeddings,
    uri="./lancedb"
)
# ✅ Avantajlar: Hızlı, yerel, multi-modal
# ❌ Dezavantajlar: Yeni, küçük topluluk


"""
📊 HANGİSİNİ SEÇMELİSİNİZ?

🎓 ÖĞRENME / PROTOTIP:
   ✅ Chroma - En kolay başlangıç
   ✅ DocArrayInMemorySearch - Test için
   
⚡ PERFORMANS (Yerel):
   ✅ FAISS - Çok hızlı arama
   ✅ LanceDB - Modern, hızlı
   
☁️ CLOUD / PRODUCTION:
   ✅ Pinecone - Yönetilmiş, kolay
   ✅ Qdrant Cloud - Güçlü filtreleme
   ✅ Weaviate Cloud - GraphQL desteği
   
🏢 ENTERPRISE:
   ✅ Milvus - Çok büyük ölçek
   ✅ Elasticsearch - Text + Vector
   
💾 MEVCUT DATABASE:
   ✅ PGVector - PostgreSQL varsa
   ✅ MongoDB Atlas - MongoDB varsa
   ✅ Redis - Redis varsa
   

📈 KARŞILAŞTIRMA:

Database        | Hız    | Ölçek  | Kolay | Ücretsiz | Kurulum
----------------|--------|--------|-------|----------|----------
Chroma          | ⭐⭐⭐  | ⭐⭐   | ⭐⭐⭐⭐⭐ | ✅      | Yok
FAISS           | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐  | ✅      | Yok
Pinecone        | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌      | Yok
Qdrant          | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐   | ✅      | Docker
Milvus          | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐    | ✅      | Docker
DocArrayInMemory| ⭐⭐⭐⭐⭐ | ⭐     | ⭐⭐⭐⭐⭐ | ✅      | Yok
LanceDB         | ⭐⭐⭐⭐  | ⭐⭐⭐⭐ | ⭐⭐⭐⭐  | ✅      | Yok
"""
