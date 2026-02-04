# embedding_models.py - LangChain Community Embedding Modelleri

"""
LangChain Community'de mevcut EMBEDDING MODELLERİ:
"""

# ============================================
# 1️⃣ HUGGING FACE EMBEDDINGS (Ücretsiz)
# ============================================
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
    # Diğer modeller:
    # - "sentence-transformers/all-MiniLM-L6-v2" (hızlı, küçük)
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (çok dilli)
    # - "BAAI/bge-small-en-v1.5" (performans/boyut dengesi)
    # - "BAAI/bge-large-en-v1.5" (yüksek performans)
)

# ============================================
# 2️⃣ OPENAI EMBEDDINGS (Ücretli)
# ============================================
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # veya "text-embedding-3-large"
)

# ============================================
# 3️⃣ COHERE EMBEDDINGS (Ücretli)
# ============================================
from langchain_community.embeddings import CohereEmbeddings

embeddings = CohereEmbeddings(
    model="embed-english-v3.0"  # veya "embed-multilingual-v3.0"
)

# ============================================
# 4️⃣ OLLAMA EMBEDDINGS (Yerel/Ücretsiz)
# ============================================
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="llama2"  # veya diğer Ollama modelleri
)

# ============================================
# 5️⃣ BEDROCK EMBEDDINGS (AWS)
# ============================================
from langchain_community.embeddings import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1"
)

# ============================================
# 6️⃣ GOOGLE PALM EMBEDDINGS (Google)
# ============================================
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# ============================================
# 7️⃣ FAKE EMBEDDINGS (Test için)
# ============================================
from langchain_community.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(size=768)

# ============================================
# 8️⃣ HUGGING FACE HUB (API üzerinden)
# ============================================
from langchain_community.embeddings import HuggingFaceHubEmbeddings

embeddings = HuggingFaceHubEmbeddings(
    repo_id="sentence-transformers/all-mpnet-base-v2"
)

# ============================================
# 9️⃣ SENTENCE TRANSFORMER (Direkt)
# ============================================
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ============================================
# 🔟 VOYAGEAI EMBEDDINGS (Ücretli)
# ============================================
# from langchain_community.embeddings import VoyageEmbeddings
# embeddings = VoyageEmbeddings(model="voyage-01")


"""
📊 HANGİSİNİ SEÇMELİSİNİZ?

✅ BAŞLANGIÇ İÇİN:
   - HuggingFaceEmbeddings (ücretsiz, yerel)
   
🚀 PERFORMANS İÇİN:
   - OpenAI text-embedding-3-large
   - Cohere embed-english-v3.0
   
💰 BUDGET-FRIENDLY:
   - HuggingFace modelleri (ücretsiz)
   - Ollama (yerel, ücretsiz)
   
🌍 ÇOK DİLLİ:
   - paraphrase-multilingual-MiniLM-L12-v2
   - Cohere embed-multilingual-v3.0
   
⚡ HIZLI:
   - all-MiniLM-L6-v2 (küçük, hızlı)
   
🎯 EN İYİ DOĞRULUK:
   - BAAI/bge-large-en-v1.5
   - OpenAI text-embedding-3-large
"""
