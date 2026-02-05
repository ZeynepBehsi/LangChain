"""
🎯 ÇOK TOOL'LU RAG SİSTEMİ
==========================

Bu örnekte agent 3 farklı tool arasından seçim yapıyor:
1. RAG Retriever - AI agents hakkında bilgi
2. Calculator - Matematiksel hesaplamalar
3. Current Time - Güncel saat bilgisi

Agent otomatik olarak hangi tool'u kullanacağına karar verir!
"""

import os
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import create_react_agent


# 1️⃣ MODEL VE RETRIEVER HAZIRLIĞI
print("🎯 ÇOK TOOL'LU RAG SİSTEMİ")
print("="*60)
print("\n1️⃣ Model ve veritabanı hazırlanıyor...")

# LLM modeli
model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

# Embedding ve Chroma DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("✅ Hazır!")


# 2️⃣ TOOL'LAR OLUŞTURMA

# Tool 1: RAG Retriever
print("\n2️⃣ Tool'lar oluşturuluyor...")

retriever_tool = create_retriever_tool(
    retriever,
    name="search_ai_agents_docs",
    description="AI agents, LLM applications, task decomposition, memory systems hakkında sorular için kullan. Bu dokümanlarda Lilian Weng'in AI agents üzerine yazıları var."
)

# Tool 2: Hesap Makinesi
@tool
def calculator(expression: str) -> str:
    """
    Matematiksel hesaplamalar yapmak için kullanın.
    Örnek: '2 + 2' veya '100 * 50 / 2'
    """
    try:
        # Güvenli eval için sadece sayılar ve operatörler
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "❌ Sadece sayılar ve +, -, *, /, () kullanabilirsiniz!"
        
        result = eval(expression)
        return f"📊 Sonuç: {result}"
    except Exception as e:
        return f"❌ Hata: {str(e)}"


# Tool 3: Güncel Saat
@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Güncel tarih ve saat bilgisini almak için kullanın.
    timezone parametresi: 'UTC', 'local' gibi
    """
    now = datetime.now()
    return f"🕐 Şu an: {now.strftime('%Y-%m-%d %H:%M:%S')} ({timezone})"


# Tool'ları listeye ekle
tools = [retriever_tool, calculator, get_current_time]

print("✅ 3 tool oluşturuldu:")
print(f"   - {retriever_tool.name}")
print(f"   - {calculator.name}")
print(f"   - {get_current_time.name}")


# 3️⃣ AGENT OLUŞTURMA
print("\n3️⃣ Agent oluşturuluyor...")
agent = create_react_agent(model, tools)
print("✅ Agent hazır! Agent şimdi 3 tool arasından seçim yapabilir.")


# 4️⃣ TESTLER
print("\n" + "="*60)
print("🧪 TESTLER - Agent hangi tool'u kullanacak?")
print("="*60)

test_questions = [
    # RAG Tool kullanmalı
    "What is task decomposition in AI agents?",
    
    # Calculator tool kullanmalı
    "What is 1234 multiplied by 56?",
    
    # Time tool kullanmalı
    "What time is it now?",
    
    # 🔥 İLGİNÇ: Hem RAG hem Calculator kullanmalı!
    "What is task decomposition and what is 100 + 50?"
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{'─'*60}")
    print(f"SORU {i}: {question}")
    print('─'*60)
    
    response = agent.invoke({"messages": [("user", question)]})
    
    # Son mesajı al (agent'ın cevabı)
    final_message = response["messages"][-1].content
    
    print(f"\n💬 CEVAP:\n{final_message}")


print("\n" + "="*60)
print("✅ TAMAMLANDI!")
print("="*60)

print("\n📚 ÖĞRENİLENLER:")
print("-" * 60)
print("""
1. Agent OTOMATIK olarak doğru tool'u seçiyor
2. Bazı sorular için BIRDEN FAZLA tool kullanılabilir
3. Her tool'un açık bir DESCRIPTION'ı olmalı
4. Agent, description'lara bakarak karar veriyor
5. Tool-based RAG, basic RAG'den çok daha ESNEK!
""")
