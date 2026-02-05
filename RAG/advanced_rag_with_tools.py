# advanced_rag_with_tools.py - Tool Kullanarak RAG Sistemi

"""
TOOL-BASED RAG SİSTEMİ

Fark: Basic RAG'den farklı olarak, burada:
1. RAG bir "tool" olarak tanımlanır
2. Agent bu tool'u kullanabilir
3. Birden fazla tool ile kombine edilebilir
4. Dinamik karar verme (ne zaman RAG kullanılacak?)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# ADIM 1: TEMEL SETUP
# ============================================
print("🚀 TOOL-BASED RAG SİSTEMİ")
print("="*60)

from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# AI Model
model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Mevcut Vector Store'a bağlan
print("\n📚 Mevcut Chroma DB'ye bağlanıyor...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
print(f"✅ {vectorstore._collection.count()} döküman yüklendi")

# ============================================
# ADIM 2: RETRIEVER OLUŞTURMA
# ============================================
print("\n🔍 Retriever oluşturuluyor...")

# Retriever = Arama yapan nesne
retriever = vectorstore.as_retriever(
    search_type="similarity",  # Benzerlik araması
    search_kwargs={"k": 3}     # En yakın 3 sonuç
)

print("✅ Retriever hazır")

# ============================================
# ADIM 3: RETRIEVER'I TOOL'A ÇEVİRME
# ============================================
print("\n🛠️ Retriever bir tool'a dönüştürülüyor...")

from langchain_core.tools import create_retriever_tool

# Tool oluşturma
retriever_tool = create_retriever_tool(
    retriever,
    name="agent_documentation_search",
    description="""
    Search for information about AI agents, LLM agents, and autonomous agents.
    Use this tool when you need to answer questions about:
    - Task decomposition
    - Agent planning
    - Memory types (short-term, long-term)
    - Self-reflection in agents
    - Tool use in agents
    - Agent frameworks (ReAct, Reflexion, etc.)
    
    Input should be a search query about agents.
    """
)

print("✅ RAG artık bir tool!")
print(f"   Tool adı: {retriever_tool.name}")
print(f"   Açıklama: {retriever_tool.description[:100]}...")

# ============================================
# ADIM 4: EK TOOL'LAR EKLEME (Opsiyonel)
# ============================================
print("\n🔧 Ek tool'lar ekleniyor...")

from langchain_core.tools import tool
from datetime import datetime

# Tool 1: Mevcut tarih/saat
@tool
def get_current_time() -> str:
    """Get the current date and time. Use this when the user asks about current time or date."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Tool 2: Basit hesap makinesi
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions. Input should be a valid Python math expression like '2+2' or '10*5'."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Tüm tool'ları listele
tools = [retriever_tool, get_current_time, calculator]

print(f"✅ {len(tools)} tool hazır:")
for i, tool in enumerate(tools, 1):
    print(f"   {i}. {tool.name}")

# ============================================
# ADIM 5: AGENT OLUŞTURMA
# ============================================
print("\n🤖 Agent oluşturuluyor...")

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Agent için prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to tools.
    
    You have access to:
    1. A documentation search tool about AI agents
    2. A current time tool
    3. A calculator tool
    
    Use the tools when needed to answer questions accurately.
    If you don't need any tools, just answer directly.
    """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent oluştur
agent = create_tool_calling_agent(model, tools, prompt)

# AgentExecutor - Agent'ı çalıştıran wrapper
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Detaylı log
    max_iterations=3,  # Max kaç adım
    handle_parsing_errors=True
)

print("✅ Agent hazır!")

# ============================================
# ADIM 6: AGENT'I TEST ETME
# ============================================
print("\n" + "="*60)
print("🧪 AGENT TESTLERİ")
print("="*60)

test_questions = [
    "What is task decomposition in AI agents?",
    "What time is it now?",
    "Calculate 15 * 7",
    "What are the types of memory in agents and what is 100 + 50?"
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{'─'*60}")
    print(f"TEST {i}: {question}")
    print('─'*60)
    
    try:
        response = agent_executor.invoke({"input": question})
        print(f"\n💬 CEVAP:")
        print(response["output"])
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    print()

print("\n" + "="*60)
print("✅ TÜM TESTLER TAMAMLANDI!")
print("="*60)
