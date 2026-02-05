# simple_tool_rag.py - Basit Tool-Based RAG

"""
BASIT TOOL-BASED RAG
Öğrenmek için daha basit bir yaklaşım
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("🎯 BASIT TOOL-BASED RAG")
print("="*60)

from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

# ====== ADIM 1: SETUP ======
print("\n1️⃣ Model ve Retriever hazırlanıyor...")

model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Hazır!")

# ====== ADIM 2: RETRIEVER'I TOOL'A ÇEVİRME ======
print("\n2️⃣ Retriever tool'a dönüştürülüyor...")

retriever_tool = create_retriever_tool(
    retriever,
    name="search_agent_docs",
    description="Search for information about AI agents, task decomposition, memory, planning, and self-reflection. Use this tool when asked about AI agents."
)

print(f"✅ Tool oluşturuldu: {retriever_tool.name}")

# ====== ADIM 3: AGENT OLUŞTURMA ======
print("\n3️⃣ Agent oluşturuluyor...")

tools = [retriever_tool]

agent = create_react_agent(model, tools)

print("✅ Agent hazır!")

# ====== ADIM 4: TEST ======
print("\n" + "="*60)
print("🧪 TESTLER")
print("="*60)

questions = [
    "What is task decomposition?",
    "What are the types of memory in AI agents?",
]

for i, question in enumerate(questions, 1):
    print(f"\n{'─'*60}")
    print(f"SORU {i}: {question}")
    print('─'*60)
    
    # Agent'ı çalıştır
    result = agent.invoke({
        "messages": [("human", question)]
    })
    
    # Son mesajı göster (cevap)
    print(f"\n💬 CEVAP:")
    print(result["messages"][-1].content)

print("\n" + "="*60)
print("✅ TAMAMLANDI!")
print("="*60)
