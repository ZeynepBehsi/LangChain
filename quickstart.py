"""
LangChain Quickstart - İlk Chain Örneği (Claude ile)
Tarih: 1 Şubat 2026
Zeynep - LangChain Öğrenme Projesi
"""

# 1. Import'lar
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic  # ← CLAUDE İÇİN
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 2. Environment variables yükle
load_dotenv()

# 3. Model oluştur (CLAUDE)
print("🤖 Claude model oluşturuluyor...")
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",  # ← Claude 3 Haiku (en ucuz ve hızlı model)
    temperature=0  # Deterministik cevaplar için
)

# 4. Prompt template oluştur
print("📝 Prompt template hazırlanıyor...")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specialized in explaining technical concepts clearly and concisely."),
    ("user", "{input}")
])

# 5. Output parser
# ne yapar: AI'ın cevabını string formatına çevirir, cevaptaki sadece metin kısmını alır. 
output_parser = StrOutputParser()

# 6. Chain oluştur (LCEL - LangChain Expression Language)
print("🔗 Chain oluşturuluyor...")
chain = prompt | llm | output_parser

# 7. Chain'i çalıştır
print("\n" + "="*50)
print("🚀 CHAIN ÇALIŞTIRILIYOR")
print("="*50 + "\n")

question = "What is LangChain in 2 sentences?"
print(f"❓ Soru: {question}\n")

response = chain.invoke({"input": question})

print(f"💬 Claude'un Cevabı:\n{response}")
print("\n" + "="*50)
print("✅ BAŞARILI!")
print("="*50)


#---------------------------------------------------------
# Phase 2: Farklı bir soru ile deneme

print("\n" + "="*50)
print("🧪 TEST: Farklı Sorular")
print("="*50 + "\n")

questions = [
    "What is LangChain in 2 sentences?",
    "Explain graph neural networks simply.",
    "What is the difference between LangChain and LangGraph?",
    "How does fraud detection work with graph ML?"  # ← Senin alanın!
]

for i, q in enumerate(questions, 1):
    print(f"\n[Test {i}] ❓ {q}")
    response = chain.invoke({"input": q})
    print(f"💬 {response}")
    print("-" * 50)


"""
NOT: Öğrenme noktası: Chain bir kez oluşturulur, defalarca kullanılır! 🔄
"""

#---------------------------------------------------------
# Phase 3: Türkçe cevaplar
prompt_tr = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımcı bir yapay zeka asistanısın. Türkçe açıklamalar yaparsın."),
    ("user", "{input}")
])

chain_tr = prompt_tr | llm | output_parser

# Türkçe sor
soru = "LangChain nedir? Kısaca açıkla."
cevap = chain_tr.invoke({"input": soru})
print(f"💬 {cevap}")