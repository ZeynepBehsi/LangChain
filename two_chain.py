"""
Temperature farkını göstermek için iki farklı Claude modeli ile iki chain oluşturma.
"""

# import libraries
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 2. Environment variables yükle
load_dotenv()

# 3. İki farklı model oluştur - temperature farkı ile

# Yaratıcı Claude
llm_creative = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=1.0  # Maksimum yaratıcılık
)

# Deterministik Claude
llm_strict = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0  # Sıfır randomness
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


# 6. İki chain oluştur
chain_creative = prompt | llm_creative | output_parser
chain_strict = prompt | llm_strict | output_parser

# Karşılaştır
question = "Tell me a story about AI and graphs."

print("🎨 CREATIVE (temp=1.0):")
print(chain_creative.invoke({"input": question}))

print("\n📏 STRICT (temp=0):")
print(chain_strict.invoke({"input": question}))

