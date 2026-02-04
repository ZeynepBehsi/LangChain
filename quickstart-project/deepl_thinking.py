# Claude'un nasıl düşündüğünü görmek için:

"""
Deep thinking için sadece promting aşamasında farklılık var. Yapı aynı.
"""

# import libraries
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 2. Environment variables yükle
load_dotenv()

# 3. Model oluştur (CLAUDE)
llm_thinking = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
)

# 4. Prompt template oluştur
print("📝 Prompt template hazırlanıyor...")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specialized in explaining technical concepts clearly and concisely."),
    ("user", "{input}")
])


# 5. Output parser
output_parser = StrOutputParser()

# 6. Chain oluştur
chain = prompt | llm_thinking | output_parser

# Karmaşık soru sor
complex_question = """
I have a graph with 1000 nodes representing bank transactions.
Some are fraudulent. How would you approach detecting fraud using graph ML?
Think step by step.
"""

print("🔍 Claude'a soru soruluyor...\n")
response = chain.invoke({"input": complex_question})
print("💡 Claude'un Cevabı:\n")
print(response)


