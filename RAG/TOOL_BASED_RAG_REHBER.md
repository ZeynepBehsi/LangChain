# TOOL-BASED RAG - ÖĞRENME REHBERİ

## 🎯 BU DOSYADA NE ÖĞRENECEKSINIZ?

1. **Tool Nedir?** - Kavramsal açıklama
2. **RAG'i Tool'a Çevirme** - Nasıl yapılır?
3. **Agent Nedir?** - Tool kullanan akıllı sistem
4. **Multi-Tool Sistemler** - Birden fazla tool kullanma
5. **Pratik Örnekler** - Gerçek kullanım senaryoları

---

## 📚 BASIC RAG vs TOOL-BASED RAG

### Basic RAG (Öğrendiğiniz)
```
Soru → Retriever → Chunk Bul → LLM'e Gönder → Cevap
```

**Özellikler:**
- ✅ Basit, anlaşılır
- ✅ Her zaman RAG kullanır
- ❌ Esneklik yok
- ❌ Tek kaynak

---

### Tool-Based RAG (Şimdi öğreneceğiniz)
```
Soru → Agent → Karar Ver → Tool Seç → Kullan → Cevap
                 ↓
        [RAG Tool | Hesap Tool | Zaman Tool | ...]
```

**Özellikler:**
- ✅ Dinamik karar verme
- ✅ Çoklu tool desteği
- ✅ Esnek
- ✅ Karmaşık sorguları çözebilir

---

## 🔧 TOOL NEDİR?

**Tool** = Agent'ın kullanabileceği bir fonksiyon/yetenek

### Örnekler:
```python
# Tool 1: RAG (Döküman arama)
"What is task decomposition?" → RAG Tool → Dökümanlardan bul

# Tool 2: Calculator (Hesaplama)
"What is 125 * 47?" → Calculator Tool → Hesapla

# Tool 3: Current Time (Tarih/saat)
"What time is it?" → Time Tool → Sistem saatini al

# Tool 4: Web Search (İnternet arama)
"Latest news about AI?" → Web Search Tool → Google'da ara
```

---

## 🤖 AGENT NEDİR?

**Agent** = Tool'ları kullanabilen akıllı sistem

### Agent'ın Görevi:
1. Soruyu anla
2. Hangi tool gerekli? (Karar ver)
3. Tool'u kullan
4. Sonucu yorumla
5. Cevap ver

### Örnek Akış:
```
Soru: "What is task decomposition and what is 10+20?"

Agent Düşüncesi:
1. İki farklı soru var
2. "task decomposition" → RAG tool gerek
3. "10+20" → Calculator tool gerek
4. Her ikisini de kullanmalıyım

Aksiyonlar:
- RAG Tool → "Task decomposition is..."
- Calculator Tool → "30"

Cevap: "Task decomposition is the process of breaking down... 
        and 10+20 equals 30."
```

---

## 📋 ADIM ADIM AÇIKLAMA

Dosyadaki her adımı açıklayalım:

---

### ADIM 1: TEMEL SETUP

```python
model = ChatAnthropic(...)
embeddings = HuggingFaceEmbeddings(...)
vectorstore = Chroma(persist_directory="./chroma_db", ...)
```

**Ne yapıyor?**
- AI modelini yükle (Claude)
- Embedding modelini yükle
- Mevcut vector database'e bağlan

**Neden?**
- Tool'lar için temel altyapı

---

### ADIM 2: RETRIEVER OLUŞTURMA

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

**Ne yapıyor?**
- VectorStore → Retriever'a çevir
- Arama tipi: Benzerlik
- Her aramada 3 sonuç getir

**Neden?**
- Retriever, tool'a çevrilebilir bir ara katman

**Analoji:**
```
VectorStore = Kütüphane
Retriever = Kütüphaneci (arama yapan kişi)
Tool = Kütüphaneciye soru sorma yöntemi
```

---

### ADIM 3: RETRIEVER'I TOOL'A ÇEVİRME ⭐ (EN ÖNEMLİ)

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    name="agent_documentation_search",
    description="Search for information about AI agents..."
)
```

**Ne yapıyor?**
- Retriever'ı bir tool'a çeviriyor
- Tool'a isim veriyor
- Tool'un ne yaptığını açıklıyor

**Neden `description` önemli?**
```
Agent şunu sorar: "Hangi tool'u kullanmalıyım?"

Descriptions:
- "agent_documentation_search" → "AI agents hakkında bilgi ara"
- "calculator" → "Matematik hesapla"
- "get_current_time" → "Şu anki saati al"

Agent karar verir: "Ah, bu soru AI agents hakkında, 
                    agent_documentation_search kullanmalıyım!"
```

**Kritik:** Description ne kadar iyi → Agent o kadar doğru karar verir!

---

### ADIM 4: EK TOOL'LAR EKLEME

```python
# Tool 1: Tarih/Saat
time_tool = Tool(
    name="get_current_time",
    func=get_current_time,
    description="Get the current date and time..."
)

# Tool 2: Hesap Makinesi
calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description="Calculate mathematical expressions..."
)

tools = [retriever_tool, time_tool, calculator_tool]
```

**Ne yapıyor?**
- Basit Python fonksiyonlarını tool'a çeviriyor
- Her tool'a isim ve açıklama veriyor
- Listeye ekliyor

**Neden birden fazla tool?**
```
Soru: "What is self-reflection in agents and what time is it?"

Agent:
1. "self-reflection" → retriever_tool kullan
2. "what time is it" → time_tool kullan
3. Her iki cevabı birleştir
```

---

### ADIM 5: AGENT OLUŞTURMA

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant with access to tools..."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(model, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3
)
```

**Ne yapıyor?**
1. System prompt oluştur (Agent'a rolünü söyle)
2. Agent oluştur (Model + Tools + Prompt)
3. AgentExecutor oluştur (Agent'ı çalıştıran wrapper)

**AgentExecutor parametreleri:**
- `verbose=True` → Her adımı göster (debug için)
- `max_iterations=3` → Max 3 tool çağrısı (sonsuz döngü önleme)
- `handle_parsing_errors=True` → Hata toleransı

---

### ADIM 6: AGENT'I TEST ETME

```python
question = "What is task decomposition?"
response = agent_executor.invoke({"input": question})
print(response["output"])
```

**Ne yapıyor?**
1. Soruyu agent'a gönder
2. Agent düşünür, tool seçer, kullanır
3. Cevabı döndürür

---

## 🎬 ÖRNEK AKIŞ (Detaylı)

Soruyu adım adım takip edelim:

### Soru: "What are the types of memory in agents?"

```
1. USER → AGENT
   Input: "What are the types of memory in agents?"

2. AGENT DÜŞÜNÜR
   "Bu soru AI agents hakkında...
    agent_documentation_search tool'unu kullanmalıyım"

3. AGENT → RETRIEVER TOOL
   Tool Input: "types of memory in agents"

4. RETRIEVER TOOL → VECTOR DB
   Arama yap, en yakın 3 chunk bul

5. VECTOR DB → RETRIEVER TOOL
   Chunk 1: "Short-term memory: I would consider..."
   Chunk 2: "Long-term memory: This provides the agent..."
   Chunk 3: "Sensory memory as learning embedding..."

6. RETRIEVER TOOL → AGENT
   Tool Output: [3 chunk döndürür]

7. AGENT → LLM
   Prompt: "Bu bilgilere göre soruyu yanıtla:
            [3 chunk]
            Soru: What are the types of memory in agents?"

8. LLM → AGENT
   "There are three main types of memory:
    1. Sensory memory...
    2. Short-term memory...
    3. Long-term memory..."

9. AGENT → USER
   Final Output: [Yukarıdaki yanıt]
```

---

## 🔥 KARMAŞIK ÖRNEK

### Soru: "What is task decomposition and what is 125 * 8?"

```
AGENT DÜŞÜNÜR:
"İki farklı soru var!
 1. task decomposition → retriever_tool
 2. 125 * 8 → calculator_tool
 İkisini de kullanmalıyım"

ADIM 1: Retriever Tool
Input: "task decomposition"
Output: "Task decomposition is the process..."

ADIM 2: Calculator Tool
Input: "125 * 8"
Output: "Result: 1000"

FINAL ANSWER:
"Task decomposition is the process of breaking down complex tasks
 into smaller steps. And 125 * 8 equals 1000."
```

---

## 💡 AVANTAJLAR

### 1️⃣ Dinamik Karar Verme
```python
# Agent karar verir
"What is task decomposition?" → RAG tool kullan
"What time is it?" → Time tool kullan
"Hello!" → Hiçbir tool gerekmiyor, direkt cevapla
```

### 2️⃣ Çoklu Kaynak
```python
tools = [
    rag_tool,           # Dökümanlardan
    web_search_tool,    # İnternetten
    database_tool,      # Veritabanından
    api_tool           # API'den
]
```

### 3️⃣ Genişletilebilir
```python
# Yeni tool eklemek kolay
new_tool = create_tool(...)
tools.append(new_tool)
```

---

## 🎯 KULLANIM SENARYOLARI

### Senaryo 1: Teknik Dokümantasyon + Kod Çalıştırma
```python
tools = [
    documentation_search_tool,  # Döküman ara
    code_executor_tool,         # Kod çalıştır
    syntax_checker_tool         # Syntax kontrol et
]

Soru: "How do I use pandas DataFrame and show me an example?"
→ Döküman ara + Kod çalıştır
```

### Senaryo 2: E-ticaret Asistanı
```python
tools = [
    product_search_tool,    # Ürün ara
    price_calculator_tool,  # Fiyat hesapla
    inventory_checker_tool, # Stok kontrol
    order_tracker_tool      # Sipariş takip
]
```

### Senaryo 3: Araştırma Asistanı
```python
tools = [
    internal_docs_tool,     # Şirket dökümanları
    web_search_tool,        # İnternet araması
    database_query_tool,    # Veritabanı sorgusu
    calculator_tool         # Hesaplama
]
```

---

## ⚠️ DİKKAT EDİLMESİ GEREKENLER

### 1️⃣ Tool Description Kalitesi
❌ Kötü: `description="Search tool"`
✅ İyi: `description="Search for AI agent information including task decomposition, memory types, and planning"`

### 2️⃣ Max Iterations
```python
max_iterations=3  # Çok düşük → Karmaşık soruları çözemez
max_iterations=20 # Çok yüksek → Yavaş + sonsuz döngü riski
```

### 3️⃣ Tool Sayısı
- Az tool → Sınırlı yetenek
- Çok tool → Agent karışabilir
- **Optimal:** 3-7 tool

### 4️⃣ Maliyet
Her tool çağrısı = Ekstra LLM çağrısı = Ekstra maliyet

---

## 🚀 SONRAKI ADIMLAR

Bu dosyayı öğrendikten sonra:
1. ✅ Çalıştırın ve çıktıları inceleyin
2. ✅ Kendi tool'unuzu ekleyin
3. ✅ Farklı sorular test edin
4. ⬜ Multi-document RAG öğrenin
5. ⬜ Streaming responses ekleyin

---

## 📚 ÖZET

✅ **Tool** = Agent'ın kullanabileceği fonksiyon
✅ **Agent** = Tool'ları akıllıca kullanan sistem
✅ **RAG Tool** = Retriever'ı tool'a çevirme
✅ **Multi-Tool** = Birden fazla yetenek
✅ **Avantaj** = Esneklik, dinamik karar verme

**Şimdi kodu çalıştırın ve öğrenmeye devam edin!** 🎉
