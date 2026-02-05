# 🎯 ÇOK TOOL'LU RAG SİSTEMİ REHBERİ

## İçindekiler
1. [Giriş](#giriş)
2. [Basic RAG vs Tool-based RAG vs Multi-Tool RAG](#karşılaştırma)
3. [Nasıl Çalışır?](#nasıl-çalışır)
4. [Tool Tanımlama](#tool-tanımlama)
5. [Agent Karar Mekanizması](#agent-karar-mekanizması)
6. [Gerçek Dünya Örnekleri](#gerçek-dünya-örnekleri)

---

## Giriş

**Multi-Tool RAG**, agent'ın birden fazla araç (tool) arasından seçim yaparak en uygun bilgi kaynağından cevap üretmesini sağlar.

### Neden Multi-Tool RAG?

```python
# ❌ Basic RAG - Sadece tek kaynak
question = "What is 2 + 2?"
# Her zaman vector DB'yi arar, ama bu matematik sorusu!

# ✅ Multi-Tool RAG - Doğru tool seçimi
question = "What is 2 + 2?"
# Agent düşünür: "Bu matematik, calculator tool kullanmalıyım!"
```

---

## Karşılaştırma

### 1. Basic RAG
```
Kullanıcı Sorusu
      ↓
   Vector DB
      ↓
    LLM
      ↓
    Cevap
```

**Özellikler:**
- ✅ Basit ve hızlı
- ❌ Sadece tek kaynak (vector DB)
- ❌ Her soru için DB'ye gider
- ❌ Matematik, zaman, API çağrıları yapamaz

### 2. Tool-based RAG (Tek Tool)
```
Kullanıcı Sorusu
      ↓
    Agent
      ↓
  Karar: RAG gerekli mi?
   ↙        ↘
Evet       Hayır
  ↓          ↓
Vector DB   Direkt Cevap
  ↓          ↓
 LLM ← ← ← ←┘
  ↓
Cevap
```

**Özellikler:**
- ✅ Akıllı karar mekanizması
- ✅ Gereksiz DB sorgularını önler
- ❌ Hala tek kaynak

### 3. Multi-Tool RAG
```
Kullanıcı Sorusu
      ↓
    Agent
      ↓
  Hangi tool?
   ↙    ↓    ↘
RAG  Calc  Time  ...
  ↓    ↓    ↓
  └─→ LLM ←┘
      ↓
    Cevap
```

**Özellikler:**
- ✅ Çoklu kaynak desteği
- ✅ Her soru tipine özel tool
- ✅ Tool kombinasyonları
- ✅ Genişletilebilir

---

## Nasıl Çalışır?

### Adım 1: Tool'ları Tanımlayın

```python
from langchain_core.tools import tool
from langchain_core.tools import create_retriever_tool

# Tool 1: RAG Retriever
retriever_tool = create_retriever_tool(
    retriever,
    name="search_docs",
    description="Dokümanlarda arama yapmak için kullan"
)

# Tool 2: Hesap Makinesi
@tool
def calculator(expression: str) -> str:
    """Matematiksel hesaplamalar için kullan"""
    return str(eval(expression))

# Tool 3: Güncel Saat
@tool
def get_time() -> str:
    """Güncel saat bilgisi için kullan"""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

tools = [retriever_tool, calculator, get_time]
```

### Adım 2: Agent Oluşturun

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-haiku-20240307")
agent = create_react_agent(model, tools)
```

### Adım 3: Sorular Sorun

```python
# Agent otomatik olarak doğru tool'u seçer
response = agent.invoke({
    "messages": [("user", "What is task decomposition?")]
})
# Agent: "search_docs tool'unu kullanacağım"

response = agent.invoke({
    "messages": [("user", "What is 100 * 5?")]
})
# Agent: "calculator tool'unu kullanacağım"

response = agent.invoke({
    "messages": [("user", "What time is it?")]
})
# Agent: "get_time tool'unu kullanacağım"
```

---

## Tool Tanımlama

### Yöntem 1: `@tool` Decorator

```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """
    Veritabanında arama yapar.
    
    Args:
        query: Arama terimi
    
    Returns:
        Bulunan sonuçlar
    """
    # Arama mantığı
    results = db.search(query)
    return str(results)
```

**Önemli:**
- Docstring MUTLAKA olmalı (agent bunu okur!)
- Type hints ekleyin (`query: str`)
- Return type belirtin (` -> str`)

### Yöntem 2: `create_retriever_tool`

```python
from langchain_core.tools import create_retriever_tool

tool = create_retriever_tool(
    retriever=my_retriever,
    name="search_company_docs",  # Tool adı
    description="Şirket dökümanlarında arama yapmak için kullanın. "
                "Ürünler, politikalar, prosedürler hakkında sorular için uygundur."
)
```

**İpucu:**
- `description` çok önemli! Agent bunu okuyarak karar verir
- Net ve spesifik olun
- Ne zaman kullanılacağını açıklayın

### Yöntem 3: `Tool` Class

```python
from langchain_core.tools import Tool

def my_function(input: str) -> str:
    return f"Processed: {input}"

tool = Tool(
    name="my_tool",
    func=my_function,
    description="Bu tool X işi yapar"
)
```

---

## Agent Karar Mekanizması

### Agent Nasıl Karar Verir?

```python
# Kullanıcı sorusu
question = "What is task decomposition and what is 100 + 50?"

# Agent'ın düşünce süreci:
"""
1. Soruyu analiz et:
   - "task decomposition" → Bilgi gerektiriyor
   - "100 + 50" → Matematik gerektiriyor

2. Tool'ları değerlendir:
   - search_docs: "Dokümanlarda arama için" ✅ İlk kısım için uygun
   - calculator: "Matematik hesaplamalar için" ✅ İkinci kısım için uygun
   - get_time: "Saat bilgisi için" ❌ Bu soru için değil

3. Karar:
   - Önce search_docs tool'unu kullan → task decomposition bilgisini al
   - Sonra calculator tool'unu kullan → 100 + 50'yi hesapla
   - Her iki cevabı birleştir

4. Cevap oluştur
"""
```

### Description'ın Önemi

```python
# ❌ KÖTÜ Description
@tool
def my_tool(x: str) -> str:
    """A tool"""  # ← Agent ne zaman kullanacağını bilemiyor!
    return process(x)

# ✅ İYİ Description  
@tool
def weather_tool(city: str) -> str:
    """
    Belirtilen şehir için güncel hava durumu bilgisini getirir.
    
    Ne zaman kullanılır:
    - Kullanıcı hava durumunu sorduğunda
    - Sıcaklık, yağmur, rüzgar gibi sorularda
    - "bugün hava nasıl?" gibi sorularda
    
    Args:
        city: Şehir adı (örn: "Istanbul", "Ankara")
    """
    return get_weather(city)
```

---

## Gerçek Dünya Örnekleri

### Örnek 1: Müşteri Destek Sistemi

```python
# 4 farklı tool
tools = [
    create_retriever_tool(
        kb_retriever,
        name="search_knowledge_base",
        description="Ürün bilgileri, SSS, kullanım kılavuzları için kullan"
    ),
    
    create_retriever_tool(
        order_retriever,
        name="search_orders",
        description="Sipariş durumu, takip numarası, teslimat bilgileri için kullan"
    ),
    
    Tool(
        name="check_inventory",
        func=check_stock,
        description="Ürün stok durumunu kontrol etmek için kullan"
    ),
    
    Tool(
        name="create_ticket",
        func=create_support_ticket,
        description="Teknik destek talebi oluşturmak için kullan"
    )
]

# Kullanım
question = "X ürününün stoğu var mı ve nasıl kullanılır?"
# Agent:
# 1. check_inventory → Stok kontrolü
# 2. search_knowledge_base → Kullanım kılavuzu
# 3. Cevabı birleştir
```

### Örnek 2: Araştırma Asistanı

```python
tools = [
    create_retriever_tool(
        paper_retriever,
        name="search_papers",
        description="Bilimsel makaleler ve araştırmalarda arama yap"
    ),
    
    create_retriever_tool(
        code_retriever,
        name="search_code",
        description="GitHub ve kod örneklerinde arama yap"
    ),
    
    Tool(
        name="web_search",
        func=web_search,
        description="Güncel haberler ve web içeriği için kullan"
    ),
    
    Tool(
        name="calculator",
        func=calculate,
        description="İstatistik ve matematik hesaplamaları için kullan"
    )
]
```

### Örnek 3: E-Ticaret Asistanı

```python
tools = [
    create_retriever_tool(
        product_retriever,
        name="search_products",
        description="Ürün katalogunda arama yap"
    ),
    
    Tool(
        name="price_compare",
        func=compare_prices,
        description="Ürün fiyatlarını karşılaştır"
    ),
    
    Tool(
        name="check_delivery",
        func=estimate_delivery,
        description="Teslimat süresi ve ücretini hesapla"
    ),
    
    Tool(
        name="apply_coupon",
        func=validate_coupon,
        description="İndirim kodu geçerliliğini kontrol et"
    )
]

# Kullanım
question = "Laptop'ları listele ve SAVE20 kodum geçerli mi?"
# Agent:
# 1. search_products → Laptop listesi
# 2. apply_coupon → SAVE20 kontrolü
```

---

## İleri Seviye: Tool Zincirleme

### Ardışık Tool Kullanımı

```python
# Soru: "OpenAI hissesi bugün ne kadar ve dünle karşılaştır?"

# Agent süreci:
"""
1. get_stock_price("OPENAI") → $150
2. get_historical_price("OPENAI", "yesterday") → $145
3. calculator("150 - 145") → $5
4. Cevap: "OpenAI hissesi bugün $150, dünden $5 yüksek"
"""
```

### Paralel Tool Kullanımı

```python
# Soru: "Istanbul'da hava nasıl ve trafik durumu?"

# Agent süreci:
"""
Paralel olarak:
├─ weather_tool("Istanbul") → "Güneşli, 25°C"
└─ traffic_tool("Istanbul") → "Orta yoğunluk"

Birleştir: "Istanbul'da hava güneşli (25°C) ve trafik orta yoğunlukta"
"""
```

---

## Hata Yönetimi

### Tool Hataları

```python
@tool
def api_call(endpoint: str) -> str:
    """External API çağrısı yapar"""
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        # Agent'a anlamlı hata mesajı dön
        return f"❌ API hatası: {str(e)}. Lütfen daha sonra tekrar deneyin."
```

### Timeout Yönetimi

```python
@tool
def slow_operation(query: str) -> str:
    """Uzun sürebilecek işlem"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("İşlem çok uzun sürdü")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 saniye timeout
    
    try:
        result = expensive_operation(query)
        signal.alarm(0)  # Timeout'u iptal et
        return result
    except TimeoutError:
        return "⏱️ İşlem zaman aşımına uğradı, lütfen sorguyu basitleştirin"
```

---

## En İyi Pratikler

### 1. Tool İsimlendirme

```python
# ❌ KÖTÜ
@tool
def tool1(x):
    """does stuff"""
    pass

# ✅ İYİ
@tool
def search_customer_orders(customer_id: str):
    """
    Müşteri siparişlerini arar.
    customer_id: Müşteri ID numarası
    """
    pass
```

### 2. Description Yazma

```python
# ❌ KÖTÜ
description = "Searches stuff"

# ✅ İYİ
description = """
Ürün katalogunda arama yapar.

Ne zaman kullanılır:
- Kullanıcı ürün özellikleri sorduğunda
- Fiyat bilgisi istendiğinde
- Stok durumu öğrenilmek istendiğinde

Örnekler:
- "iPhone 15 özellikleri nedir?"
- "En ucuz laptop hangisi?"
- "Kablosuz kulaklık var mı?"
"""
```

### 3. Tool Sayısı

```python
# 🎯 İDEAL: 3-7 tool
tools = [
    search_tool,
    calculator_tool,
    time_tool
]

# ❌ ÇOK FAZLA: 20+ tool
# Agent karışır, yanlış tool seçimi yapar!

# ❌ ÇOK AZ: 1 tool
# O zaman multi-tool'a gerek yok!
```

### 4. Tool Gruplandırma

```python
# İlgili tool'ları gruplandırın

# Grup 1: Bilgi Arama
search_docs_tool
search_web_tool
search_database_tool

# Grup 2: Hesaplamalar
calculator_tool
statistics_tool
converter_tool

# Grup 3: Aksiyonlar
send_email_tool
create_ticket_tool
update_status_tool
```

---

## Özet

### Multi-Tool RAG Ne Zaman Kullanılır?

✅ **Kullan:**
- Birden fazla bilgi kaynağınız varsa
- Matematik, zaman, API çağrıları gibi farklı işlemler gerekiyorsa
- Akıllı yönlendirme istiyorsanız

❌ **Kullanma:**
- Sadece tek bir vector DB varsa → Basic RAG yeterli
- Çok basit soru-cevap senaryoları → Fazla karmaşık

### Anahtar Noktalar

1. **Tool Description = Agent'ın Beyni**
   - Ne kadar detaylı o kadar iyi karar

2. **3-7 Tool İdeal**
   - Çok fazla tool → Karışıklık
   - Çok az tool → Gereksiz

3. **Hata Yönetimi Önemli**
   - Tool hataları agent'ı kırmamalı

4. **Test, Test, Test**
   - Farklı soru tipleriyle test edin
   - Agent'ın kararlarını gözlemleyin

---

## Sonraki Adımlar

1. ✅ `simple_tool_rag.py` çalıştırın
2. ✅ `multi_tool_rag.py` ile deneyin
3. 🎯 Kendi tool'larınızı ekleyin
4. 🚀 Production'a geçin!

**Başarılar! 🎉**
