# chroma_web_viewer.py - Basit Web Arayüzü (Flask)

"""
CHROMA DB İÇİN BASİT WEB ARAYÜZÜ

KURULUM:
  pip install flask

KULLANIM:
  python chroma_web_viewer.py
  Tarayıcıda: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = Flask(__name__)

# Embeddings ve vectorstore'u global olarak yükle
print("⏳ Chroma DB yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
print("✅ Hazır!")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Chroma DB Viewer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 { color: #333; text-align: center; }
        .stats {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .search-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        input[type="text"] {
            width: 70%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover { background-color: #0056b3; }
        .result {
            background: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .score {
            color: #28a745;
            font-weight: bold;
        }
        .content {
            margin-top: 10px;
            line-height: 1.6;
            color: #555;
        }
        .loading { display: none; color: #007bff; }
    </style>
</head>
<body>
    <h1>🔍 Chroma DB Viewer</h1>
    
    <div class="stats">
        <h3>📊 Database İstatistikleri</h3>
        <p><strong>Toplam Döküman:</strong> {{ total_docs }}</p>
        <p><strong>Konum:</strong> ./chroma_db/</p>
    </div>
    
    <div class="search-box">
        <h3>🔎 Arama</h3>
        <input type="text" id="query" placeholder="Sorunuzu yazın...">
        <button onclick="search()">Ara</button>
        <p class="loading" id="loading">⏳ Aranıyor...</p>
    </div>
    
    <div id="results"></div>
    
    <script>
        function search() {
            const query = document.getElementById('query').value;
            if (!query) return;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            fetch('/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query, k: 5})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                let html = '<h3>📚 Sonuçlar (' + data.results.length + ' bulundu)</h3>';
                
                data.results.forEach((result, i) => {
                    html += `
                        <div class="result">
                            <h4>Sonuç ${i+1} <span class="score">(Skor: ${result.score.toFixed(4)})</span></h4>
                            <div class="content">${result.content}</div>
                        </div>
                    `;
                });
                
                document.getElementById('results').innerHTML = html;
            });
        }
        
        // Enter tuşu ile arama
        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') search();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    total_docs = vectorstore._collection.count()
    return render_template_string(HTML_TEMPLATE, total_docs=total_docs)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 5)
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    response = {
        'results': [
            {
                'content': doc.page_content,
                'score': float(score),
                'metadata': doc.metadata
            }
            for doc, score in results
        ]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 WEB ARAYÜZÜ BAŞLATILIYOR")
    print("="*60)
    print("\n📍 Tarayıcınızda açın: http://localhost:5000")
    print("⏹️  Durdurmak için: Ctrl+C")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, port=5000)
