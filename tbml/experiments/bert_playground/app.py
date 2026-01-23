from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline

app = Flask(__name__)

# Load the BERT fill-mask pipeline (this will download ~400MB on first run)
# We use bert-base-uncased which expects the [MASK] token.
unmasker = pipeline("fill-mask", model="bert-base-uncased")

# HTML Template (Inline for simplicity)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BERT Mask Playground</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }
        textarea { width: 100%; height: 100px; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px; }
        button { background: #4A90E2; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        .result-box { margin-top: 20px; border-top: 2px solid #eee; padding-top: 20px; }
        .mask-group { margin-bottom: 20px; background: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 5px solid #4A90E2; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        .prob-bar { background: #4A90E2; height: 18px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>BERT Mask Playground 🤖</h1>
    <p>Enter text using <strong>[MASK]</strong> where you want BERT to predict the word.</p>
    <textarea id="textInput">The capital of France is [MASK].</textarea>
    <br>
    <label>Top K: </label>
    <input type="number" id="topK" value="5" min="1" max="20" style="width: 50px;">
    <button onclick="predict()">Predict</button>

    <div id="results" class="result-box"></div>

    <script>
        async function predict() {
            const text = document.getElementById('textInput').value;
            const k = document.getElementById('topK').value;
            const resDiv = document.getElementById('results');
            resDiv.innerHTML = "Processing...";

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text, k: parseInt(k) })
            });
            const data = await response.json();
            
            if (data.error) {
                resDiv.innerHTML = `<p style="color:red">${data.error}</p>`;
                return;
            }

            resDiv.innerHTML = "";
            // Ensure data is treated as an array of lists (one list per mask)
            const predictions = Array.isArray(data[0]) ? data : [data];

            predictions.forEach((maskResults, index) => {
                let html = `<div class="mask-group"><h3>Mask #${index + 1}</h3><table>`;
                html += "<tr><th>Token</th><th>Probability</th><th>Confidence</th></tr>";
                maskResults.forEach(res => {
                    const percentage = (res.score * 100).toFixed(2);
                    html += `<tr>
                        <td><strong>${res.token_str}</strong></td>
                        <td>${res.score.toFixed(4)}</td>
                        <td style="width: 200px;">
                            <div class="prob-bar" style="width: ${percentage}%"></div>
                            <small>${percentage}%</small>
                        </td>
                    </tr>`;
                });
                html += "</table></div>";
                resDiv.innerHTML += html;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    k = data.get('k', 5)
    
    if '[MASK]' not in text:
        return jsonify({"error": "Please include at least one [MASK] token in your text."})
    
    try:
        # The pipeline returns a list if 1 mask is present, or a list of lists if multiple are present
        results = unmasker(text, top_k=k)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8798)
