from flask import Flask, request, jsonify
from flask_cors import CORS  # Added CORS support
import google.generativeai as genai
import dotenv
import os

dotenv.load_dotenv()
app = Flask(__name__)

# Enable CORS for all routes under /api with specific origins
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://localhost:5174"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ====== CONFIGURATION ====== #
API_KEY = os.getenv('GEMINI_API')

SYSTEM_GUIDELINES = """
You are an expert financial assistant for StockMaster Pro. 
Your responses must:
1. Only answer questions related to stocks, investments, and personal finance
2. Reject other topics with: "I specialize only in financial topics."
3. Use simple language with examples
4. Cite reliable financial sources
"""
# =========================== #

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    'gemini-2.0-flash',
    system_instruction=SYSTEM_GUIDELINES
)

def _build_cors_preflight_response():
    response = jsonify({"status": "preflight"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response

@app.route('/api/finance-chat', methods=['POST', 'OPTIONS'])
def finance_chat():
    """Chat endpoint with CORS handling"""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '').strip()
        
        if not user_prompt:
            return jsonify({"error": "Empty prompt"}), 400

        response = model.generate_content(user_prompt)
        return jsonify({
            "user_prompt": user_prompt,
            "bot_response": response.text
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)