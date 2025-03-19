from flask import Flask, request, jsonify
import google.generativeai as genai
import dotenv
import os

dotenv.load_dotenv()
app = Flask(__name__)

# ====== CONFIGURATION ====== #
API_KEY = os.getenv('GEMINI_API') # Replace with your key

# System instruction to control responses
SYSTEM_GUIDELINES = """
You are an expert financial assistant for StockMaster Pro. 
Your responses must:
1. Only answer questions related to stocks, investments, and personal finance
2. Reject any other topics (like fashion, cooking, etc.) with: 
   "I specialize only in financial topics. Please ask about stocks or investments."
3. Use simple language with examples where appropriate
4. Cite reliable financial sources when possible
"""
# =========================== #

# Initialize Gemini with system guidelines
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    'gemini-2.0-flash',
    system_instruction=SYSTEM_GUIDELINES
)

@app.route('/api/finance-chat', methods=['POST'])
def finance_chat():
    """Chat endpoint with domain restrictions"""
    try:
        # Get user input
        data = request.get_json()
        user_prompt = data.get('prompt', '').strip()
        
        if not user_prompt:
            return jsonify({"error": "Empty prompt"}), 400

        # Generate response with enforced guidelines
        response = model.generate_content(user_prompt)
        
        return jsonify({
            "user_prompt": user_prompt,
            "bot_response": response.text
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)