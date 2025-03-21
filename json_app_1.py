from flask import Flask, jsonify, request
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_amount(amount_str):
    """
    Given a string like "200.00(Dr)" or "200.00(Cr)", extract the numeric part as a float.
    """
    try:
        amt = amount_str.split("(")[0].replace(",", "").strip()
        return float(amt)
    except Exception:
        return 0.0

def categorize_transaction(narration):
    """
    Simple categorization based on narration.
    For example, use the second segment if available.
    """
    parts = narration.split("/")
    if len(parts) > 1:
        return parts[1].strip()
    return "General"

def gemini_analyze(transactions):
    """
    Simulate Gemini analysis by grouping debit transactions (spends) by category and predicting future spends.
    Returns a list of dictionaries with keys:
      "Category", "Amount", "Spends", "Predicted Future Spends"
    """
    results = {}
    for tx in transactions:
        narration = tx.get("Narration", "")
        category = categorize_transaction(narration)
        amt_field = tx.get("Withdrawal(Dr)/ Deposit(Cr)", "")
        # Only count transactions that are debits (spends)
        if "(Dr)" in amt_field:
            amt = parse_amount(amt_field)
        else:
            amt = 0.0

        if category in results:
            results[category]["spends"] += amt
        else:
            results[category] = {"spends": amt}
    
    gemini_response = []
    for cat, data in results.items():
        predicted = data["spends"] * 1.1  # example: 10% increase prediction
        gemini_response.append({
            "Category": cat,
            "Amount": data["spends"],
            "Spends": data["spends"],
            "Predicted Future Spends": predicted
        })
    return gemini_response

@app.route('/gemini', methods=['GET'])
def gemini_endpoint():
    """
    Reads bank_statement.json from the working directory,
    performs analysis, and returns a JSON response with each category's predicted future spends.
    """
    file_path = os.path.join(os.getcwd(), 'bank_statement.json')
    try:
        with open(file_path, 'r') as f:
            transactions = json.load(f)
    except Exception as e:
        return jsonify({"error": "Could not read bank_statement.json", "details": str(e)}), 500

    analysis = gemini_analyze(transactions)
    # Format the output as required: { "Category": "(output)", "Predicted_spends": "(output)" }
    output = []
    for entry in analysis:
        output.append({
            "Category": entry["Category"],
            "Predicted_spends": entry["Predicted Future Spends"]
        })
    
    return jsonify(output)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint to upload a PDF file.
    Expects the file in a form field named 'file' and an optional 'password' field.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    password = request.form.get('password', None)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({"message": "File uploaded successfully", "filepath": filepath})
    else:
        return jsonify({"error": "File type not allowed"})

if __name__ == '__main__':
    app.run(debug=True)
