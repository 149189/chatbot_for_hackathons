from flask import Flask, request, jsonify, render_template
import json
import pdfplumber
import os
import re
import requests
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai  # Import the Gemini library

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')  # Use the Gemini 2.0 Flash model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_transactions_from_pdf(pdf_path, password=None):
    transactions = []
    date_pattern = re.compile(r'^\d{2}-\d{2}-\d{4}$')  # Date format DD-MM-YYYY

    try:
        with pdfplumber.open(pdf_path, password=password) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                lines = text.split('\n')
                for line in lines:
                    # Skip header lines if they contain the words "Date", "Narration", and "Balance"
                    if 'Date' in line and 'Narration' in line and 'Balance' in line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue  # Not enough parts for a valid transaction
                    # Check if the first part is a date
                    date_part = parts[0]
                    if not date_pattern.match(date_part):
                        continue
                    # Extract fields by position
                    date = date_part
                    balance = parts[-1].strip()
                    debit_credit = parts[-2].strip()
                    chq_ref_no = parts[-3].strip() if len(parts) >= 5 else ""
                    # Narration is everything between date and the last three parts
                    narration_parts = parts[1:-3]
                    if not narration_parts and len(parts) >= 4:
                        narration_parts = parts[1:2]
                    narration = ' '.join(narration_parts)
                    
                    transaction = {
                        "Date": date,
                        "Narration": narration,
                        "Chq/Ref No": chq_ref_no,
                        "Withdrawal(Dr)/ Deposit(Cr)": debit_credit,
                        "Balance": balance
                    }
                    transactions.append(transaction)
    except Exception as e:
        return {"error": str(e)}
    
    return transactions

def analyze_with_gemini(transactions):
    try:
        # Convert transactions to a string for input to the Gemini model
        transactions_str = json.dumps(transactions, indent=2)
        
        # Define the prompt for the Gemini model with explicit instructions
        prompt = f"""
Analyze the following transaction data and provide the following:
1. Predicted spends in different categories (e.g., Food, Entertainment).
2. Category for each transaction.

Return the response in JSON format **only** with the following structure:
{{
    "Predicted_spends": {{
        "Category1": "Amount1",
        "Category2": "Amount2"
    }},
    "Category": {{
        "Transaction1 Narration": "Category1",
        "Transaction2 Narration": "Category2"
    }}
}}

Do NOT include any explanations, markdown formatting, or extra text.
Use only valid JSON syntax with double quotes for keys and values.

Transaction Data:
{transactions_str}
        """
        
        # Send the prompt to the Gemini model
        response = model.generate_content(prompt)
        
        # Clean the response text (remove markdown formatting if present)
        response_text = response.text.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Debug: Print the raw Gemini response
        print("Raw Gemini response:", response_text)
        
        # Parse the cleaned response text as JSON
        response_json = json.loads(response_text)
        return response_json
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON from Gemini: {str(e)}. Response: {response_text}"}
    except Exception as e:
        return {"error": f"Error analyzing transactions with Gemini: {str(e)}"}

def download_pdf_from_url(url, dest_path):
    """
    Downloads a PDF from the provided URL and saves it to dest_path.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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
        
        # Extract transactions from the PDF file
        transactions = extract_transactions_from_pdf(filepath, password)
        
        if "error" in transactions:
            return jsonify(transactions)
        
        # Save the extracted transactions to a JSON file (optional)
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        with open(json_filepath, 'w') as json_file:
            json.dump(transactions, json_file, indent=4)
        
        # Analyze transactions with Gemini
        gemini_response = analyze_with_gemini(transactions)
        if "error" in gemini_response:
            return jsonify(gemini_response)
        
        return jsonify({
            "transactions": transactions,
            "gemini_analysis": gemini_response,
            "json_file": json_filename
        })
    
    return jsonify({"error": "Invalid file format"})

@app.route('/upload_url', methods=['POST'])
def upload_file_from_url():
    # Accept a JSON payload with the URL (and optionally a password)
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL is required"}), 400
    
    pdf_url = data["url"]
    password = data.get("password", None)
    
    # Define a filename for the downloaded PDF
    filename = os.path.basename(pdf_url.split("?")[0])
    if not allowed_file(filename):
        return jsonify({"error": "Invalid file format from URL"}), 400
    
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Download the PDF from the URL
    download_result = download_pdf_from_url(pdf_url, filepath)
    if isinstance(download_result, dict) and "error" in download_result:
        return jsonify(download_result), 500
    
    # Extract transactions from the downloaded PDF
    transactions = extract_transactions_from_pdf(filepath, password)
    if "error" in transactions:
        return jsonify(transactions)
    
    # Save the extracted transactions to a JSON file (optional)
    json_filename = os.path.splitext(filename)[0] + '.json'
    json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
    with open(json_filepath, 'w') as json_file:
        json.dump(transactions, json_file, indent=4)
    
    # Analyze transactions with Gemini
    gemini_response = analyze_with_gemini(transactions)
    if "error" in gemini_response:
        return jsonify(gemini_response)
    
    return jsonify({
        "transactions": transactions,
        "gemini_analysis": gemini_response,
        "json_file": json_filename
    })

@app.route('/demo', methods=['GET'])
def demo():
    # Sample transaction data for demonstration purposes
    sample_transactions = [
        {
            "Date": "07-01-2024",
            "Narration": "UPI/Mrs. RANI KISHO/400767825706/UPI",
            "Chq/Ref No": "UPI-400725221113",
            "Withdrawal(Dr)/ Deposit(Cr)": "200.00(Cr)",
            "Balance": "238.10(Cr)"
        },
        {
            "Date": "07-01-2024", 
            "Narration": "UPI/JAGDAMBA BHOJAN/437308394607/UPI",
            "Chq/Ref No": "UPI-400725428424",
            "Withdrawal(Dr)/ Deposit(Cr)": "200.00(Dr)",
            "Balance": "38.10(Cr)"
        }
    ]
    
    # Sample Gemini API response for demonstration purposes
    sample_gemini_response = {
        "Predicted_spends": {
            "Food": 500.00,
            "Entertainment": 640.00,
            "Transfer": 200.00
        },
        "Category": {
            "07-01-2024 UPI/Mrs. RANI KISHO/400767825706/UPI": "Transfer",
            "07-01-2024 UPI/JAGDAMBA BHOJAN/437308394607/UPI": "Food"
        }
    }
    
    return jsonify({
        "transactions": sample_transactions,
        "gemini_analysis": sample_gemini_response
    })

if __name__ == '__main__':
    app.run(debug=True)
