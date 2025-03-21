from flask import Flask, request, jsonify, Response
import json
import pdfplumber
import os
import re
import requests
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_pdf_from_url(url, dest_path):
    """Downloads a PDF from the given URL and saves it to the destination path."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        return True
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download PDF: {str(e)}"}

def categorize_transaction(narration):
    """Categorizes a transaction based on its narration."""
    narration = narration.lower()

    categories = {
        "Food": ["restaurant", "cafe", "food", "swiggy", "zomato", "hotel", "dining"],
        "Travel": ["train", "flight", "bus", "uber", "ola", "metro", "taxi", "tours", "trip"],
        "Investment": ["mutual fund", "sip", "groww", "zerodha", "navi", "investment", "stock"],
        "Utilities": ["electricity", "water", "gas", "mobile", "recharge", "rent", "dth", "broadband", "bill"],
        "Entertainment": ["movie", "netflix", "prime video", "steam", "game", "sports", "music", "concert"],
        "Medical": ["hospital", "doctor", "clinic", "medicine", "pharmacy", "health"],
        "Shopping": ["amazon", "flipkart", "myntra", "shopping", "electronics", "fashion", "mall"],
    }

    for category, keywords in categories.items():
        if any(keyword in narration for keyword in keywords):
            return category

    return "Other"  # Default category if no match is found

def extract_and_categorize_transactions(pdf_path, password=None):
    """Extract transactions from a bank statement PDF and categorize them."""
    transactions = []
    date_pattern = re.compile(r'^\d{2}-\d{2}-\d{4}$')

    try:
        with pdfplumber.open(pdf_path, password=password) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                lines = text.split('\n')

                for line in lines:
                    if 'Date' in line and 'Narration' in line and 'Balance' in line:
                        continue  # Skip headers

                    parts = line.split()
                    if len(parts) < 5:
                        continue  # Not enough parts for a valid transaction

                    date_part = parts[0]
                    if not date_pattern.match(date_part):
                        continue  # Skip lines that don't start with a valid date

                    date = date_part
                    balance = parts[-1].strip()
                    debit_credit = parts[-2].strip()
                    chq_ref_no = parts[-3].strip() if len(parts) >= 5 else ""
                    narration_parts = parts[1:-3]
                    narration = ' '.join(narration_parts) if narration_parts else "Unknown"

                    category = categorize_transaction(narration)

                    transaction = {
                        "Date": date,
                        "Narration": narration,
                        "Chq/Ref No": chq_ref_no,
                        "Withdrawal(Dr)/ Deposit(Cr)": debit_credit,
                        "Balance": balance,
                        "Category": category
                    }
                    transactions.append(transaction)

    except Exception as e:
        return {"error": f"Failed to process PDF: {str(e)}"}

    return transactions

@app.route('/upload_url', methods=['POST'])
def upload_file_from_url():
    """Uploads a bank statement PDF from a URL, processes it, and returns categorized transactions."""
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL is required"}), 400

    pdf_url = data["url"]
    password = data.get("password", None)

    filename = os.path.basename(pdf_url.split("?")[0])
    if not allowed_file(filename):
        return jsonify({"error": "Invalid file format from URL"}), 400

    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Download PDF from the URL
    download_result = download_pdf_from_url(pdf_url, filepath)
    if isinstance(download_result, dict) and "error" in download_result:
        return jsonify(download_result), 500

    # Extract and categorize transactions
    transactions = extract_and_categorize_transactions(filepath, password)
    if "error" in transactions:
        return jsonify(transactions), 500

    # Save categorized transactions
    json_filename = os.path.splitext(filename)[0] + '_categorized.json'
    json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
    with open(json_filepath, 'w') as json_file:
        json.dump(transactions, json_file, indent=4)

    # Return the categorized transactions as JSON response
    return Response(json.dumps(transactions, indent=4), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)