from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# PostgreSQL database URL (e.g., "postgresql://user:password@host:port/dbname")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# -----------------------------
# Data Loading and Aggregation
# -----------------------------
def load_data_from_db():
    """
    Load transactions from the PostgreSQL database.
    Assumes a table named "transactions" with the necessary columns.
    """
    query = "SELECT * FROM transactions"
    df = pd.read_sql(query, engine)
    # Filter for debit transactions only (where spend is indicated with (Dr))
    df = df[df["Withdrawal(Dr)/ Deposit(Cr)"].str.contains(r"\(Dr\)")]
    # Extract numeric spend amount
    df['Amount'] = df["Withdrawal(Dr)/ Deposit(Cr)"].apply(
        lambda x: float(x.split("(")[0].replace(",", "").strip())
    )
    # Convert Date from string to datetime if needed
    # (Assumes the date is stored as text in DD-MM-YYYY format, adjust format if necessary)
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    return df

def aggregate_category(df, category):
    """
    For a given category, group transactions by Date and sum spend amounts.
    Missing dates are filled with 0 spend.
    """
    df_cat = df[df['Category'] == category]
    ts = df_cat.groupby('Date')['Amount'].sum().reset_index()
    ts = ts.sort_values('Date')
    if ts.empty:
        return None
    start_date = ts['Date'].min()
    end_date = ts['Date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    ts = ts.set_index('Date').reindex(all_dates, fill_value=0).rename_axis('Date').reset_index()
    return ts

# -----------------------------
# Time Series Dataset Definition
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len):
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len]
        y = self.series[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Transformer Model Definition
# -----------------------------
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (seq_len, batch, input_size)
        src = self.embedding(src)  # (seq_len, batch, d_model)
        output = self.transformer_encoder(src)  # (seq_len, batch, d_model)
        # Use the last time step's output for prediction
        output = self.fc_out(output[-1, :, :])  # (batch, 1)
        return output

# -----------------------------
# Training and Prediction Functions
# -----------------------------
def train_model(model, dataloader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x = x.unsqueeze(-1).transpose(0, 1)  # (seq_len, batch, 1)
            y = y.unsqueeze(-1)  # (batch, 1)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

def predict_future(model, series, seq_len, steps=7):
    model.eval()
    predictions = []
    current_seq = series[-seq_len:].copy()
    for _ in range(steps):
        x = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(-1).unsqueeze(1)  # (seq_len, 1, 1)
        with torch.no_grad():
            pred = model(x).item()
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], pred)
    return predictions

# -----------------------------
# Flask Endpoint for Next Week's Prediction
# -----------------------------
@app.route('/predict_next_week', methods=['POST'])
def predict_next_week():
    """
    Expects JSON payload:
    {
      "category": "Food"   // One of: Food, Travel, Investment, Utilities, Entertainment, Medical, Shopping
    }
    This endpoint predicts the spending for the next week (7 days) for the specified category.
    """
    data = request.get_json()
    if not data or "category" not in data:
        return jsonify({"error": "Category is required"}), 400

    category = data["category"]

    try:
        df = load_data_from_db()
    except Exception as e:
        return jsonify({"error": f"Error loading data from DB: {str(e)}"}), 500

    ts_df = aggregate_category(df, category)
    if ts_df is None:
        return jsonify({"error": f"No data found for category {category}"}), 404

    series = ts_df['Amount'].values

    # Fixed sequence length for model input (adjust if needed)
    seq_len = 5

    dataset = TimeSeriesDataset(series, seq_len)
    if len(dataset) < 1:
        return jsonify({"error": "Not enough data points for training"}), 400
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize and train the Transformer model
    model = TransformerTimeSeries()
    train_model(model, dataloader, epochs=20, lr=0.001)

    # Predict spending for the next 7 days
    future_steps = 7
    predictions = predict_future(model, series, seq_len, steps=future_steps)
    total_predicted = sum(predictions)

    return jsonify({
        "category": category,
        "next_week_predictions": predictions,
        "total_predicted_next_week": total_predicted
    })

if __name__ == '__main__':
    app.run(debug=True)
