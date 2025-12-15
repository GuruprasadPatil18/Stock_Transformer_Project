import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import StockTransformer
from preprocess import get_data

# Settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "transformer_stock_model.pth"

def evaluate_model():
    print("--- Loading Data for Evaluation ---")
    # 1. Get Data and the Scaler (to reverse the scaling later)
    X_train, y_train, X_test, y_test, scaler = get_data()
    
    # Move test data to device
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    
    # 2. Load the Trained Model
    model = StockTransformer(input_dim=4, d_model=64, nhead=4, num_layers=2, dropout=0.1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode
    
    print("--- Making Predictions ---")
    with torch.no_grad():
        predictions = model(X_test)
        
        # Move back to CPU for plotting
        predictions = predictions.cpu().numpy()
        y_test_actual = y_test.cpu().numpy()
    
    # 3. Inverse Scaling (The Tricky Part)
    # Our scaler expects 4 columns (Close, RSI, MACD, EMA), but we only predicted 1 column (Close).
    # We need to create a dummy matrix to fool the scaler.
    
    # Create empty matrix with same shape as original data features
    dummy_pred = np.zeros((len(predictions), 4))
    dummy_actual = np.zeros((len(y_test_actual), 4))
    
    # Fill the first column (Close Price) with our data
    dummy_pred[:, 0] = predictions.flatten()
    dummy_actual[:, 0] = y_test_actual.flatten()
    
    # Inverse Transform
    actual_prices = scaler.inverse_transform(dummy_actual)[:, 0]
    predicted_prices = scaler.inverse_transform(dummy_pred)[:, 0]
    
    print("--- Plotting Results ---")
    
    # 4. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Price', color='blue', alpha=0.6)
    plt.plot(predicted_prices, label='AI Prediction', color='red', linestyle='dashed')
    
    plt.title('Stock Price Prediction: Transformer vs Reality')
    plt.xlabel('Days (Test Set)')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot for your report
    plt.savefig('prediction_result.png')
    print("Graph saved as 'prediction_result.png'. Check your folder!")
    plt.show()

if __name__ == "__main__":
    evaluate_model()