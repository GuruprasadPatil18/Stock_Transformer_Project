import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def get_data(csv_path='stock_data.csv', sequence_length=60):
    print("--- Starting Preprocessing ---")
    
    # 1. Load the CSV we created in Phase 1
    df = pd.read_csv(csv_path)
    
    # We only want specific columns (Features)
    # 'Close' is what we want to predict. Others are helpers.
    feature_cols = ['Close', 'RSI', 'MACD', 'EMA_20']
    
    # Extract the numbers as a matrix
    data_matrix = df[feature_cols].values
    
    print(f"Original Data Shape: {data_matrix.shape}")

    # 2. Scaling (Normalization)
    # Transformers work best when data is between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_matrix)

    # 3. Create Sequences (Sliding Window)
    # We want: Input = Past 60 days -> Output = Next 1 day price
    X = [] # Input sequences
    y = [] # Target values (Close Price)

    for i in range(len(data_scaled) - sequence_length):
        # Take a window of data (e.g., Day 0 to Day 59)
        window = data_scaled[i : i + sequence_length]
        
        # Target is the 'Close' price of the NEXT day (Day 60)
        # Note: 'Close' is at index 0 in our feature_cols
        target = data_scaled[i + sequence_length, 0] 
        
        X.append(window)
        y.append(target)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # 4. Split into Train and Test (80% Train, 20% Test)
    split_index = int(len(X) * 0.8)
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    print(f"Training Data: {X_train.shape}") # (Samples, Seq_Len, Features)
    print(f"Testing Data: {X_test.shape}")

    # 5. Convert to PyTorch Tensors (The format the model understands)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # Shape becomes (Samples, 1)
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Return everything we need
    return X_train, y_train, X_test, y_test, scaler

# Test the function
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler = get_data()
    print("Preprocessing Complete. Tensors ready.")