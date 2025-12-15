import torch
import torch.nn as nn
import torch.optim as optim
from model import StockTransformer
from preprocess import get_data
import time

# --- Hyperparameters (Settings you can tweak) ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50  # How many times to loop over the data
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using Device: {DEVICE}")

def train_model():
    # 1. Get Data
    X_train, y_train, X_test, y_test, scaler = get_data()
    
    # Move data to GPU/CPU
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
    
    # 2. Initialize Model
    model = StockTransformer(input_dim=4, d_model=64, nhead=4, num_layers=2, dropout=0.1)
    model.to(DEVICE)
    
    # 3. Define Loss and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("--- Starting Training ---")
    start_time = time.time()
    
    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        
        # Shuffle batches (Simplified manual batching for clarity)
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        
        for i in range(0, X_train.size()[0], BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            # A. Forward Pass (Make a prediction)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # B. Backward Pass (Calculate errors)
            optimizer.zero_grad() # Clear old gradients
            loss.backward()       # Calculate new gradients
            optimizer.step()      # Update model weights
            
            epoch_loss += loss.item()
            
        # Average loss for this epoch
        avg_train_loss = epoch_loss / (X_train.size()[0] / BATCH_SIZE)
        train_losses.append(avg_train_loss)
        
        # Validation (Test on unseen data)
        model.eval() # Set to evaluation mode
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_losses.append(test_loss)
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}")

    print(f"--- Training Complete in {time.time() - start_time:.2f} seconds ---")
    
    # 4. Save the Model
    torch.save(model.state_dict(), "transformer_stock_model.pth")
    print("Model saved as 'transformer_stock_model.pth'")
    
    return train_losses, test_losses

if __name__ == "__main__":
    train_model()