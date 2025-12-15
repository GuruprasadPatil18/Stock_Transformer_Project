import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer (not a learnable parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to the input
        # x shape: (Batch, Seq_Len, Features)
        return x + self.pe[:, :x.size(1), :]

class StockTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(StockTransformer, self).__init__()
        
        # 1. Input Embedding
        # We project our 4 features (Close, RSI, etc.) up to 'd_model' dimensions (e.g., 64)
        # This gives the model more "space" to learn complex patterns.
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder (The "Brain")
        # nhead=4 means it looks at the data in 4 different ways simultaneously
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Layer
        # We take the complex features and map them back to 1 single number (Predicted Price)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: [Batch, Seq_Len, Features]
        
        # Step 1: Embed and add position info
        src = self.input_linear(src) 
        src = self.pos_encoder(src)
        
        # Step 2: Pass through Transformer
        output = self.transformer_encoder(src)
        
        # Step 3: We only care about the output of the LAST time step for forecasting
        # (The model predicts based on the full sequence, but we want the final result)
        last_time_step_output = output[:, -1, :]
        
        # Step 4: Final prediction
        prediction = self.decoder(last_time_step_output)
        
        return prediction

# Quick Test to see if it works
if __name__ == "__main__":
    # Simulate some random data: 32 samples, 60 days, 4 features
    dummy_input = torch.rand(32, 60, 4)
    
    # Initialize model
    model = StockTransformer()
    
    # Get prediction
    output = model(dummy_input)
    
    print("Model Structure Created Successfully.")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") # Should be (32, 1)