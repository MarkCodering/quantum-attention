import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
num_tokens = 2000  # Vocabulary size
embed_size = 512  # Embedding dimension
num_heads = 8  # Number of heads in multi-head attention
num_layers = 3  # Number of transformer layers
dropout_rate = 0.1  # Dropout rate

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, num_tokens, embed_size, num_heads, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, embed_size, num_tokens))
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, 
                                          num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, 
                                          dropout=dropout)
        self.output_layer = nn.Linear(embed_size, num_tokens)

    def forward(self, src, tgt):
        src = self.token_embedding(src) + self.positional_encoding[:, :, :src.size(1)]
        tgt = self.token_embedding(tgt) + self.positional_encoding[:, :, :tgt.size(1)]
        output = self.transformer(src, tgt)
        return self.output_layer(output)

# Instantiate the model
model = TransformerModel(num_tokens, embed_size, num_heads, num_layers, dropout_rate)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example input (batch size, sequence length)
src = torch.randint(0, num_tokens, (10, 35))  # Example source batch
tgt = torch.randint(0, num_tokens, (10, 35))  # Example target batch

# Training Loop (simplified)
for epoch in range(1):
    optimizer.zero_grad()
    output = model(src, tgt)
    loss = criterion(output.view(-1, num_tokens), tgt.view(-1))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')