import torch
import torch.nn as nn
import torch.nn.functional as F

# Time2Vec Embedding Layer
class Time2VecEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Time2VecEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.sin_activation = nn.Linear(input_dim, embedding_dim)
        
    def forward(self, x):
        lin_part = self.linear(x)
        sin_part = torch.sin(self.sin_activation(x))
        return lin_part + sin_part

# Gated Residual Network (GRN) Layer
class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRN, self).__init__()
        self.W2 = nn.Linear(input_dim, hidden_dim)
        self.b2 = nn.Parameter(torch.zeros(hidden_dim))
        self.W1 = nn.Linear(hidden_dim, input_dim)
        self.b1 = nn.Parameter(torch.zeros(input_dim))
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, z):
        g2 = F.elu(self.W2(z) + self.b2)
        g1 = self.W1(g2) + self.b1
        glu_output = F.glu(g1, dim=-1)
        return self.layer_norm(z + glu_output)

# Encoder Layer with Multi-Head Attention and GRN
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.grn = GRN(embedding_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        grn_output = self.grn(x)
        x = self.norm2(x + grn_output)
        return x

# Decoder Layer with Masked Multi-Head Attention, GRN
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super(DecoderLayer, self).__init__()
        self.masked_attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.grn = GRN(embedding_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, memory):
        masked_attn_output, _ = self.masked_attention(x, x, x)
        x = self.norm1(x + masked_attn_output)
        attn_output, _ = self.attention(x, memory, memory)
        x = self.norm2(x + attn_output)
        grn_output = self.grn(x)
        x = self.norm3(x + grn_output)
        return x

# Full Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, hidden_dim, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = Time2VecEmbedding(input_dim, embedding_dim)
        
        # Encoder Stack
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        
        # Decoder Stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output Layer
        self.output_layer = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, x, y):
        # Embedding
        x = self.embedding(x)
        y = self.embedding(y)
        
        # Encoder Pass
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Decoder Pass
        for layer in self.decoder_layers:
            y = layer(y, x)
        
        # Output Layer - Linear + Sign + Softmax
        output = self.output_layer(y)
        output = torch.sign(output) * F.softmax(output, dim=-1)
        return output

# Example Usage
input_dim = 10  # Example input dimension
embedding_dim = 64  # Example embedding dimension
num_heads = 4
hidden_dim = 128
num_layers = 4
output_dim = 1  # Output dimension for regression/Sharpe loss

# Instantiate the model
model = TransformerModel(input_dim, embedding_dim, num_heads, hidden_dim, num_layers, output_dim)

# Sample Input (e.g., batch size of 2, sequence length of 5)
x = torch.randn(5, 2, input_dim)  # Sequence length x Batch size x Input dim
y = torch.randn(5, 2, input_dim)  # Same for decoder input

# Forward Pass
output = model(x, y)
print(output)
