import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape: (1, max_len, dim)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class PointTransformerBlockWithPE(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(PointTransformerBlockWithPE, self).__init__()
        self.self_attention = nn.MultiheadAttention(output_dim, num_heads)
        self.positional_encoding = PositionalEncoding(output_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Add positional encoding to the input
        x_pe = self.positional_encoding(x)
        
        # Self-attention layer
        attn_output, _ = self.self_attention(x_pe, x_pe, x_pe)
        x = x + self.norm1(attn_output)  # Residual connection and normalization
        
        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = x + self.norm2(ff_output)  # Residual connection and normalization
        return x


class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=3, input_dim=3, feature_dim=64):
        super(ObjectDetectionModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, feature_dim)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([PointTransformerBlockWithPE(feature_dim, feature_dim) for _ in range(4)])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # x, y, z, width, height, depth, yaw
        )

    def forward(self, x):
        # Initial embedding
        x = self.embedding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Classification and regression heads
        class_output = self.classifier(x.mean(dim=1))  # Aggregating features across points
        bbox_output = self.regressor(x.mean(dim=1))
        
        return class_output, bbox_output