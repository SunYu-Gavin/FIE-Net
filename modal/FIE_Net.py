import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

# Self-Attention Module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # Pre-Norm before attention
        x_norm = self.layernorm(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # Pre-Norm before feed-forward network
        x_norm = self.layernorm(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        return x


# Cross-Modal Attention Module
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key, value):
        query_norm = self.layernorm(query).transpose(0, 1)
        key_norm = key.transpose(0, 1)
        value_norm = value.transpose(0, 1)

        # Multi-head attention
        attn_output, _ = self.attn(query_norm, key_norm, value_norm)

        # Back to original shape
        attn_output = attn_output.transpose(0, 1)
        query = query + attn_output

        # Pre-Norm before feed-forward network
        query_norm = self.layernorm(query)
        ffn_output = self.ffn(query_norm)
        query = query + ffn_output
        return query


# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# EEG Embedding
class EEGEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(EEGEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, embed_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        return x


# Polynomial Module
class PolynomialModule(nn.Module):
    def __init__(self, input_dim=36, degree=2, activation='relu', output_dim=64):
        super(PolynomialModule, self).__init__()
        self.BN1 = torch.nn.BatchNorm1d(input_dim)
        self.BN2 = torch.nn.BatchNorm1d(output_dim)
        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = output_dim
        self.activation = self._get_activation(activation)
        self.fc = nn.LazyLinear(output_dim)

    def forward(self, x):
        x = self.BN1(x)
        batch_size, input_dim = x.size()
        assert input_dim == self.input_dim, "Input dimensions do not match."

        poly_features = []
        for d in range(1, self.degree + 1):
            for combination in product(range(input_dim), repeat=d):
                feature_product = torch.prod(x[:, combination], dim=1, keepdim=True)
                poly_features.append(feature_product)

        poly_features = torch.cat(poly_features, dim=1)
        reshaped_poly_features = self.activation(poly_features) if self.activation else poly_features
        reshaped_poly_features = reshaped_poly_features.view(batch_size, input_dim, -1)
        out = self.fc(reshaped_poly_features)
        out = self.BN2(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        return out

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation is None:
            return None
        else:
            raise ValueError(f"Unsupported activation function: {activation}")


# MLP Block
class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


# MLP Mixer Layer
class MLPMixerLayer(nn.Module):
    def __init__(self, num_tokens, embed_dim, token_dim, channel_dim):
        super(MLPMixerLayer, self).__init__()
        self.token_mixer = MLPBlock(num_tokens, token_dim)
        self.channel_mixer = MLPBlock(embed_dim, channel_dim)
        self.layer_norm1 = nn.LayerNorm(num_tokens)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.token_mixer(self.layer_norm1(x.transpose(1, 2))).transpose(1, 2)
        x = x + self.channel_mixer(self.layer_norm2(x))
        return x


# FIENet Model
class FIENet(nn.Module):
    def __init__(self):
        super(FIENet, self).__init__()
        self.eeg_embedding = EEGEmbedding(in_channels=17, embed_dim=64)
        self.eog_embedding = PolynomialModule()

        self.mlp_mixer_eeg = MLPMixerLayer(num_tokens=5, embed_dim=64, token_dim=128, channel_dim=64)
        self.mlp_mixer_eog = MLPMixerLayer(num_tokens=36, embed_dim=64, token_dim=64, channel_dim=64)

        self.cross_attention_eeg_to_eog = CrossModalAttention(embed_dim=64, num_heads=4)
        self.cross_attention_eog_to_eeg = CrossModalAttention(embed_dim=64, num_heads=4)

        self.transformer_eeg = TransformerEncoderLayer(embed_dim=64, num_heads=4)
        self.transformer_eog = TransformerEncoderLayer(embed_dim=64, num_heads=4)

        self.transformer_cross_eeg = TransformerEncoderLayer(embed_dim=64, num_heads=4)
        self.transformer_cross_eog = TransformerEncoderLayer(embed_dim=64, num_heads=4)

        self.dropout3 = nn.Dropout1d(p=0.3)
        self.fusion_fc = nn.LazyLinear(64)
        self.fc = nn.Linear(64, 1)

    def forward(self, eeg, eog):
        eeg_emb = self.eeg_embedding(eeg)
        eog_emb = self.eog_embedding(eog)
        eeg_mixer = self.mlp_mixer_eeg(eeg_emb)
        eog_mixer = self.mlp_mixer_eog(eog_emb)

        cross_attn_eeg_to_eog = self.cross_attention_eeg_to_eog(eeg_emb, eog_emb, eog_emb)
        cross_attn_eog_to_eeg = self.cross_attention_eog_to_eeg(eog_emb, eeg_emb, eeg_emb)

        eeg_transformer = self.transformer_eeg(eeg_mixer)
        eog_transformer = self.transformer_eog(eog_mixer)

        cross_eeg_transformer = self.transformer_cross_eeg(cross_attn_eeg_to_eog)
        cross_eog_transformer = self.transformer_cross_eog(cross_attn_eog_to_eeg)

        combined_features = torch.cat([eeg_transformer, eog_transformer, cross_eeg_transformer, cross_eog_transformer], dim=1)
        fused_features = self.fusion_fc(combined_features)
        combined_features_drop = self.dropout3(fused_features)
        output = torch.sigmoid(self.fc(combined_features_drop.mean(dim=1)))
        return output
