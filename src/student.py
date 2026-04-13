import torch
import torch.nn as nn


class TransformerLayerWithAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ffn_output))
        return x, attn_weights


class StudentModel(nn.Module):
    def __init__(self, num_labels=2, hidden_size=256, num_heads=4,
                 num_layers=3, max_length=128, vocab_size=30522):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayerWithAttention(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        x = self.embedding(input_ids) + self.position_embedding(positions)

        key_padding_mask = (attention_mask == 0)

        attentions = []
        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask=key_padding_mask)
            attentions.append(attn_weights)

        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)

        return {
            "logits": logits,
            "attentions": attentions
        }