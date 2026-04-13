import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, num_labels=2, hidden_size=256, num_heads=4,
                 num_layers=3, max_length=128, vocab_size=30522):
        super().__init__()

        # 1. Embedding: converts token IDs → vectors of size hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        # 2. Positional encoding: tells the model where each token is in the sequence
        self.position_embedding = nn.Embedding(max_length, hidden_size)

        # 3. Transformer layers (this is where attention happens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True  # input shape: [batch, seq, hidden]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Classifier head: maps [CLS] vector → class scores
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length

        def forward(self, input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape

            # Build position indices [0, 1, 2, ..., seq_len-1] for each item in batch
            positions = torch.arange(seq_len, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)  # [batch, seq_len]

            # Combine token embeddings + position embeddings
            x = self.embedding(input_ids) + self.position_embedding(positions)

            # Convert attention_mask for PyTorch: it expects True where tokens should be IGNORED
            # Our mask has 1=real token, 0=padding → invert it
            key_padding_mask = (attention_mask == 0)

            # Run through transformer layers, collecting attention weights per layer
            attentions = []
            for layer in self.transformer.layers:
                attn_output, attn_weights = layer.self_attn(
                    x, x, x,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False
                )
                attentions.append(attn_weights)
                x = layer.norm1(x + layer.dropout1(attn_output))   # dropout on attention output
                ffn_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(ffn_output))    # dropout on ffn output
        # Take the [CLS] token (position 0) as the sentence representation
        cls_output = x[:, 0, :]  # [batch, hidden_size]

        logits = self.classifier(cls_output)  # [batch, num_labels]

        return {
            "logits": logits,
            "attentions": attentions  # list of 3 tensors [batch, heads, seq, seq]
        }