import torch.nn as nn

class StudentModel(nn.Module):
    """
    3-layer Transformer classifier.

    Args:
        num_labels      : int, number of output classes (default 2)
        hidden_size     : int, hidden dimension (default 256)
        num_heads       : int, attention heads (default 4)
        num_layers      : int, transformer layers (default 3)
        max_length      : int, max sequence length (default 128)
        vocab_size      : int, vocabulary size (default 30522 for bert-base-uncased)

    Forward input:
        input_ids       : LongTensor [batch, seq_len]
        attention_mask  : LongTensor [batch, seq_len]

    Forward output — dict with keys:
        "logits"        : FloatTensor [batch, num_labels]
        "attentions"    : list of FloatTensor [batch, num_heads, seq_len, seq_len]
                          one per layer, used for feature-based KD
    """
    def __init__(self, num_labels=2, hidden_size=256, num_heads=4,
                 num_layers=3, max_length=128, vocab_size=30522):
        raise NotImplementedError

    def forward(self, input_ids, attention_mask):
        raise NotImplementedError