import torch
import torch.nn as nn
from transformers import BertModel

class TeacherModel(nn.Module):
    """
    BERT-base fine-tuned classifier.

    Forward input:
        input_ids      : LongTensor [batch, seq_len]
        attention_mask : LongTensor [batch, seq_len]

    Forward output — dict:
        "logits"      : FloatTensor [batch, num_labels]
        "attentions"  : list of 12 FloatTensor [batch, heads, seq_len, seq_len]
    """

    def __init__(self, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            output_attentions=True,   # mandatory — partner's loss needs these
        )
        hidden_size = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # outputs.last_hidden_state : [batch, seq_len, 768]
        # outputs.pooler_output     : [batch, 768]  — CLS token, projected
        # outputs.attentions        : tuple of 12 tensors [batch, 12, seq_len, seq_len]

        pooled = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled)

        return {
            "logits":     logits,
            "attentions": list(outputs.attentions),
        }