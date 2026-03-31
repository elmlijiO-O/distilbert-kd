# INTERFACE.md — Project Contract

This file defines all shared interfaces.
**Do not change function signatures without discussing first.**

---

## Repo structure
See README.md for full folder layout.
Branches: `main`, `partner-a` (teacher + response KD), `partner-b` (student + feature KD)

---

## Data contract (`src/data.py`) — DONE, do not modify

`get_tokenizer()` → returns a BertTokenizerFast

`get_datasets(tokenizer)` → returns dict: `{ 'train', 'validation', 'test' }`
Each dataset has columns: `input_ids [128]`, `attention_mask [128]`, `label (0/1)`

`get_dataloaders(datasets)` → returns dict of DataLoaders, batch_size=32

---

## Model output contract

Both TeacherModel and StudentModel **must** return a dict from `.forward()`:
```python
{
  "logits":     FloatTensor,  # shape [batch, 2]
  "attentions": list          # list of attention tensors per layer
}
```

This is mandatory. The training loop and loss functions depend on this.

---

## Checkpoint format

Save checkpoints like this (both models):
```python
torch.save({
    "epoch":       epoch,
    "model_state": model.state_dict(),
    "val_accuracy": val_acc,
    "config":      model_config_dict,
}, "checkpoints/teacher_best.pt")
```
Student checkpoint → `checkpoints/student_best.pt`

---

## Loss functions (`src/losses.py`)

| Function | Owner | Status |
|---|---|---|
| `response_kd_loss(student_logits, teacher_logits, temperature)` | Partner A | To implement |
| `feature_kd_loss(student_attentions, teacher_attentions)` | Partner B | To implement |
| `combined_loss(ce, kd, feat, alpha, beta, gamma)` | Done | ✅ |

**Attention layer mapping (teacher 12 layers → student 3 layers):**
Student layer 0 ← Teacher layer 3
Student layer 1 ← Teacher layer 7
Student layer 2 ← Teacher layer 11

---

## API contract (`api/`)

Both `/predict` endpoints must accept and return the same JSON schema:

**Request:**
```json
{ "text": "This movie was absolutely fantastic!" }
```

**Response:**
```json
{
  "label":      "positive",
  "confidence": 0.97,
  "model":      "teacher"
}
```
`model` field is either `"teacher"` or `"student"`.

---

## Constants (do not change)

| Name | Value |
|---|---|
| `TOKENIZER_NAME` | `"bert-base-uncased"` |
| `MAX_LENGTH` | `128` |
| `BATCH_SIZE` | `32` |
| `NUM_LABELS` | `2` |
| `TEMPERATURE` | `4.0` |
| `ALPHA` | `0.5` |
| `BETA` | `0.3` |
| `GAMMA` | `0.2` |

---

## Git workflow

- Never push directly to `main`
- Open a PR when your phase is done — the other person must review before merge
- Commit messages: `feat:`, `fix:`, `docs:`, `refactor:`