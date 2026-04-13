# DistilBERT Reproduction — Sentiment Analysis

Reproduction of the DistilBERT approach for binary sentiment classification on film reviews (IMDb dataset).

A `bert-base-uncased` Teacher model is distilled into a lightweight 3-layer Transformer Student using two techniques combined:

- **Response-Based KD** — KL divergence between Teacher and Student logits
- **Feature-Based KD** — attention transfer loss between mapped Teacher and Student layers

---

## Repo Structure

```
distilbert-reproduction/
├── src/
│   ├── data.py           # tokenization, dataset loading, dataloaders
│   ├── teacher.py        # BERT-base Teacher model
│   ├── student.py        # lightweight 3-layer Transformer Student model
│   ├── losses.py         # response KD loss, feature KD loss, combined loss
│   └── train.py          # full training loop
├── api/
│   ├── app.py            # FastAPI endpoints for Teacher and Student
│   └── schemas.py        # request/response schemas
├── notebooks/
│   └── exploration.ipynb # dataset exploration
├── scripts/
│   └── benchmark.py      # API throughput benchmarking
├── checkpoints/          # saved model weights (gitignored)
├── config.yaml           # all hyperparameters — edit here, not in code
├── INTERFACE.md          # shared contracts between partners
└── requirements.txt
```

---

## Setup

**1. Clone the repo**

```bash
git clone <repo-url>
cd distilbert-reproduction
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Configuration

All hyperparameters are controlled from `config.yaml` — never hardcode values in the source files.

Key parameters:

| Parameter        | Value               | Description                             |
| ---------------- | ------------------- | --------------------------------------- |
| `teacher_name`   | `bert-base-uncased` | HuggingFace model name for Teacher      |
| `student_layers` | `3`                 | number of Transformer layers in Student |
| `hidden_size`    | `256`               | Student hidden dimension                |
| `temperature`    | `4.0`               | softening temperature for response KD   |
| `alpha`          | `0.5`               | weight for response-based KD loss       |
| `beta`           | `0.3`               | weight for feature-based KD loss        |
| `gamma`          | `0.2`               | weight for cross-entropy loss           |
| `batch_size`     | `32`                | training batch size                     |
| `epochs`         | `3`                 | number of training epochs               |

---

## Training

**1. Train the Teacher first:**

```bash
python src/train.py --mode teacher
```

Checkpoint saved to `checkpoints/teacher_best.pt`

**2. Distill the Student:**

```bash
python src/train.py --mode student
```

Checkpoint saved to `checkpoints/student_best.pt`

---

## Running the API

Make sure the checkpoints exist before launching.

```bash
uvicorn api.app:app --reload
```

API will be available at `http://localhost:8000`

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

**Example response:**

```json
{
  "label": "positive",
  "confidence": 0.97,
  "model": "student"
}
```

---

## Benchmarking

To measure API throughput (requests/second):

```bash
python scripts/benchmark.py
```

---

## Results

_To be filled in after Phase 3 training and evaluation._

|                    | Teacher (BERT-base) | Student (3-layer) |
| ------------------ | ------------------- | ----------------- |
| Accuracy           | -                   | -                 |
| Model size         | -                   | -                 |
| Inference speed    | -                   | -                 |
| Throughput (req/s) | -                   | -                 |

---

## Attention Layer Mapping

The feature-based KD uses the following Teacher → Student layer mapping (0-indexed):

| Student Layer | Teacher Layer |
| ------------- | ------------- |
| 0             | 2             |
| 1             | 6             |
| 2             | 10            |

---

## Git Workflow

- Never push directly to `main`
- Open a pull request when your phase is done — the other partner must review before merge
- Commit message prefixes: `feat:`, `fix:`, `docs:`, `refactor:`
- Branches: `student` (Partner B), `teacher` (Partner A)
