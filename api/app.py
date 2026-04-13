import torch
import torch.nn.functional as F
from fastapi import FastAPI
from src.student import StudentModel
from src.data import get_tokenizer
from api.schemas import PredictRequest, PredictResponse

app = FastAPI()

# Load once on startup
tokenizer = get_tokenizer()
model = StudentModel()

checkpoint = torch.load("checkpoints/student_best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()


@app.post("/predict", response_model=PredictResponse)

def predict(request: PredictRequest):
    # Step 1: tokenize
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # Step 2: run through model
    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # Step 3: logits → probabilities → label
    probs = F.softmax(output["logits"], dim=1)
    confidence, predicted_class = probs.max(dim=1)

    label = "positive" if predicted_class.item() == 1 else "negative"

    return PredictResponse(
        label=label,
        confidence=round(confidence.item(), 4),
        model="student"
    )