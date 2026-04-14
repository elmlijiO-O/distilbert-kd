import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data import get_tokenizer, get_datasets, get_dataloaders
from src.student import StudentModel

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS           = 5
LEARNING_RATE    = 3e-4
WARMUP_RATIO     = 0.1
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH  = "checkpoints/student_baseline.pt"


def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss    = criterion(outputs["logits"], labels)

            total_loss += loss.item()
            preds       = outputs["logits"].argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return {
        "loss":     total_loss / len(dataloader),
        "accuracy": correct / total,
    }


def train():
    print(f"Using device: {DEVICE}")
    os.makedirs("checkpoints", exist_ok=True)

    tokenizer   = get_tokenizer()
    datasets    = get_datasets(tokenizer)
    dataloaders = get_dataloaders(datasets)

    model     = StudentModel(num_labels=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps  = len(dataloaders["train"]) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(dataloaders["train"], desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in progress:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(device=DEVICE)
            labels         = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss    = criterion(outputs["logits"], labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, dataloaders["validation"], DEVICE)
        print(
            f"Epoch {epoch} — "
            f"train_loss: {running_loss / len(dataloaders['train']):.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_accuracy": val_metrics["accuracy"],
                "config": {
                    "num_labels":  2,
                    "hidden_size": 256,
                    "num_heads":   4,
                    "num_layers":  3,
                    "max_length":  128,
                    "vocab_size":  30522,
                },
            }, CHECKPOINT_PATH)
            print(f"  ✓ Saved best checkpoint (val_acc={best_val_acc:.4f})")

    # Final test evaluation
    print("\nLoading best checkpoint for final evaluation...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(model, dataloaders["test"], DEVICE)
    model_size   = os.path.getsize(CHECKPOINT_PATH) / (1024 ** 2)

    print("\n── Student baseline evaluation results ──────────────────")
    print(f"  Test accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Model size    : {model_size:.1f} MB")
    print("─────────────────────────────────────────────────────────")
    print("These numbers go in your report comparison table.")


if __name__ == "__main__":
    train()