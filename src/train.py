import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data import get_tokenizer, get_datasets, get_dataloaders
from src.teacher import TeacherModel
from src.student import StudentModel
from src.losses import response_kd_loss, feature_kd_loss, combined_loss

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS           = 5
LEARNING_RATE    = 3e-4
WARMUP_RATIO     = 0.1
TEMPERATURE      = 4.0
ALPHA            = 0.5   # weight for CE loss
BETA             = 0.3   # weight for response KD loss
GAMMA            = 0.2   # weight for feature KD loss
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

TEACHER_CKPT     = "checkpoints/teacher_best.pt"
STUDENT_KD_CKPT  = "checkpoints/student_kd_best.pt"


# ── Helper: load the frozen teacher ───────────────────────────────────────────
def load_teacher(path, device):
    model = TeacherModel(num_labels=2).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    # Freeze all teacher parameters — it must never be updated during KD
    for param in model.parameters():
        param.requires_grad = False
    print(f"Teacher loaded from {path} "
          f"(val_acc={checkpoint['val_accuracy']:.4f})")
    return model


# ── Helper: evaluate on a dataloader ──────────────────────────────────────────
def evaluate(student, teacher, dataloader, device):
    student.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    total_ce, total_kd, total_feat, total_combined = 0., 0., 0., 0.

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            # Teacher forward (frozen)
            teacher_out = teacher(input_ids, attention_mask)

            # Student forward
            student_out = student(input_ids, attention_mask)

            # Losses
            ce   = criterion(student_out["logits"], labels)
            kd   = response_kd_loss(
                       student_out["logits"],
                       teacher_out["logits"],
                       TEMPERATURE,
                   )
            feat = feature_kd_loss(
                       student_out["attentions"],
                       teacher_out["attentions"],
                   )
            loss = combined_loss(ce, kd, feat, ALPHA, BETA, GAMMA)

            total_ce       += ce.item()
            total_kd       += kd.item()
            total_feat     += feat.item()
            total_combined += loss.item()

            preds    = student_out["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    n = len(dataloader)
    return {
        "accuracy":      correct / total,
        "loss_combined": total_combined / n,
        "loss_ce":       total_ce       / n,
        "loss_kd":       total_kd       / n,
        "loss_feat":     total_feat     / n,
    }


# ── Main training loop ─────────────────────────────────────────────────────────
def train():
    print(f"Using device: {DEVICE}")
    os.makedirs("checkpoints", exist_ok=True)

    # Data
    tokenizer   = get_tokenizer()
    datasets    = get_datasets(tokenizer)
    dataloaders = get_dataloaders(datasets)

    # Models
    teacher = load_teacher(TEACHER_CKPT, DEVICE)
    student = StudentModel(num_labels=2).to(DEVICE)

    # Optimizer + scheduler (higher LR than teacher — student trains from scratch)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps  = len(dataloaders["train"]) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        student.train()
        running = {"combined": 0., "ce": 0., "kd": 0., "feat": 0.}
        progress = tqdm(dataloaders["train"], desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in progress:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            # Teacher forward — no gradients, teacher is frozen
            with torch.no_grad():
                teacher_out = teacher(input_ids, attention_mask)

            # Student forward — gradients flow here
            student_out = student(input_ids, attention_mask)

            # Individual losses
            ce   = criterion(student_out["logits"], labels)
            kd   = response_kd_loss(
                       student_out["logits"],
                       teacher_out["logits"],
                       TEMPERATURE,
                   )
            feat = feature_kd_loss(
                       student_out["attentions"],
                       teacher_out["attentions"],
                   )

            # Combined loss
            loss = combined_loss(ce, kd, feat, ALPHA, BETA, GAMMA)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running["combined"] += loss.item()
            running["ce"]       += ce.item()
            running["kd"]       += kd.item()
            running["feat"]     += feat.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{ce.item():.4f}",
                kd=f"{kd.item():.4f}",
            )

        # ── Validation ─────────────────────────────────────────────────────────
        n          = len(dataloaders["train"])
        val_metrics = evaluate(student, teacher, dataloaders["validation"], DEVICE)

        print(
            f"\nEpoch {epoch} — "
            f"train_loss: {running['combined']/n:.4f} "
            f"(ce={running['ce']/n:.4f} "
            f"kd={running['kd']/n:.4f} "
            f"feat={running['feat']/n:.4f})\n"
            f"           val_acc: {val_metrics['accuracy']:.4f} | "
            f"val_loss: {val_metrics['loss_combined']:.4f}"
        )

        history.append({
            "epoch":    epoch,
            "val_acc":  val_metrics["accuracy"],
            "val_loss": val_metrics["loss_combined"],
        })

        # Save best checkpoint
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch":        epoch,
                "model_state":  student.state_dict(),
                "val_accuracy": val_metrics["accuracy"],
                "config": {
                    "num_labels":  2,
                    "hidden_size": 256,
                    "num_heads":   4,
                    "num_layers":  3,
                    "max_length":  128,
                    "vocab_size":  30522,
                },
            }, STUDENT_KD_CKPT)
            print(f"  ✓ Saved best checkpoint (val_acc={best_val_acc:.4f})")

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("\nLoading best checkpoint for final evaluation...")
    checkpoint = torch.load(STUDENT_KD_CKPT, map_location=DEVICE)
    student.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(student, teacher, dataloaders["test"], DEVICE)
    model_size   = os.path.getsize(STUDENT_KD_CKPT) / (1024 ** 2)

    print("\n── Student KD evaluation results ────────────────────────")
    print(f"  Test accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"  Model size      : {model_size:.1f} MB")
    print(f"  Loss breakdown  : ce={test_metrics['loss_ce']:.4f} | "
          f"kd={test_metrics['loss_kd']:.4f} | "
          f"feat={test_metrics['loss_feat']:.4f}")
    print("─────────────────────────────────────────────────────────")
    print("These numbers go in your report comparison table.")


if __name__ == "__main__":
    train()