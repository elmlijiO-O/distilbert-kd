import torch
import torch.nn as nn
import torch.nn.functional as F


def response_kd_loss(student_logits, teacher_logits, temperature=4.0):
    """
    KL divergence between softened teacher and student distributions.

    Args:
        student_logits  : FloatTensor [batch, num_labels]
        teacher_logits  : FloatTensor [batch, num_labels]
        temperature     : float, softening temperature T

    Returns:
        scalar loss
    """
    # Step 1 — soften both distributions with temperature
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits    / temperature, dim=-1)

    # Step 2 — KL divergence: measures how far student is from teacher
    # F.kl_div expects (log_probs, probs) — reduction='batchmean' is the correct setting
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

    # Step 3 — rescale by T² to restore gradient magnitude
    return kl * (temperature ** 2)

def feature_kd_loss(student_attentions, teacher_attentions):
    """
    MSE between student and teacher attention maps.
    Teacher has 12 layers, student has 3 — use every 4th teacher layer.
    Mapping: student layer i ← teacher layer (i+1)*4 - 1  (i.e. 3, 7, 11)

    Args:
        student_attentions : list of 3 FloatTensor [batch, heads, seq, seq]
        teacher_attentions : list of 12 FloatTensor [batch, heads, seq, seq]

    Returns:
        scalar loss

    """
    mapped_teacher = [teacher_attentions[2], teacher_attentions[6], teacher_attentions[10]]
    total_loss = 0

    for i in range(3):
        s =  student_attentions[i]
        t = mapped_teacher[i]

        s = s.mean(dim=1)
        t = t.mean(dim=1)   


        # Compute MSE for this layer
        total_loss += F.mse_loss(s, t)
    return total_loss / 3  # average over the 3 layers
    raise NotImplementedError  # PARTNER implements this

def combined_loss(ce_loss, kd_loss, feat_loss, alpha=0.5, beta=0.3, gamma=0.2):
    """
    L = alpha * ce_loss + beta * kd_loss + gamma * feat_loss

    Args:
        ce_loss   : scalar, cross-entropy on hard labels
        kd_loss   : scalar, response-based KD loss
        feat_loss : scalar, feature-based KD loss
        alpha, beta, gamma : floats, must sum to 1.0

    Returns:
        scalar combined loss
    """
    return alpha * ce_loss + beta * kd_loss + gamma * feat_loss