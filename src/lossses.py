import torch
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
    raise NotImplementedError  # YOU implement this

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