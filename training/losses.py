import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017): FL = -(1 - p_t)^gamma * log(p_t)
    ลดน้ำหนักรูปที่ทายถูกง่าย ๆ แล้วเน้นรูปที่ยาก — อีกวิธีจัดการ class imbalance แทนการถ่วงตามความถี่
    weight (alpha) = None -> ใช้ focal อย่างเดียว เพื่อแยกผลจาก class weight
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()
