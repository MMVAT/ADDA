import torch
import torch.nn as nn
import torch.nn.functional as F


class Uncertainty_Guided_Mutual_Alignment_Loss(nn.Module):
    def __init__(self, eta=1.0):
        super().__init__()
        self.eta = eta
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, target, global_prob, a_prob, v_prob, a_uct, v_uct, global_uct, batch_idx):
        eps = 1e-8
        a_uct = torch.clamp(a_uct, min=eps)
        v_uct = torch.clamp(v_uct, min=eps)
        global_uct = torch.clamp(global_uct, min=eps)

        loss_audio = (a_uct * F.kl_div(torch.log(global_prob + eps), a_prob.detach())).mean()
        loss_visual = (v_uct * F.kl_div(torch.log(global_prob + eps), v_prob.detach())).mean()
        loss_global = (global_uct * (F.kl_div(torch.log(a_prob + eps), global_prob.detach()) + F.kl_div(torch.log(v_prob + eps), global_prob.detach()))).mean()

        total_loss = (loss_audio + loss_visual + loss_global) * self.eta
        return total_loss
