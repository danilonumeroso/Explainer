from torch.nn import functional as F

def tox21_loss_explainer(p1, p2):
    p1 = F.softmax(p1, dim=-1).detach().squeeze()
    return p1[1 - p2]

def esol_loss_explainer(p1, p2):
    return F.l1_loss(p1, p2)


explain_loss = {
    'tox21': tox21_loss_explainer,
    'esol': esol_loss_explainer
}
