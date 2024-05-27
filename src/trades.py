import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from core.utils import track_bn_stats
from core.metrics import correct
from adversarial.standard_ae_classifier import attack_pgd


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, classifier_ae=False):
    """
    TRADES training (Zhang et al, 2019).
    """

    criterion_ce = CrossEntropyLoss(reduction='mean')
    criterion_kl = nn.KLDivLoss(reduction='sum')

    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    outputs, _ = model(x_natural)
    p_natural = F.softmax(outputs, dim=1)

    if classifier_ae:
        x_adv = attack_pgd(model, x_natural, y, epsilon, step_size, perturb_steps)
    else:
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                outputs, _ = model(x_adv)
                loss_kl = criterion_kl(F.log_softmax(outputs, dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    track_bn_stats(model, True)

    x_adv_clone = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    # calculate robust loss
    logits_natural, _ = model(x_natural)
    logits_adv, _ = model(x_adv_clone)
    loss_natural = criterion_ce(logits_natural, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust

    batch_metrics = {'loss': loss.item(), 'clean_correct': correct(y, logits_natural.detach()),
                     'adversarial_correct': correct(y, logits_adv.detach())}

    return loss, batch_metrics, torch.clamp(x_adv, 0.0, 1.0)