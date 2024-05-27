import logging
import os
import torch
import torch.nn.functional as F
from torch import nn

upper_limit = torch.tensor(1, dtype=torch.float32).cuda()
lower_limit = torch.tensor(0, dtype=torch.float32).cuda()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd_correct_only(model, X, y, epsilon, alpha, attack_iters, restarts, device):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    output, _ = model(X)
    index = torch.where(output.max(1)[1] == y)
    if len(index[0]) == 0:
        return max_delta, 0
    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        delta.uniform_(-epsilon, epsilon)
        # delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output, _ = model(clamp(X + delta, lower_limit, upper_limit))
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            # d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        with torch.no_grad():
            output, _ = model(clamp(X + delta, lower_limit, upper_limit))
            all_loss = F.cross_entropy(output, y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta, len(index[0])


def evaluate_pgd_correct_only(test_loader, model, restarts, attack_iters, epsilon, alpha, device):
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        pgd_delta, init_acc = attack_pgd_correct_only(model, X, y, epsilon, alpha, attack_iters, restarts, device)
        with torch.no_grad():
            output, _ = model(clamp(X + pgd_delta, lower_limit, upper_limit))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * init_acc
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += init_acc
    print(f"PGD Loss: {pgd_loss / n} \t PGD Acc: {pgd_acc / n} \t on correctly classified examples")
    logging.info(f"PGD Loss: {pgd_loss / n} \t PGD Acc: {pgd_acc / n} \t on correctly classified examples")
    return pgd_acc / n

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, device):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        delta.uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output, _ = model(clamp(X + delta, lower_limit, upper_limit))
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.grad.zero_()
        with torch.no_grad():
            output, _ = model(clamp(X + delta, lower_limit, upper_limit))
            all_loss = F.cross_entropy(output, y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, restarts, attack_iters, epsilon, alpha, device):
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, device)
        with torch.no_grad():
            output, _ = model(clamp(X + pgd_delta, lower_limit, upper_limit))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    print(f"PGD Loss: {pgd_loss / n} \t PGD Acc: {pgd_acc / n}")
    logging.info(f"PGD Loss: {pgd_loss / n} \t PGD Acc: {pgd_acc / n}")
    return pgd_acc / n