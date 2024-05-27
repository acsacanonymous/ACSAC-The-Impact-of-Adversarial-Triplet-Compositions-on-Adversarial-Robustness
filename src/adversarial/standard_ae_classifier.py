import torch
import torch.nn as nn
import torch.nn.functional as F

lower_limit = torch.tensor(0, dtype=torch.float32).cuda()
upper_limit = torch.tensor(1, dtype=torch.float32).cuda()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters):
    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    for _ in range(attack_iters):
        with torch.enable_grad():
            output, _ = model(torch.clamp(X + delta, lower_limit, upper_limit))
            loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.grad.zero_()
    return torch.clamp(X + delta, lower_limit, upper_limit).detach()


def attack_pgd_targeted_opt(model, X, y_target, epsilon, alpha, attack_iters):
    criterion = nn.CrossEntropyLoss()
    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    opt = torch.optim.SGD([delta], lr=0.1, momentum=0.9)
    for _ in range(attack_iters):
        output, _ = model(torch.clamp(X + delta, lower_limit, upper_limit))
        loss = criterion(output, y_target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        delta.grad.zero_()
        torch.clamp(delta, -epsilon, epsilon)
    return torch.clamp(X + delta, lower_limit, upper_limit).detach(), delta


def attack_pgd_targeted(model, X, y_target, epsilon, alpha, attack_iters):
    criterion = nn.CrossEntropyLoss()
    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output, _ = model(torch.clamp(X + delta, lower_limit, upper_limit))
        loss = criterion(output, y_target)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta - alpha * torch.sign(grad), -epsilon, epsilon)
        delta.grad.zero_()
        torch.clamp(delta, -epsilon, epsilon)
    return torch.clamp(X + delta, lower_limit, upper_limit).detach(), delta


def attack_fgsm_targeted(model, X, y_target, epsilon):
    criterion = nn.CrossEntropyLoss()
    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    output, _ = model(torch.clamp(X + delta, lower_limit, upper_limit))
    loss = criterion(output, y_target)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = torch.clamp(delta - epsilon * torch.sign(grad), -epsilon, epsilon)
    delta.grad.zero_()
    return torch.clamp(X + delta, lower_limit, upper_limit).detach()