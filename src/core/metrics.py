import torch

def correct(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float()
    return accuracy.item()