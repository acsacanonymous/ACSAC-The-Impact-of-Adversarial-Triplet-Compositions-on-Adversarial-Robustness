# https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
# https://omoindrot.github.io/triplet-loss
import torch
import torch.nn.functional as F


def mine_adversarial_triplets_all_atn(labels_clean, labels_adv, embedding_clean, embedding_adv, margin):
    pdist_adv_clean = _pairwise_distance_adv_clean(embedding_clean, embedding_adv)

    anchor_positive_dist = pdist_adv_clean.unsqueeze(2)
    anchor_negative_dist = pdist_adv_clean.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask_atn(labels_clean, labels_adv)

    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = F.relu(triplet_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, num_positive_triplets


def _get_triplet_mask_atn(labels_clean, labels_adv):
    label_equal_adv = labels_adv.unsqueeze(1) == labels_clean.unsqueeze(0)
    i_adv_equal_j = label_equal_adv.unsqueeze(2)
    i_adv_equal_k = label_equal_adv.unsqueeze(1)

    valid_labels_adv = i_adv_equal_k & ~ i_adv_equal_j

    label_equal_clean = labels_clean.unsqueeze(1) == labels_clean.unsqueeze(0)
    i_equal_j = label_equal_clean.unsqueeze(2)
    i_equal_k = label_equal_clean.unsqueeze(1)

    valid_labels_clean = ~i_equal_k & i_equal_j

    valid_labels = valid_labels_adv & valid_labels_clean

    indices_equal = torch.eye(labels_clean.size(0), device=labels_clean.device).bool()
    indices_not_equal = ~indices_equal
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    i_not_equal_j = indices_not_equal.unsqueeze(2)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    return valid_labels & distinct_indices


# --------------------------------- NTA -------------------------------------------------


def mine_adversarial_triplets_all_nta(labels_clean, labels_adv, embedding_clean, embedding_adv, margin):
    pdist_clean = _pairwise_distance_clean(embedding_clean)
    pdist_clean_adv = _pairwise_distance_clean_adv(embedding_clean, embedding_adv)

    anchor_positive_dist = pdist_clean.unsqueeze(2)
    anchor_negative_dist = pdist_clean_adv.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask_nta(labels_clean, labels_adv)

    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = F.relu(triplet_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, num_positive_triplets


def _get_triplet_mask_nta(labels_clean, labels_adv):
    indices_equal = torch.eye(labels_clean.size(0), device=labels_clean.device).bool()
    indices_not_equal = ~indices_equal
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    i_not_equal_j = indices_not_equal.unsqueeze(2)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal_adv = labels_clean.unsqueeze(1) == labels_adv.unsqueeze(0)
    i_equal_adv_k = label_equal_adv.unsqueeze(1)

    valid_labels_adv = i_equal_adv_k

    label_equal_clean = labels_clean.unsqueeze(0) == labels_clean.unsqueeze(1)
    i_equal_j = label_equal_clean.unsqueeze(2)
    i_equal_k = label_equal_clean.unsqueeze(1)

    valid_labels_clean = i_equal_j & ~ i_equal_k

    valid_labels = valid_labels_adv & valid_labels_clean

    return valid_labels & distinct_indices


# ---------------- BATCH HARD ATN ---------------------------------------------------


def mine_adversarial_triplets_batch_hard_atn(labels_clean, labels_adv, embeddings_clean, embedding_adv, margin):
    pairwise_dist = _pairwise_distance_adv_clean(embeddings_clean, embedding_adv)
    mask_anchor_positive = _get_anchor_positive_triplet_mask_atn(labels_clean, labels_adv)

    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    mask_anchor_negative = _get_anchor_negative_triplet_mask_atn(labels_clean, labels_adv).float()

    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    eligible_mask = mask_anchor_negative.max(1, keepdim=True)[0] == 1

    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    hardest_positive_dist = hardest_positive_dist[eligible_mask]
    hardest_negative_dist = hardest_negative_dist[eligible_mask]

    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = tl[tl > 1e-16]
    num_positive_triplets = tl.size(0)
    if num_positive_triplets == 0:
        return torch.tensor(0, dtype=torch.float32), 0

    triplet_loss = tl.mean()

    return triplet_loss, num_positive_triplets


def _get_anchor_positive_triplet_mask_atn(labels_clean, labels_adv):
    indices_equal = torch.eye(labels_clean.size(0), device=labels_clean.device).bool()
    indices_not_equal = ~indices_equal

    labels_equal = labels_clean.unsqueeze(1) == labels_clean.unsqueeze(0)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask_atn(labels_clean, labels_adv):
    labels_adv_equal = labels_adv.unsqueeze(1) == labels_clean.unsqueeze(0)

    return labels_adv_equal


# ---------------- BATCH HARD NTA ---------------------------------------------------
# FIXED

def mine_adversarial_triplets_batch_hard_nta(labels_clean, labels_adv, embeddings_clean, embeddings_adv, margin):
    dist_anchor_positive = _pairwise_distance_clean(embeddings_clean)
    mask_anchor_positive = _get_anchor_positive_triplet_mask_nta(labels_clean).float()

    dist_anchor_positive = mask_anchor_positive * dist_anchor_positive

    hardest_positive_dist, _ = dist_anchor_positive.max(1, keepdim=True)

    dist_anchor_negative = _pairwise_distance_clean_adv(embeddings_clean, embeddings_adv)
    mask_anchor_negative = _get_anchor_negative_triplet_mask_nta(labels_clean, labels_adv).float()

    # dist_anchor_negative = mask_anchor_negative * dist_anchor_negative

    max_anchor_negative_dist, _ = dist_anchor_negative.max(1, keepdim=True)
    # print(max_anchor_negative_dist.min(0))

    anchor_negative_dist = dist_anchor_negative + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    eligible_mask = mask_anchor_negative.max(1, keepdim=True)[0] == 1

    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    hardest_positive_dist = hardest_positive_dist[eligible_mask]
    hardest_negative_dist = hardest_negative_dist[eligible_mask]

    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = tl[tl > 1e-16]
    num_positive_triplets = tl.size(0)
    if num_positive_triplets == 0:
        return torch.tensor(0, dtype=torch.float32), 0

    triplet_loss = tl.mean()

    return triplet_loss, num_positive_triplets


def _get_anchor_positive_triplet_mask_nta(labels_clean):
    indices_equal = torch.eye(labels_clean.size(0), device=labels_clean.device).bool()
    indices_not_equal = ~indices_equal

    labels_equal = labels_clean.unsqueeze(1) == labels_clean.unsqueeze(0)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask_nta(labels_clean, labels_adv):

    labels_adv_equal = labels_clean.unsqueeze(1) == labels_adv.unsqueeze(0)

    return labels_adv_equal


# ------------ PDISTS ----------------------------------------------


def _pairwise_distance_clean(embeddings_clean):
    # return torch.cdist(embeddings, embeddings, compute_mode='donot_use_mm_for_euclid_dist')
    return 1 - F.cosine_similarity(embeddings_clean[:,None,:], embeddings_clean[None,:,:], dim=-1)


def _pairwise_distance_clean_adv(embeddings_clean, embedding_adv):
    # return torch.cdist(embeddings_clean, embedding_adv, compute_mode='donot_use_mm_for_euclid_dist')
    return 1 - F.cosine_similarity(embeddings_clean[:,None,:], embedding_adv[None,:,:], dim=-1)


def _pairwise_distance_adv_clean(embeddings_clean, embedding_adv):
    # return torch.cdist(embedding_adv, embeddings_clean, compute_mode='donot_use_mm_for_euclid_dist')
    return 1 - F.cosine_similarity(embedding_adv[:,None,:], embeddings_clean[None,:,:], dim=-1)


# --------- PDIST original -------------------------------------------

def _pairwise_distances_clean_orig(embeddings, squared=False):
    dot_product = torch.matmul(embeddings, embeddings.t())

    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)

    distances[distances < 0] = 0

    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances


def _pairwise_distances_clean_adv_orig(embeddings_clean, embeddings_adv, squared=False):
    dot_product = torch.matmul(embeddings_clean, embeddings_adv.t())

    dot_product_clean = torch.matmul(embeddings_clean, embeddings_clean.t())

    dot_product_adv = torch.matmul(embeddings_adv, embeddings_adv.t())

    square_norm_clean = torch.diag(dot_product_clean)

    square_norm_adv = torch.diag(dot_product_adv)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm_clean.unsqueeze(1) - 2.0 * dot_product + square_norm_adv.unsqueeze(0)

    distances[distances < 0] = 0

    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances


def _pairwise_distances_adv_clean_orig(embeddings_clean, embeddings_adv, squared=False):
    dot_product = torch.matmul(embeddings_adv, embeddings_clean.t())

    dot_product_clean = torch.matmul(embeddings_clean, embeddings_clean.t())

    dot_product_adv = torch.matmul(embeddings_adv, embeddings_adv.t())

    square_norm_clean = torch.diag(dot_product_clean)

    square_norm_adv = torch.diag(dot_product_adv)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm_adv.unsqueeze(1) - 2.0 * dot_product + square_norm_clean.unsqueeze(0)

    distances[distances < 0] = 0

    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * torch.sqrt(distances)

    return distances


def count_adversarial_triplets_all_nta(labels_clean, labels_adv, embedding_clean, embedding_adv, margin):
    margin_tensor = torch.tensor(margin, dtype=torch.float32, device=embedding_clean.device)
    zero_tensor = torch.tensor(0, dtype=torch.float32, device=embedding_clean.device)

    pdist_clean = _pairwise_distance_clean(embedding_clean)
    pdist_clean_adv = _pairwise_distance_clean_adv(embedding_clean, embedding_adv)

    anchor_positive_dist = pdist_clean.unsqueeze(2)
    anchor_negative_dist = pdist_clean_adv.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    valid_mask = _get_triplet_mask_nta(labels_clean, labels_adv)
    invalid_mask = ~(torch.clone(valid_mask))
    valid_mask = valid_mask.float().to(embedding_clean.device)
    invalid_mask = invalid_mask.float().to(embedding_clean.device)

    low_values = torch.full(invalid_mask.size(), -200).to(embedding_clean.device)
    high_values = torch.full(invalid_mask.size(), 200).to(embedding_clean.device)

    for_soft_semihard = valid_mask * triplet_loss + invalid_mask * high_values
    for_hard = valid_mask * triplet_loss + invalid_mask * low_values

    soft_mask = torch.le(for_soft_semihard, zero_tensor)
    semihard_mask = torch.logical_and(torch.gt(for_soft_semihard, zero_tensor),
                                      torch.le(for_soft_semihard, margin_tensor))
    hard_mask = torch.gt(for_hard, margin_tensor)
    softs = soft_mask.sum().item()
    semihards = semihard_mask.sum().item()
    hards = hard_mask.sum().item()
    return softs, semihards, hards


def count_adversarial_triplets_all_atn(labels_clean, labels_adv, embedding_clean, embedding_adv, margin):
    pdist_adv_clean = _pairwise_distance_adv_clean(embedding_clean, embedding_adv)

    margin_tensor = torch.tensor(margin, dtype=torch.float32, device=embedding_clean.device)
    zero_tensor = torch.tensor(0, dtype=torch.float32, device=embedding_clean.device)

    anchor_positive_dist = pdist_adv_clean.unsqueeze(2)
    anchor_negative_dist = pdist_adv_clean.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    valid_mask = _get_triplet_mask_atn(labels_clean, labels_adv)
    invalid_mask = ~(torch.clone(valid_mask))
    valid_mask = valid_mask.float().to(embedding_clean.device)
    invalid_mask = invalid_mask.float().to(embedding_clean.device)

    low_values = torch.full(invalid_mask.size(), -200).to(embedding_clean.device)
    high_values = torch.full(invalid_mask.size(), 200).to(embedding_clean.device)

    for_soft_semihard = valid_mask * triplet_loss + invalid_mask * high_values
    for_hard = valid_mask * triplet_loss + invalid_mask * low_values

    soft_mask = torch.le(for_soft_semihard, zero_tensor)
    semihard_mask = torch.logical_and(torch.gt(for_soft_semihard, zero_tensor),
                                      torch.le(for_soft_semihard, margin_tensor))
    hard_mask = torch.gt(for_hard, margin_tensor)
    softs = soft_mask.sum().item()
    semihards = semihard_mask.sum().item()
    hards = hard_mask.sum().item()
    return softs, semihards, hards


# ---------- Pairwise Cosine similarity ------------------------
# Credit to Torchmetrics https://lightning.ai/docs/torchmetrics/stable/pairwise/cosine_similarity.html

def _pairwise_cosine_similarity_update( x, y=None, zero_diagonal=None):
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)

    norm = torch.norm(x, p=2, dim=1)
    x = x / norm.unsqueeze(1)
    norm = torch.norm(y, p=2, dim=1)
    y = y / norm.unsqueeze(1)

    distance = _safe_matmul(x, y)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance


def pairwise_cosine_similarity(
    x, y=None, reduction=None, zero_diagonal=None):

    distance = _pairwise_cosine_similarity_update(x, y, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)

def _safe_matmul(x, y):
    """Safe calculation of matrix multiplication.

    If input is float16, will cast to float32 for computation and back again.

    """
    if x.dtype == torch.float16 or y.dtype == torch.float16:
        return (x.float() @ y.T.float()).half()
    return x @ y.T

def _check_input(
    x, y=None, zero_diagonal=None):

    if x.ndim != 2:
        raise ValueError(f"Expected argument `x` to be a 2D tensor of shape `[N, d]` but got {x.shape}")

    if y is not None:
        if y.ndim != 2 or y.shape[1] != x.shape[1]:
            raise ValueError(
                "Expected argument `y` to be a 2D tensor of shape `[M, d]` where"
                " `d` should be same as the last dimension of `x`"
            )
        zero_diagonal = False if zero_diagonal is None else zero_diagonal
    else:
        y = x.clone()
        zero_diagonal = True if zero_diagonal is None else zero_diagonal
    return x, y, zero_diagonal


def _reduce_distance_matrix(distmat, reduction=None):

    if reduction == "mean":
        return distmat.mean(dim=-1)
    if reduction == "sum":
        return distmat.sum(dim=-1)
    if reduction is None or reduction == "none":
        return distmat
    raise ValueError(f"Expected reduction to be one of `['mean', 'sum', None]` but got {reduction}")