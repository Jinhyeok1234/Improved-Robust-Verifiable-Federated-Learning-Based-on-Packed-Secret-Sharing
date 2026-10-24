"""
Poisoning attacks of the robustness experiments.

Data poisoning (label flipping, and the data side of the Scaling attack) is set up once, in
the malicious clients' data loaders (poisoning_config). The model-poisoning attacks act on
the local update: malicious clients train normally, then their `client.g_i` is REPLACED with
a crafted vector before any aggregation reads it (apply_model_poisoning, once per round,
right after every client's training).

Threat model matches the FLTrust paper's evaluation: FULL KNOWLEDGE -- the attacker
sees the benign clients' current-round updates (and, for the adaptive attack, the
server root update g_0 and the aggregation rule).

Attacks:
  - "trim_fltrust": Fang et al. (2020) "Trim attack" (crafted against coordinate-wise
    trimmed mean), evaluated as an existing attack in the FLTrust paper. FLTrust's
    released formulation.
  - "adaptive_fltrust": the FLTrust paper's OWN adaptive attack (Cao et al., NDSS'21,
    Section V, Algorithm 3). Zeroth-order coordinate ascent on each malicious client's
    unit update vector e'_i to maximize s^T(g - g'), where g/g' are the FLTrust
    aggregates before/after attack and s = sign(g). Initialized from the Trim attack.
  - "scaling_fltrust": the Scaling (backdoor) attack. TARGETED: malicious clients train on
    trigger-stamped, target-relabeled data (poisoning_config) and submit their update scaled by
    lambda = n. Evaluated by the attack success rate on trigger-embedded inputs, not by test
    error. FLTrust normalizes every g_i to ||g_S||, so the scaling is normalized away there; in
    RVFL the scaled perpendicular vector fails the norm check and is rejected (by the
    protocol's verification, and by rvfl.baselines.rvfl_plaintext_aggregate in the clear).
  - "krum_fltrust": Fang et al. (2020) Krum attack (full knowledge), crafted against the Krum
    aggregation rule and applied to FLTrust (transferability, as the FLTrust paper does). All
    c malicious send the SAME update g'_1 = -lambda*s, s = sign(sum of honest updates); lambda
    is the largest value (upper-bounded by Fang Thm 1, then binary-searched) for which Krum
    still SELECTS a malicious update. Working in update space, w'_1 = w_Re - lambda*s maps to
    g'_1 = -lambda*s and D(w_i,w_Re) = ||g_i||.
"""
from __future__ import annotations

import math
from typing import List

import torch


DATA_POISON_ATTACKS = ("label_flip_fltrust", "scaling_fltrust")
# Scaling is not here: its data side is a data poisoning, and its lambda scaling is applied
# where the update is submitted (Client.submission_scale in the protocol, BREA included;
# rvfl.baselines in the clear), so it needs no per-round g_i overwrite.
MODEL_POISON_ATTACKS = ("trim_fltrust", "adaptive_fltrust", "krum_fltrust")
ATTACKS = DATA_POISON_ATTACKS + MODEL_POISON_ATTACKS

# Scaling attack: CIFAR-10 target class "bird" (2); half of each malicious client's samples carry
# the trigger.
SCALING_TARGET = 2
SCALING_POISON_RATE = 0.5


def malicious_ids(n_clients: int, rate: float) -> List[int]:
    """The malicious clients: the first floor(n_clients * rate) ids."""
    return list(range(int(math.floor(n_clients * rate))))


def submission_scale(attack, n_clients: int) -> int:
    """The factor lambda a malicious client multiplies its submission by: n under the Scaling
    attack, 1 otherwise."""
    return n_clients if attack == "scaling_fltrust" else 1


def poisoning_config(attack, num_classes: int = 10):
    """The data poisoning of a malicious client for `attack`, or None if the attack has none."""
    from .data import PoisoningConfig
    if attack == "label_flip_fltrust":
        return PoisoningConfig(poison_type="label_flip_fltrust", poison_rate=1.0, num_classes=num_classes)
    if attack == "scaling_fltrust":
        return PoisoningConfig(poison_type="backdoor", target_label=SCALING_TARGET,
                               poison_rate=SCALING_POISON_RATE, num_classes=num_classes)
    return None

# FLTrust paper hyperparameters for the adaptive attack (p.10, CIFAR-10 / all datasets):
#   sigma^2 = 0.5, gamma = 0.005, eta = 0.01, V = Q = 10.
ADAPT_SIGMA2 = 0.5      # variance of the ZO perturbation u ~ N(0, sigma^2 I)
ADAPT_GAMMA  = 0.005    # ZO smoothing parameter
ADAPT_ETA    = 0.01     # gradient-ascent step size
ADAPT_V      = 10       # outer sweeps over the malicious clients
ADAPT_Q      = 10       # inner ZO iterations per client

# Fang Krum attack: the other c-1 malicious are placed within KRUM_EPS of w'_1 (NOT identical --
# Fang uses eps=0.01 so the clique isn't trivially detectable as duplicates).
KRUM_EPS = 0.01


def _stack(vectors) -> torch.Tensor:
    return torch.stack([v.detach().reshape(-1).float() for v in vectors], dim=0)


def _benign_stack(clients, malicious_ids: List[int]) -> torch.Tensor:
    mal = set(malicious_ids)
    benign = [c.g_i for i, c in enumerate(clients) if i not in mal]
    if not benign:
        raise ValueError("model-poisoning needs >=1 benign client to craft against.")
    return _stack(benign)


def _trim_vectors(benign: torch.Tensor, m: int) -> torch.Tensor:
    """Return (m, d) Fang/FLTrust Trim-attack vectors crafted from the benign updates.

    s = sign(sum benign); directed = min (s>0) / max (s<0); each malicious coord is
    pushed just past that boundary opposite the aggregate by a per-coord factor U(1,2).
    """
    grad_sum = benign.sum(dim=0)
    direction = torch.sign(grad_sum)
    dmax = benign.max(dim=0).values
    dmin = benign.min(dim=0).values
    directed = torch.where(direction > 0, dmin, dmax)
    same = (direction * directed) > 0
    out = []
    for _ in range(m):
        rand = 1.0 + torch.rand_like(directed)                 # per-coord U(1,2)
        out.append(torch.where(same, directed / rand, directed * rand))
    return torch.stack(out, dim=0)                             # (m, d)


def trim_attack(clients, malicious_ids: List[int]) -> None:
    """Fang et al. Trim attack (full knowledge): overwrite malicious g_i in place."""
    if not malicious_ids:
        return
    benign = _benign_stack(clients, malicious_ids)
    mal_vecs = _trim_vectors(benign, len(malicious_ids))
    for k, cid in enumerate(malicious_ids):
        ref = clients[cid].g_i
        clients[cid].g_i = mal_vecs[k].reshape(ref.shape).to(ref.dtype).to(ref.device)


def adaptive_attack(clients, malicious_ids: List[int], g_server: torch.Tensor,
                    sigma2: float = ADAPT_SIGMA2, gamma: float = ADAPT_GAMMA,
                    eta: float = ADAPT_ETA, V: int = ADAPT_V, Q: int = ADAPT_Q) -> None:
    """FLTrust adaptive attack (Cao et al., Algorithm 3), full knowledge.

    Optimizes each malicious client's unit update vector e'_i by zeroth-order coordinate
    ascent to maximize h = s^T(g - g'):
      - g  = FLTrust aggregate of the honest updates (before attack), s = sign(g)   (Eq 7)
      - g' = FLTrust aggregate with malicious e'_i substituted                       (Eq 8)
      - grad estimated by  [h(e'_i + gamma u) - h(e'_i)]/gamma * u,  u ~ N(0, sigma2 I) (Eq 11)
      - e'_i projected back to the unit sphere each step; initialized from Trim.
    Final malicious update:  g'_i = ||g_0|| * e'_i.

    Efficiency: while optimizing client i, only its own term changes, so (a) each h-eval is a
    couple of dot products over d rather than a full re-aggregation, and (b) the malicious block's
    weighted sum is maintained incrementally (subtract i's old term, add its new one) instead of
    re-summed for every i -- making the per-client setup O(d), not O(m*d), so the whole attack is
    O(V*m*d) rather than O(V*m^2*d).
    """
    mal = list(malicious_ids)
    m = len(mal)
    if m == 0:
        return
    n = len(clients)
    ref0 = clients[mal[0]].g_i
    device, dtype = ref0.device, ref0.dtype
    g0 = g_server.detach().reshape(-1).float().to(device)
    g0_norm = torch.norm(g0).clamp_min(1e-12)
    e0 = g0 / g0_norm
    d = g0.numel()

    # unit vectors e_i and cosines c_i for ALL clients, from the HONEST updates.
    G = _stack([clients[i].g_i for i in range(n)]).to(device)          # (n, d)
    norms = torch.norm(G, dim=1, keepdim=True).clamp_min(1e-12)
    E = G / norms                                                     # (n, d)
    C = E @ e0                                                        # (n,)
    reluC = torch.relu(C)

    # before-attack aggregate g (all honest, Eq 7) and s = sign(g)
    g_clean = g0_norm * (reluC.unsqueeze(1) * E).sum(0) / reluC.sum().clamp_min(1e-12)
    s = torch.sign(g_clean)                                          # (d,)
    sTg = torch.dot(s, g_clean)                                      # constant

    mal_set = set(mal)
    benign_idx = [i for i in range(n) if i not in mal_set]
    reluC_benign_sum = reluC[benign_idx].sum()
    Vec_benign = (reluC[benign_idx].unsqueeze(1) * E[benign_idx]).sum(0)   # (d,) sum ReLU(c_j) e_j

    # initialize malicious unit vectors from the Trim attack
    Ep = _trim_vectors(_stack([clients[i].g_i for i in benign_idx]).to(device), m)   # (m, d)
    Ep = Ep / torch.norm(Ep, dim=1, keepdim=True).clamp_min(1e-12)
    relu_mal = torch.relu(Ep @ e0)                                   # (m,)

    # Running aggregates over the malicious block, so each client's O(d) setup replaces the O(m*d)
    # re-sum: Vec_mal = sum_j ReLU(c'_j) e'_j and relu_mal_sum = sum_j ReLU(c'_j). Only client k's
    # term changes during its own Q steps, so after optimizing k we subtract its old term and add the
    # new one (O(d)). This is algebraically identical to recomputing the full sum each k -- "total
    # minus k" == "sum over j != k" -- it just drops the redundant re-summation (O(V m^2 d) -> O(V m d)).
    Vec_mal = (relu_mal.unsqueeze(1) * Ep).sum(0)                    # (d,)
    relu_mal_sum = relu_mal.sum()                                    # scalar

    for _v in range(V):
        for k in range(m):
            # k's current contribution to the running aggregates (captured before its Q steps;
            # relu_mal[k] is a view, so clone it before the loop overwrites it)
            old_term = relu_mal[k] * Ep[k]
            old_relu_k = relu_mal[k].clone()

            # "rest" = everything except malicious client k (constant during its Q steps)
            rest_relu = relu_mal_sum - old_relu_k
            D_rest = reluC_benign_sum + rest_relu
            Vec_rest = Vec_benign + Vec_mal - old_term
            sT_Vec_rest = torch.dot(s, Vec_rest)

            def h_of(ei: torch.Tensor) -> torch.Tensor:
                r = torch.relu(torch.dot(ei, e0))
                D = (D_rest + r).clamp_min(1e-12)
                sT_gprime = g0_norm * (sT_Vec_rest + r * torch.dot(s, ei)) / D
                return sTg - sT_gprime                              # h = s^T(g - g')

            for _t in range(Q):
                ei = Ep[k]
                u = torch.randn(d, device=device) * math.sqrt(sigma2)
                grad = ((h_of(ei + gamma * u) - h_of(ei)) / gamma) * u
                ei = ei + eta * grad
                ei = ei / torch.norm(ei).clamp_min(1e-12)           # project to unit sphere
                Ep[k] = ei
                relu_mal[k] = torch.relu(torch.dot(ei, e0))

            # k's term changed -> update the running aggregates in O(d) (subtract old, add new)
            Vec_mal = Vec_mal - old_term + relu_mal[k] * Ep[k]
            relu_mal_sum = relu_mal_sum - old_relu_k + relu_mal[k]

    # final malicious updates: g'_i = ||g_0|| * e'_i
    for k, cid in enumerate(mal):
        ref = clients[cid].g_i
        clients[cid].g_i = (g0_norm * Ep[k]).reshape(ref.shape).to(dtype).to(device)


def _krum_selects_malicious(g_mal: torch.Tensor, B: torch.Tensor, Bsq: torch.Tensor,
                            c: int, knn: int) -> bool:
    """True iff Krum (score = sum of squared dist to the knn nearest, argmin selected) picks
    one of the c IDENTICAL malicious updates g_mal, given benign updates B (with precomputed
    benign-benign squared-dist Bsq)."""
    n_b = B.shape[0]
    n = c + n_b
    mb = ((B - g_mal) ** 2).sum(dim=1)                # (n_b,) squared dist malicious<->benign
    Dsq = torch.zeros(n, n, device=B.device, dtype=B.dtype)
    Dsq[:c, c:] = mb.unsqueeze(0).expand(c, n_b)      # malicious rows -> benign cols
    Dsq[c:, :c] = mb.unsqueeze(1).expand(n_b, c)      # benign rows -> malicious cols
    Dsq[c:, c:] = Bsq                                 # benign-benign (malicious-malicious = 0)
    idx = torch.arange(n, device=B.device)
    Dsq[idx, idx] = float("inf")                      # exclude self from neighbors
    scores = torch.sort(Dsq, dim=1).values[:, :knn].sum(dim=1)   # Krum score per update
    return int(torch.argmin(scores)) < c              # a malicious index is 0..c-1


def krum_attack(clients, malicious_ids: List[int]) -> None:
    """Fang et al. Krum attack (full knowledge): all c malicious send g'_1 = -lambda*s, with
    lambda the largest value for which Krum selects a malicious update (Thm 1 bound, then
    binary search). Applied here to FLTrust (transferability)."""
    mal = list(malicious_ids)
    c = len(mal)
    if c == 0:
        return
    n = len(clients)
    ref0 = clients[mal[0]].g_i
    device, dtype = ref0.device, ref0.dtype
    G = _stack([clients[i].g_i for i in range(n)]).to(device)     # (n, d) honest updates
    d = G.shape[1]
    s = torch.sign(G.sum(dim=0))                                  # natural change direction

    mal_set = set(mal)
    B = G[[i for i in range(n) if i not in mal_set]]              # (n_b, d) benign
    n_b = B.shape[0]
    knn = max(n - c - 2, 1)                                       # Krum neighbor count
    Bsq = (torch.cdist(B, B) ** 2)                               # (n_b, n_b) benign sq-dist
    max_bnorm = torch.norm(B, dim=1).max()

    # lambda upper bound (Fang Thm 1, Eq 3), in update space. Degenerates when n-2c-1 <= 0
    # (c >= ~n/2): there Krum is trivially broken (malicious majority), so start large.
    denom = n - 2 * c - 1
    if denom > 0:
        Bdist = torch.sqrt(Bsq.clamp_min(0.0))
        k2 = min(n - c - 2, n_b - 1)
        sum_knn = torch.sort(Bdist, dim=1).values[:, 1:k2 + 1].sum(dim=1)   # skip self (col 0)
        lam = float(sum_knn.min() / (denom * math.sqrt(d)) + max_bnorm / math.sqrt(d))
    else:
        lam = float(max_bnorm * 10.0)

    # binary search: largest lambda for which Krum still selects a malicious update
    chosen = None
    for _ in range(60):
        if lam < 1e-5:
            break
        if _krum_selects_malicious((-lam) * s, B, Bsq, c, knn):
            chosen = lam
            break
        lam /= 2.0
    chosen = chosen if chosen is not None else 1e-5
    g_final = (-chosen) * s
    for k, cid in enumerate(mal):
        ref = clients[cid].g_i
        v = g_final
        if k > 0:                       # other c-1 malicious: within KRUM_EPS of w'_1 (Fang eps=0.01)
            noise = torch.randn(d, device=device)
            v = g_final + KRUM_EPS * noise / torch.norm(noise).clamp_min(1e-12)
        clients[cid].g_i = v.reshape(ref.shape).to(dtype).to(device)


def apply_model_poisoning(clients, malicious_ids: List[int], attack: str,
                          g_server: torch.Tensor = None) -> None:
    """Dispatch: craft malicious clients' `g_i` in place (Trim/Adaptive/Krum full-knowledge)."""
    if attack == "trim_fltrust":
        trim_attack(clients, malicious_ids)
    elif attack == "adaptive_fltrust":
        if g_server is None:
            raise ValueError("adaptive_fltrust needs g_server (the server root update g_0).")
        adaptive_attack(clients, malicious_ids, g_server)
    elif attack == "krum_fltrust":
        krum_attack(clients, malicious_ids)
    else:
        raise ValueError(f"apply_model_poisoning: unsupported attack '{attack}'")