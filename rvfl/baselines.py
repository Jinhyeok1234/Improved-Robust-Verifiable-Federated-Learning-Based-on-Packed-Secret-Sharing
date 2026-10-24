"""Aggregation rules computed in the clear, for the accuracy experiments, and the BREA baseline
of the cost comparison.

  fltrust_aggregate          plaintext FLTrust (Cao et al., NDSS'21)
  rvfl_plaintext_aggregate   RVFL's aggregation without the secret sharing: the same
                             perpendicular decomposition, quantization over Z_q, norm and
                             orthogonality checks and trust scores, on the plaintext vectors.
                             The model it produces is the one the protocol produces, without the
                             cryptographic cost per round, so long training runs are feasible.
  BreaServer / BreaClient    BREA (So et al., 2021): verifiable secret sharing of every update,
                             pairwise distances reconstructed by the server, Multi-Krum selection
"""
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import vpss_module as vpss

from .party import (Client, Party, SERVER_ID, Server, TS_EXP, quantization_exponent,
                    quantize_trust_score, to_signed)

# Relative tolerance of the plaintext norm and orthogonality checks.
PLAINTEXT_VERIFY_TOLERANCE = 1e-3


def fltrust_aggregate(g_server: torch.Tensor, updates: Dict[int, torch.Tensor]):
    """FLTrust: sum_i ReLU(cos_i) * (||g_S|| / ||g_i||) g_i / sum_i ReLU(cos_i).
    Returns (global update, values for the log)."""
    cos_sum = 0
    g_server_norm = torch.norm(g_server)
    final_g = torch.zeros_like(g_server)
    cos_list = []
    for cid in updates:
        cos = float(F.cosine_similarity(updates[cid], g_server, dim=0))
        cos_list.append(cos)
        if cos > 0:
            cos_sum += cos
            final_g += (g_server_norm / (torch.norm(updates[cid]) + 1e-8) * cos) * updates[cid]
    if cos_sum > 1e-8:
        final_g /= cos_sum
    else:
        final_g = torch.zeros_like(g_server)
    values = {'cos': cos_list, 'final_g_norm': torch.norm(final_g).item(),
              'g_server_norm': torch.norm(g_server).item()}
    return final_g, values


def rvfl_plaintext_aggregate(g_server: torch.Tensor, updates: Dict[int, torch.Tensor], val_max: int,
                             q_int: int, attack: Optional[str] = None,
                             malicious_ids: Optional[List[int]] = None):
    """RVFL's aggregation on plaintext vectors. Each client submits its perpendicular component
    scaled to ||g_S|| (a Scaling attacker scales it by the number of clients); a submission
    failing ||v|| = ||g_S|| or <v, g_S> = 0 is discarded, as the protocol's verification does.
    Returns (global update, values for the log)."""
    exp = quantization_exponent(torch.norm(g_server), val_max, len(updates), g_server.numel())
    mult = 10 ** exp
    g_server_norm = torch.norm(g_server)
    gS_sq = float(g_server_norm ** 2)
    malicious = set(malicious_ids or [])
    scaling = attack == "scaling_fltrust"
    g_perp = [0] * g_server.numel()
    cos_all, cos_valid, rejected = {}, {}, []
    sum_cos = 0
    perps = {}
    for cid in updates:
        cos = Party.cosine(updates[cid], g_server)          # what the client reports
        cos_all[cid] = cos
        perp = Party.get_perpendicular_vector(updates[cid], g_server)
        if scaling and cid in malicious:
            perp = perp * len(updates)
        perps[cid] = perp
        ratio1 = float(torch.dot(perp, perp)) / gS_sq
        ratio2 = float(torch.dot(perp, g_server)) / gS_sq
        if abs(ratio1 - 1.0) > PLAINTEXT_VERIFY_TOLERANCE or abs(ratio2) > PLAINTEXT_VERIFY_TOLERANCE:
            rejected.append(cid)
            continue
        if cos > 0:
            cos_valid[cid] = cos
            sum_cos += cos
    if cos_valid:
        ts_scale = 10 ** TS_EXP
        for cid in cos_valid:
            scaled = (g_server_norm / torch.norm(perps[cid])) * perps[cid]
            quantized = Party.quantize(scaled, mult)
            sin2 = 2 * cos_valid[cid] * math.sqrt(1 - cos_valid[cid] ** 2)
            quantized_sin2 = quantize_trust_score(sin2)
            quantized = [v * quantized_sin2 for v in quantized]
            g_perp = [g_perp[i] + quantized[i] for i in range(len(g_perp))]
        if max(g_perp) > q_int:
            raise ValueError("An entry of the aggregate exceeds q: the quantization multiplier is too large.")
        div = mult * ts_scale
        g_perp = torch.tensor([v / div for v in g_perp])
        sum_cos2 = sum((2 * (cos_valid[cid] ** 2) - 1) for cid in cos_valid)
        final_g = g_server * (len(cos_valid) + sum_cos2) + g_perp
        final_g /= 2 * sum_cos
    else:
        final_g = g_server
    values = {'g_server_norm': torch.norm(g_server).item(), 'final_g_norm': torch.norm(final_g).item(),
              'cos': cos_all, 'rejected': rejected}
    return final_g, values


# --- BREA -----------------------------------------------------------------------------------

# Updates are quantized with 10 ** BREA_EXP; the pairwise squared distances then carry
# 10 ** (2 * BREA_EXP).
BREA_EXP = 40


class MultiKrum:
    """Multi-Krum selection (Blanchard et al., 2017) on a matrix of pairwise distances: m
    rounds, each picking the remaining update with the least sum of distances to its
    n - k + 1 - a - 2 nearest remaining neighbours."""

    def __init__(self, pairwise_distances: List[List[float]], m: int, a: int) -> None:
        self.d = pairwise_distances
        self.m = m
        self.a = a
        self.n = len(pairwise_distances)

    def select(self) -> List[int]:
        selected: List[int] = []
        for k in range(1, self.m + 1):
            best, best_score = None, float('inf')
            for j in range(self.n):
                if j in selected:
                    continue
                dists = sorted(self.d[j][u] for u in range(self.n) if u not in selected and u != j)
                num_closest = (self.n - k + 1) - self.a - 2
                if num_closest <= 0:
                    raise ValueError("The number of closest distances to sum must be greater than 0.")
                score = sum(dists[:num_closest])
                if score < best_score:
                    best_score, best = float(score), j
            if best is not None:
                selected.append(best)
        return selected


class BreaServer(Server):
    def brea_phase(self, lr_global: float, byzantine_rate: float = 0.3, thread_count: int = 1):
        tl = self._time_logs['BREA'] = {}
        t_start = self._clock()
        target_ids = sorted(pid for pid in self.party_map if pid != SERVER_ID)
        n = len(target_ids)
        x, d = {}, {}
        while len(x) < n:
            for pid in target_ids:
                if pid in x:
                    continue
                data = self.receive(pid, block=True, timeout=3.0)
                if data is not None and 'd' in data:
                    x[pid] = data['x']
                    d[pid] = data['d']
        t_recv = self._clock()
        tl['receive_share_ms'] = (t_recv - t_start) * 1000

        d_vec = vpss.BatchShareNoneVector()
        for i in target_ids:
            d_i = vpss.BatchShareNone()
            d_i.x = x[i]
            d_i.s_values = vpss.BigIntVector()
            for j in target_ids:
                for k in target_ids:
                    if j >= k:
                        continue
                    try:
                        d_i.s_values.append(d[i][j][k])
                    except (KeyError, TypeError):
                        print(f"Party {i} does not have ({j},{k})")
                        d_i.s_values.append(vpss.BigInt('0'))
            d_vec.append(d_i)
        d_bar = vpss.NoneVPSS.reconstruct(self.pp, self.config2, d_vec, useErrorCorrection=True,
                                          thread_count=thread_count)
        t_rec = self._clock()
        tl['reconstruct_d_ms'] = (t_rec - t_recv) * 1000

        q_int = self.pp.q.to_int()
        mult = 10 ** (BREA_EXP * 2)
        d_bar_list = [to_signed(v.to_int(), q_int) / mult for v in d_bar]
        t_deq = self._clock()
        tl['dequantize_d_ms'] = (t_deq - t_rec) * 1000
        self._value_logs['d_bar'] = d_bar_list

        pairwise = [[0.0] * n for _ in range(n)]
        index = 0
        for j in range(n):
            for k in range(j + 1, n):
                pairwise[j][k] = pairwise[k][j] = d_bar_list[index]
                index += 1
        a = math.floor(n * byzantine_rate)
        m = n - 2 * a - 3
        selected = MultiKrum(pairwise, m, a).select()
        S = [target_ids[i] for i in selected]
        t_krum = self._clock()
        tl['multi_krum_ms'] = (t_krum - t_deq) * 1000
        self._value_logs['A'] = a
        self._value_logs['m'] = m
        self._value_logs['selected set'] = selected

        self.broadcast({'S': S}, logging_info='S')
        t_sent = self._clock()
        tl['send_set_ms'] = (t_sent - t_krum) * 1000
        yield

        t_after = self._clock()
        aggregated = vpss.BatchShareNoneVector()
        cnt = 0
        while cnt < n:
            for pid in target_ids:
                data = self.receive(pid, block=True, timeout=None)
                if data is not None and 'aggregated_share' in data:
                    aggregated.append(data['aggregated_share'])
                    cnt += 1
        t_recv = self._clock()
        tl['receive_agg_ms'] = (t_recv - t_after) * 1000

        reconstructed = vpss.NoneVPSS.reconstruct(self.pp, self.config1, aggregated,
                                                  useErrorCorrection=True, thread_count=thread_count)
        t_rec = self._clock()
        tl['reconstruct_agg_ms'] = (t_rec - t_recv) * 1000

        mult = 10 ** BREA_EXP
        g_sum = torch.tensor([to_signed(v.to_int(), q_int) / mult for v in reconstructed],
                             device='cpu')[:self.weights.size(0)]
        self.weights += g_sum * lr_global
        t_upd = self._clock()
        tl['dequantize_and_update_ms'] = (t_upd - t_rec) * 1000
        self._value_logs['final_g_norm'] = torch.norm(g_sum).item()

        self.broadcast({'global_weights': self.weights}, logging_info='updated_weights')
        tl['send_updated_global_weights_ms'] = (self._clock() - t_upd) * 1000
        yield


class BreaClient(Client):
    def brea_phase(self, thread_count: int = 1):
        tl = self._time_logs['BREA'] = {}
        t_start = self._clock()
        # negative coordinates are shared as their residues mod q, the values the commitments
        # bind; the server maps the reconstructed sums back to signed values
        q_int = self.pp.q.to_int()
        quantized = [v % q_int for v in Party.quantize(self.g_i, 10 ** BREA_EXP)]
        shares, commitments = vpss.PedersenVPSS.distribute(
            self.pp, self.config1, Party.to_bigint_vector(quantized), thread_count=thread_count,
            return_blinds=False)
        t_share = self._clock()
        tl['calc_share_ms'] = (t_share - t_start) * 1000

        target_ids = sorted(pid for pid in self.party_map if pid != SERVER_ID)
        self.broadcast({'commitments': commitments}, logging_info='commitment')
        for pid in target_ids:
            self.unicast({'share': shares[pid]}, pid, logging_info='share')
        commitments = None
        t_sent = self._clock()
        tl['send_share_ms'] = (t_sent - t_share) * 1000
        yield

        t_wait = self._clock()
        s, C = {}, {}
        while len(s) < len(target_ids) or len(C) < len(target_ids):
            for pid in target_ids:
                if pid in s and C:
                    continue
                data = self.receive(pid, block=True, timeout=3.0)
                if data is not None:
                    if 'share' in data:
                        s[pid] = data['share']
                    elif 'commitments' in data:
                        C[pid] = data['commitments']
        t_recv = self._clock()
        tl['receive_share_ms'] = (t_recv - t_wait) * 1000

        valid = {pid: vpss.PedersenVPSS.verify_batch_share(self.pp, s[pid], C[pid], thread_count=thread_count)
                 for pid in target_ids}
        C = None
        t_ver = self._clock()
        tl['verification_ms'] = (t_ver - t_recv) * 1000

        # shares of ||v_j - v_k||^2 = ||v_j||^2 + ||v_k||^2 - 2 <v_j, v_k>
        valid_ids = sorted(pid for pid, ok in valid.items() if ok)
        q = self.pp.q
        norm_sq = {pid: vpss.compute_dot_product(s[pid].s_values, s[pid].s_values, q) for pid in valid_ids}
        d = {}
        for a in range(len(valid_ids)):
            for b in range(a + 1, len(valid_ids)):
                j, k = valid_ids[a], valid_ids[b]
                two_jk = (vpss.compute_dot_product(s[j].s_values, s[k].s_values, q) * 2) % q
                d.setdefault(j, {})[k] = ((norm_sq[j] + norm_sq[k]) % q - two_jk) % q
        t_dist = self._clock()
        tl['calc_pairwise_distance_ms'] = (t_dist - t_ver) * 1000

        self.unicast({'x': s[self.id].x, 'd': d}, SERVER_ID, logging_info='d')
        t_sent = self._clock()
        tl['send_pairwise_distance_ms'] = (t_sent - t_dist) * 1000
        yield

        t_wait = self._clock()
        S = self._receive_key(SERVER_ID, 'S', None)['S']
        t_recv = self._clock()
        tl['recv_set_ms'] = (t_recv - t_wait) * 1000

        aggregated = vpss.BatchShareNone()
        aggregated.x = s[self.id].x
        aggregated.s_values = s[self.id].s_values
        vpss.scaleVector(aggregated.s_values, vpss.BigInt("0"), q)
        for pid in S:
            vpss.addVectors(aggregated.s_values, s[pid].s_values, q)
        t_agg = self._clock()
        tl['aggregate_shares_ms'] = (t_agg - t_recv) * 1000

        self.unicast({'aggregated_share': aggregated}, SERVER_ID, logging_info='aggregated_share')
        t_sent = self._clock()
        tl['send_aggregated_shares_ms'] = (t_sent - t_agg) * 1000
        yield

        t_wait = self._clock()
        self.weights = self._receive_key(SERVER_ID, 'global_weights', 3.0)['global_weights']
        tl['receive_global_weights_ms'] = (self._clock() - t_wait) * 1000
        yield
