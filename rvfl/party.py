"""The parties of the RVFL protocol: one server and N clients.

A round has four phases. Each client phase is a generator: it runs up to the point where it
has to wait for the other parties and yields there, so one process can drive every party
(the cost measurement) and a process per party can simply run its phases to the end (the
end-to-end run, where the barriers in _sync line the parties up).

  train         every party trains locally from the global model; the server's update is
                the root update g_S, a client's is g_i
  enrollment    the server publishes g_S; each client deals first-level shares of its
                perpendicular component g_i_perp (packed k1 values per share, threshold t1)
  verification  each client computes shares of <g_S, g_i_perp> and <g_i_perp, g_i_perp>,
                re-shares them with Pedersen commitments (pairwise verification, k2 = 1,
                threshold t2), reduces their degree (transposed degree reduction) and sends
                the results to the server, which reconstructs both dot products per client and
                derives the trust scores
  aggregation   each client sends the trust-score-weighted sum of the first-level shares it
                holds; the server reconstructs the aggregate and updates the global model

Each client reports its cosine with g_S, at enrollment or with its verification shares
(cos_at_enrollment). With the participant filter (rvfl.filtering) it is always at
enrollment, since the server decides from the cosines who takes part in the round.
"""
import math
import pickle
import time
from fractions import Fraction
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import vpss_module as vpss

from . import comms
from .data import ClientDataLoader
from .filtering import FilterPolicy, check_threshold, privacy_bound, round_threshold

SERVER_ID = -1
# Trust scores are multiplied by mult_TS = 10 ** TS_EXP and rounded before they scale the shares.
TS_EXP = 8
# Relative tolerance of the server's checks ||v||^2 = ||g_S||^2 and <v, g_S> = 0 on the
# reconstructed (quantized) dot products.
VERIFY_TOLERANCE = 5e-5
# A cosine this close to 1 is taken as 1.
FLOAT_PRECISION = 1e-8


def quantization_exponent(g_server_norm, val_max, n_clients, model_size):
    """exp of the quantization multiplier mult = 10 ** exp: the largest power of ten with
        n_clients * mult_TS * (mult * ||g_S|| + sqrt(M) / 2) < sqrt(q / 2)      (appendix G.1).
    The quantized g_S has norm at most mult * ||g_S|| + sqrt(M) / 2 (each entry is rounded by
    at most 1/2), so an aggregate of n_clients vectors of that norm weighted by trust scores of
    at most mult_TS stays below sqrt(q / 2) in norm, and the dot products the server checks
    cannot wrap around q. val_max is sqrt(q / 2)."""
    g = float(g_server_norm)
    mult_ts = 10 ** TS_EXP
    half_sqrt_m = math.sqrt(model_size) / 2
    bound = val_max / (n_clients * mult_ts) - half_sqrt_m          # the largest mult * ||g_S||
    if not (g > 0 and math.isfinite(g) and bound > 0):
        raise ValueError(f"no quantization multiplier: ||g_S|| = {g}, sqrt(q/2) = {val_max}, "
                         f"N = {n_clients}, M = {model_size}")
    exp = math.floor(math.log10(bound / g))
    while not n_clients * mult_ts * (10.0 ** exp * g + half_sqrt_m) < val_max:
        exp -= 1                                                    # floating-point rounding
    return exp


def quantize_trust_score(ts) -> int:
    """ts * mult_TS rounded to the nearest integer, halves up, computed exactly."""
    return math.floor(Fraction(ts) * 10 ** TS_EXP + Fraction(1, 2))


def as_cosine(value) -> float:
    """A reported cosine as a float; anything that is not a number becomes NaN (and fails every
    check that uses it)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def to_signed(value: int, q_int: int) -> int:
    """Z_q element to a signed integer: values above q / 2 are negative."""
    return value - q_int if value > q_int // 2 else value


class Party:
    def __init__(self,
                 id: int,
                 ip: str,
                 port: int,
                 model_name: str,
                 dataset_name: str,
                 model: torch.nn.Module,
                 trainer,
                 partition_path: str,
                 device: str = 'cpu',
                 dataset_path: str = './dataset',
                 batch_size: int = 32,
                 val_max: int = 0):
        self.id = id
        self.comms = comms.make_comms(ip, port)
        self.device = device
        self.model_name = model_name
        self.model = model
        self.trainer = trainer
        self.dataset_name = dataset_name
        self._partition_path = partition_path
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self.dataloader = ClientDataLoader(dataset_name, model_name, id, partition_path,
                                           dataset_path, batch_size)
        self.val_max = val_max          # sqrt(q / 2)
        self.cos_at_enrollment = False
        # Participant filter (enable_filter). Off: every client takes part in every round and
        # the share lists are indexed by client id.
        self.filter_on = False
        self.participants = None        # this round's client ids, sorted (filter on)
        self.pos_of = None              # client id -> position in self.participants
        self.participating = True
        self._time_logs = {}
        self._comm_logs = {'unicast': {}, 'broadcast': {}}

    # --- logs -------------------------------------------------------------------------------
    def extract_and_clear_logs(self, iteration: int) -> Dict[str, Any]:
        result = {'party_id': self.id, 'iteration': iteration,
                  'time_logs_ms': self._time_logs, 'comm_logs_bytes': self._comm_logs}
        self._time_logs = {}
        self._comm_logs = {'unicast': {}, 'broadcast': {}}
        return result

    # --- quantization -----------------------------------------------------------------------
    @staticmethod
    def quantize(update_tensor: torch.Tensor, mult) -> List[int]:
        scaled = torch.round(update_tensor.to(torch.float64) * float(mult)).tolist()
        return [int(v) for v in scaled]

    @staticmethod
    def l2norm(values: List[int]) -> int:
        return math.isqrt(sum(v * v for v in values))

    @staticmethod
    def to_bigint_vector(values) -> "vpss.BigIntVector":
        vec = vpss.BigIntVector()
        for v in values:
            vec.append(vpss.BigInt(str(v)))
        return vec

    # --- compute lock and timers (end-to-end run; see rvfl.comms) ---------------------------
    def _drop_compute(self):
        """Give up the machine before a network operation. Nothing takes the lock back until
        the next timer stamp, so a whole send or receive loop runs without it."""
        if comms.compute_lock_enabled() and not getattr(self, '_lock_dropped', False):
            comms.release_compute()
            self._lock_dropped = True

    def _clock(self):
        """Wall clock minus the time this party has spent waiting for the compute lock; also
        where a dropped lock is taken back, so the wait falls between two stamps and is
        subtracted from the interval it fell in."""
        if comms.compute_lock_enabled() and getattr(self, '_lock_dropped', False):
            t0 = time.perf_counter()
            comms.acquire_compute()
            self._lock_dropped = False
            self._lock_wait_s = getattr(self, '_lock_wait_s', 0.0) + (time.perf_counter() - t0)
        return time.perf_counter() - getattr(self, '_lock_wait_s', 0.0)

    # --- messages ---------------------------------------------------------------------------
    def unicast(self, data: Dict[str, Any], receiver_id: int, logging_info: str = ''):
        self._drop_compute()
        raw = pickle.dumps(data)
        self._comm_logs['unicast'].setdefault(logging_info, {})[receiver_id] = {'size_bytes': len(raw)}
        self.comms.send_bytes(raw, self.party_map[receiver_id])

    def broadcast(self, data: Dict[str, Any], logging_info: str = '', targets=None):
        """Send to every client in party_map, or to `targets` (this round's participants)."""
        self._drop_compute()
        raw = pickle.dumps(data)
        candidates = list(self.party_map.keys()) if targets is None else list(targets)
        receiver_ids = [pid for pid in candidates if pid != SERVER_ID]
        self._comm_logs['broadcast'][logging_info] = {'num_receivers': len(receiver_ids),
                                                      'size_bytes': len(raw)}
        self.comms.broadcast_bytes(raw, [self.party_map[pid] for pid in receiver_ids])

    def receive(self, id: int, block: bool = True, timeout: float = 30):
        self._drop_compute()
        data = self.comms.receive(self.party_map[id], block=block, timeout=timeout)
        return pickle.loads(data) if data else data

    def _receive_key(self, sender_id: int, key: str, timeout: float):
        """Block until a message carrying `key` arrives from sender_id."""
        while True:
            data = self.receive(sender_id, True, timeout)
            if data is not None and key in data:
                return data

    def _sync(self, kind: str = 'all'):
        """Barrier at a computation-to-communication boundary (end-to-end run only): every
        party of the coming exchange waits until all have finished computing, so they start
        sending together. kind='all' is the server and the clients, kind='client' the clients
        only. The compute lock is released while waiting. No-op in a single process."""
        barrier = getattr(self, '_all_barrier', None) if kind == 'all' \
            else getattr(self, '_client_barrier', None)
        if barrier is None:
            return
        self._drop_compute()
        barrier.wait()

    def _peer_send_order(self):
        """Recipients of an all-to-all send, rotated to start at this client: at step s client
        i sends to client i + s, so every step hits distinct receivers instead of every client
        sending to the same one. The client itself is included (it keeps its own share)."""
        ids = self._client_ids()
        if self.id in ids:
            k = ids.index(self.id)
            return ids[k:] + ids[:k]
        return ids

    # --- VPSS configuration -----------------------------------------------------------------
    def set_VPSS_params(self, pp, config1, config2, config1_det):
        """config1 / config2: first- and second-level sharing. config1_det is config1 with the
        randomness of the [k1, t1) coefficients set to zero: g_S is public, so every client
        derives the same shares of it on its own."""
        self.pp = pp
        self.config1 = config1
        self.config2 = config2
        self.config1_det = config1_det

    def _client_ids(self):
        """This round's clients: the participant list with the filter on, otherwise every
        client in party_map."""
        if self.filter_on and self.participants is not None:
            return list(self.participants)
        return sorted(pid for pid in self.party_map if pid != SERVER_ID)

    def _pos(self, client_id):
        """Index of a client in the share lists and in config.shareIndices: its id with the
        filter off, its position in the participant list with it on."""
        if self.filter_on and self.pos_of is not None:
            return self.pos_of[client_id]
        return client_id

    def enable_filter(self, n_total: int, k1: int, k2: int, corrupt_bound: int):
        """Turn the participant filter on. The configurations are then rebuilt every round from
        the participant list (set_round_participants). Client i keeps its evaluation point
        n_total + 1 + i in every round."""
        self.filter_on = True
        self.cos_at_enrollment = True
        self.n_total = n_total
        self.k1 = k1
        self.k2 = k2
        self.corrupt_bound = corrupt_bound
        self.privacy_bound = privacy_bound(corrupt_bound)

    @staticmethod
    def build_vpss_config(t, k, secret_count, share_indices, deterministic_rand=False):
        config = vpss.VPSSConfig()
        config.t = t
        config.k = k
        config.N = len(share_indices)
        config.padType = vpss.PaddingMode.Zero
        secret_vec = vpss.BigIntVector()
        for i in range(1, secret_count + 1):
            secret_vec.append(vpss.BigInt(i))
        config.secretIndices = secret_vec
        share_vec = vpss.BigIntVector()
        for x in share_indices:
            share_vec.append(vpss.BigInt(x))
        config.shareIndices = share_vec
        if deterministic_rand:
            config.deterministicRand = True
        return config

    def set_round_participants(self, participants, k1, t=None):
        """Fix this round's participant list and rebuild the configurations for it. k1 is the
        round's first-level packing as the server chose it. t is given only by the filter's
        "lower_bound" threshold rule; None applies filtering.round_threshold."""
        self.participants = sorted(participants)
        self.pos_of = {pid: idx for idx, pid in enumerate(self.participants)}
        self.participating = self.id in self.pos_of
        n = len(self.participants)
        k1 = int(k1)
        self.round_k1 = k1
        k_max = max(k1, self.k2)
        c = self.corrupt_bound
        if t is not None:
            t = int(t)
            if not check_threshold(n, k_max, t, c):
                raise ValueError(f"[Party {self.id}] t={t} is not valid for n={n}, k={k_max}, c={c}")
        else:
            t, valid = round_threshold(n, k_max, c)
            if not valid:
                print(f"[Party {self.id}] WARNING: n={n} leaves no t in [k + cp, n - 2c] with "
                      f"k <= n // 3 (k={k_max}, c={c}, cp={self.privacy_bound}).")
        self.round_t = t
        secret_count = max(t, k_max)
        share_indices = [self.n_total + 1 + pid for pid in self.participants]
        self.config1 = Party.build_vpss_config(t, k1, secret_count, share_indices)
        self.config2 = Party.build_vpss_config(t, self.k2, secret_count, share_indices)
        self.config1_det = Party.build_vpss_config(t, k1, secret_count, share_indices,
                                                   deterministic_rand=True)

    # --- local training ---------------------------------------------------------------------
    def set_poisoning(self, poisoning_config):
        """Rebuild this party's data loader with a data-poisoning attack."""
        self.dataloader = ClientDataLoader(self.dataset_name, self.model_name, self.id,
                                           self._partition_path, self._dataset_path,
                                           self._batch_size, poisoning_config=poisoning_config)

    def train(self, epoch: int, lr: float):
        # copy=True: vector_to_parameters makes the parameters views of the vector, and
        # training must not write into self.weights
        torch.nn.utils.vector_to_parameters(self.weights.to(self.device, copy=True),
                                            self.model.parameters())
        return self.trainer.train_and_get_update(self.dataloader, epoch, lr)

    def train_phase(self, epoch: int, lr: float):
        """Local training; the update is kept as self.update (g_S on the server, g_i on a client)."""
        self._time_logs.setdefault('train', {})
        t_start = self._clock()
        self.update = self.train(epoch, lr)
        self._time_logs['train']['process_ms'] = (self._clock() - t_start) * 1000

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        """cos(a, b), computed in float64; a value within FLOAT_PRECISION of 1 is taken as 1."""
        cos = float(F.cosine_similarity(a.to(torch.float64), b.to(torch.float64), dim=0))
        return 1.0 if cos >= 1.0 - FLOAT_PRECISION else cos

    @staticmethod
    def get_perpendicular_vector(original_vector: torch.Tensor, base_vector: torch.Tensor) -> torch.Tensor:
        """The component of original_vector perpendicular to base_vector, scaled to
        ||base_vector|| (computed in float64, returned in the input's dtype). When the two are
        parallel (cosine taken as 1), a random perpendicular vector of that norm instead."""
        if not torch.isfinite(original_vector).all():
            print("The input vector has abnormal values; using the zero vector.")
            return torch.zeros_like(original_vector)
        o = original_vector.to(torch.float64)
        b = base_vector.to(torch.float64)
        cos = Party.cosine(o, b)
        if cos == 1.0:
            return Party.random_perpendicular_vector(base_vector)
        b_norm = torch.norm(b)
        o = (b_norm / torch.norm(o)) * o
        perp = o - cos * b
        perp *= b_norm / torch.norm(perp)
        return perp.to(original_vector.dtype)

    @staticmethod
    def random_perpendicular_vector(base_vector: torch.Tensor) -> torch.Tensor:
        """A random vector perpendicular to base_vector, scaled to ||base_vector||: the
        perpendicular part of a Gaussian vector, drawn again in the (practically impossible)
        case that it is too short to normalize. Used when an update points along g_S: its own
        perpendicular component vanishes and its weight sin(2 theta) is 0, but the submission
        still has to pass the norm and orthogonality checks."""
        b = base_vector.to(torch.float64)
        bb = torch.dot(b, b)
        while True:
            r = torch.randn(b.shape, dtype=torch.float64, device=b.device)
            perp = r - (torch.dot(r, b) / bb) * b
            norm = torch.norm(perp)
            if norm > 1e-6 * torch.norm(r):
                return (perp * (torch.sqrt(bb) / norm)).to(base_vector.dtype)


class Server(Party):
    def __init__(self, ip: str, port: int, **kwargs):
        super().__init__(SERVER_ID, ip, port, **kwargs)
        self.party_map = {SERVER_ID: (ip, port)}
        self._value_logs = {}
        self.policy = None              # FilterPolicy when the filter is on
        self.addback_seed = None        # seed of this round's add-back draw (filter on)

    @property
    def g_server(self):
        return self.update

    def extract_and_clear_logs(self, iteration: int) -> Dict[str, Any]:
        result = super().extract_and_clear_logs(iteration)
        result['values'] = self._value_logs
        self._value_logs = {}
        return result

    def enable_filter(self, n_total, k1, k2, corrupt_bound, policy: FilterPolicy = None):
        super().enable_filter(n_total, k1, k2, corrupt_bound)
        self.policy = policy

    # --- enrollment -------------------------------------------------------------------------
    def enrollment_phase(self, thread_count: int = 1):
        tl = self._time_logs['enrollment'] = {}
        t_start = self._clock()
        n_clients = sum(1 for pid in self.party_map if pid != SERVER_ID)
        self.exp = quantization_exponent(torch.norm(self.g_server), self.val_max, n_clients,
                                         self.g_server.numel())
        g_server_quantize = Party.quantize(self.g_server, 10 ** self.exp)
        self.g_server_quantize_norm = Party.l2norm(g_server_quantize)
        t_made = self._clock()
        tl['make_server_data_ms'] = (t_made - t_start) * 1000

        self._sync('all')                       # region 1: the server publishes g_S
        t_made = self._clock()
        self.broadcast({'g_server': self.g_server, 'exp': self.exp}, logging_info='server_data')
        t_sent = self._clock()
        tl['send_server_data_ms'] = (t_sent - t_made) * 1000
        yield
        if not self.cos_at_enrollment:
            return

        target_ids = sorted(pid for pid in self.party_map if pid != SERVER_ID)
        self.cos_list = {}
        self._sync('all')                       # region 1b: the clients report their cosines
        t_recv_start = self._clock()
        while len(self.cos_list) < len(target_ids):
            for pid in target_ids:
                if pid in self.cos_list:
                    continue
                data = self.receive(pid, block=True, timeout=3.0)
                if data is not None and 'cos' in data:
                    self.cos_list[pid] = as_cosine(data['cos'])
        t_recv = self._clock()
        tl['receive_cos_ms'] = (t_recv - t_recv_start) * 1000
        if not self.filter_on:
            yield
            return

        self._value_logs['cos_reported'] = dict(self.cos_list)
        participants, round_k1, round_t, choice = self.policy.select(self.cos_list, self.addback_seed)
        self.set_round_participants(participants, round_k1, round_t)
        t_select = self._clock()
        tl['select_participants_ms'] = (t_select - t_recv) * 1000

        self._sync('all')                       # region 1c: the server publishes the participants
        t_select = self._clock()
        self.broadcast({'participants': participants, 'k1': round_k1, 't': round_t},
                       logging_info='participant_list')
        t_sent = self._clock()
        tl['send_participant_list_ms'] = (t_sent - t_select) * 1000
        self._value_logs['participants'] = list(participants)
        self._value_logs['num_participants'] = len(participants)
        self._value_logs['round_t'] = self.round_t
        self._value_logs['round_k1'] = self.round_k1
        self._value_logs['num_positive'] = sum(1 for c in self.cos_list.values() if c > 0)
        if choice is not None:
            self._value_logs['round_size_choice'] = choice
        yield

    # --- verification -----------------------------------------------------------------------
    def verification_phase(self, thread_count: int = 1):
        tl = self._time_logs['verification'] = {}
        target_ids = self._client_ids()
        shares_with_server = vpss.BatchShareNoneVector()
        shares_itself = vpss.BatchShareNoneVector()
        if not self.cos_at_enrollment:
            self.cos_list = {}
        received = set()
        self._sync('all')                       # region 5: the clients send their s_tilde
        t_start = self._clock()
        while len(received) < len(target_ids):
            for pid in target_ids:
                if pid in received:
                    continue
                data = self.receive(pid, block=True, timeout=3.0)
                if data is not None and 's_tilde_with_server' in data:
                    shares_with_server.append(data['s_tilde_with_server'])
                    shares_itself.append(data['s_tilde_itself'])
                    if not self.cos_at_enrollment:
                        self.cos_list[pid] = as_cosine(data.get('cos'))
                    received.add(pid)
        t_recv = self._clock()
        tl['receive_s_tilde_ms'] = (t_recv - t_start) * 1000

        # <g_S, g_i_perp> and <g_i_perp, g_i_perp> of every client, with error correction
        dot_with_server = vpss.NoneVPSS.reconstruct(self.pp, self.config2, shares_with_server,
                                                    useErrorCorrection=True, thread_count=thread_count)
        dot_itself = vpss.NoneVPSS.reconstruct(self.pp, self.config2, shares_itself,
                                               useErrorCorrection=True, thread_count=thread_count)

        # The dot products are of the quantized vectors, so they are checked relative to the
        # quantized ||g_S||^2: ||v||^2 = ||g_S||^2 and <v, g_S> = 0. The reported cosine must lie
        # in [-1, 1] (the trust score 2 cos sqrt(1 - cos^2) is not defined outside it).
        self.TS = {}
        raw_cos, ratios_i_dot_i, ratios_s_dot_i = {}, {}, {}
        s_dot_s = self.g_server_quantize_norm ** 2
        q_int = self.pp.q.to_int()
        for pos, i in enumerate(target_ids):          # results are in participant order
            i_dot_i = dot_itself[pos].to_int()
            s_dot_i = to_signed(dot_with_server[pos].to_int(), q_int)
            ratio1 = i_dot_i / s_dot_s
            ratio2 = s_dot_i / s_dot_s
            ratios_i_dot_i[i] = ratio1
            ratios_s_dot_i[i] = ratio2
            valid = (abs(ratio1 - 1) < VERIFY_TOLERANCE and abs(ratio2) < VERIFY_TOLERANCE
                     and -1.0 <= self.cos_list[i] <= 1.0)
            raw_cos[i] = self.cos_list[i]
            if valid:
                # ReLU in place, so the aggregation's sums use the same values as the trust score
                self.cos_list[i] = max(self.cos_list[i], 0)
                cos = self.cos_list[i]
                self.TS[i] = 2 * cos * math.sqrt(1 - cos ** 2)
            else:
                self.TS[i] = -1.0                     # rejected
        t_ts = self._clock()
        tl['calc_TS_ms'] = (t_ts - t_recv) * 1000
        self._value_logs['ratios_i_dot_i'] = ratios_i_dot_i
        self._value_logs['ratios_s_dot_i'] = ratios_s_dot_i
        self._value_logs['cos'] = raw_cos
        self._value_logs['g_server_norm'] = torch.norm(self.g_server).item()
        self._value_logs['exp'] = self.exp

        self._sync('all')                       # region 6: the server publishes the trust scores
        t_ts = self._clock()
        self.broadcast({"TS": self.TS}, logging_info='TS', targets=self.participants)
        tl['send_TS_ms'] = (self._clock() - t_ts) * 1000

    # --- aggregation ------------------------------------------------------------------------
    def aggregation_phase(self, lr_global: float, thread_count: int = 1):
        tl = self._time_logs['aggregation'] = {}
        target_ids = self._client_ids()
        collected = vpss.BatchShareNoneVector()
        cnt = 0
        self._sync('all')                       # region 7a: the clients send their aggregated shares
        t_start = self._clock()
        while cnt < len(target_ids):
            for pid in target_ids:
                data = self.receive(pid, True, None)
                if data and 'aggregated_share' in data:
                    collected.append(data['aggregated_share'])
                    cnt += 1
        t_recv = self._clock()
        tl['receive_agg_share_ms'] = (t_recv - t_start) * 1000

        reconstructed = vpss.NoneVPSS.reconstruct(self.pp, self.config1, collected, True,
                                                  thread_count=thread_count)
        total_scale = 10 ** TS_EXP * 10 ** self.exp
        q_int = self.pp.q.to_int()
        g_sum = torch.tensor([to_signed(v.to_int(), q_int) / total_scale for v in reconstructed],
                             device='cpu')[:self.g_server.size(0)]

        # sum_i cos_i * g_i (each g_i scaled to ||g_S||) / sum_i cos_i
        #   = (g_S * (|I| + sum_i (2 cos_i^2 - 1)) + sum_i sin(2 theta_i) g_i_perp) / (2 sum_i cos_i)
        # over the clients that passed verification; g_sum is the second sum in the numerator.
        valid = [i for i, s in self.TS.items() if s >= 0]
        sum_cos = sum(self.cos_list[i] for i in valid)
        if sum_cos == 0:
            # every valid client has a non-positive cosine
            final_g = self.g_server.cpu()
        else:
            sum_cos2 = sum((2 * (self.cos_list[i] ** 2) - 1) for i in valid)
            final_g = (self.g_server.cpu() * (len(valid) + sum_cos2) + g_sum) / (2 * sum_cos)
        self.weights += final_g * lr_global
        t_update = self._clock()
        tl['update_global_weights_ms'] = (t_update - t_recv) * 1000
        self._value_logs['final_g_norm'] = torch.norm(final_g).item()

        self._sync('all')                       # region 7b: the server publishes the new model
        t_update = self._clock()
        self.broadcast({'global_weights': self.weights}, logging_info='updated_weights')
        tl['send_updated_global_weights_ms'] = (self._clock() - t_update) * 1000


class Client(Party):
    def __init__(self, id: int, ip: str, port: int, server_addr, **kwargs):
        super().__init__(id, ip, port, **kwargs)
        self.party_map = {SERVER_ID: server_addr}
        # what the client submits is multiplied by this; a Scaling attacker sets it to N
        # (rvfl.attacks.submission_scale), and the verification then rejects the submission
        self.submission_scale = 1

    @property
    def g_i(self):
        return self.update

    @g_i.setter
    def g_i(self, value):
        self.update = value

    def _cosine(self, g_server):
        return Party.cosine(self.g_i, g_server)

    # --- enrollment -------------------------------------------------------------------------
    def enrollment_phase(self, thread_count: int = 1):
        tl = self._time_logs['enrollment'] = {}
        self._sync('all')                       # region 1: the server publishes g_S
        t_start = self._clock()
        data = self._receive_key(SERVER_ID, 'g_server', 3.0)
        g_server = data['g_server']
        mult = 10 ** data['exp']
        t_base = self._clock()
        tl['receive_server_ms'] = (t_base - t_start) * 1000

        if self.cos_at_enrollment:
            self.cos = self._cosine(g_server)
            t_cos = self._clock()
            tl['calc_cos_ms'] = (t_cos - t_base) * 1000
            self._sync('all')                   # region 1b: report the cosine
            t_cos = self._clock()
            self.unicast({'cos': self.cos}, SERVER_ID, logging_info='cos')
            tl['send_cos_ms'] = (self._clock() - t_cos) * 1000
            yield
            if self.filter_on:
                self._sync('all')               # region 1c: the server publishes the participants
                t_wait = self._clock()
                data = self._receive_key(SERVER_ID, 'participants', 3.0)
                self.set_round_participants(data['participants'], data['k1'], data['t'])
                t_base = self._clock()
                tl['receive_participant_list_ms'] = (t_base - t_wait) * 1000
            else:
                t_base = self._clock()

        if self.participating:
            # shares of the public g_S, the same at every client (config1_det)
            self.g_server_shares, _ = vpss.NoneVPSS.distribute(
                self.pp, self.config1_det,
                Party.to_bigint_vector(Party.quantize(g_server, mult)), thread_count=thread_count)
            t_gs = self._clock()
            tl['recompute_server_share_ms'] = (t_gs - t_base) * 1000
            if not self.cos_at_enrollment:
                self.cos = self._cosine(g_server)
            g_i_perp = Party.get_perpendicular_vector(self.g_i, g_server) * self.submission_scale
            self.shares, _ = vpss.NoneVPSS.distribute(
                self.pp, self.config1, Party.to_bigint_vector(Party.quantize(g_i_perp, mult)),
                thread_count=thread_count)
            tl['calc_first_level_share_ms'] = (self._clock() - t_gs) * 1000
        else:
            self.shares = None
            self.g_server_shares = None
            self.first_level_shares = {}

        self._sync('client')                    # region 2: the clients exchange first-level shares
        t_send = self._clock()
        if self.participating:
            for pid in self._peer_send_order():
                self.unicast({'share': self.shares[self._pos(pid)]}, pid,
                             logging_info='first_level_share')
        tl['send_first_level_share_ms'] = (self._clock() - t_send) * 1000
        yield

        t_wait = self._clock()
        if self.participating:
            self.first_level_shares = {}
            senders = self._client_ids()
            while len(self.first_level_shares) < len(senders):
                for pid in senders:
                    if pid in self.first_level_shares:
                        continue
                    data = self.receive(pid, block=True, timeout=3.0)
                    if data is not None and 'share' in data:
                        self.first_level_shares[pid] = data['share']
        tl['receive_first_level_share_ms'] = (self._clock() - t_wait) * 1000
        yield

    # --- verification -----------------------------------------------------------------------
    def pairwise_verification(self, first_level_shares, first_level_shares_by_dealer, thread_count: int = 1):
        """Re-share this client's shares of the two dot products of every dealer with Pedersen
        commitments, check every received second-level share against its commitments and
        against the first-level value it should carry, and agree on the set of dealers every
        client accepted. Returns (accepted dealers, their second-level shares x2)."""
        tl = self._time_logs['verification']
        t_start = self._clock()
        shares1, commitments1, blinds1 = vpss.PedersenVPSS.distribute(
            self.pp, self.config2, first_level_shares[0], thread_count=thread_count, return_blinds=True)
        shares2, commitments2, blinds2 = vpss.PedersenVPSS.distribute(
            self.pp, self.config2, first_level_shares[1], thread_count=thread_count, return_blinds=True)
        t_made = self._clock()
        tl['make_second_level_share_ms'] = (t_made - t_start) * 1000

        self._sync('client')                    # region 3: second-level shares and commitments
        t_made = self._clock()
        self.broadcast({'commitment1': commitments1, 'commitment2': commitments2},
                       logging_info='commitment', targets=self.participants)
        for pid in self._peer_send_order():
            pos = self._pos(pid)
            self.unicast({'share1': shares1[pos], 'blind1': blinds1[pos],
                          'share2': shares2[pos], 'blind2': blinds2[pos]}, pid,
                         logging_info='second_level_share')
        tl['send_second_level_share_ms'] = (self._clock() - t_made) * 1000
        yield

        t_wait = self._clock()
        shares_in1, blinds_in1, commits_in1 = {}, {}, {}
        shares_in2, blinds_in2, commits_in2 = {}, {}, {}
        target_ids = self._client_ids()
        while (len(shares_in1) < len(target_ids) or len(commits_in1) < len(target_ids)
               or len(shares_in2) < len(target_ids) or len(commits_in2) < len(target_ids)):
            for pid in target_ids:
                if pid in shares_in1 and pid in commits_in1:
                    continue
                data = self.receive(pid, True, 1.0)
                if data is None:
                    continue
                if 'share1' in data:
                    shares_in1[pid] = data['share1']
                    blinds_in1[pid] = data['blind1']
                    shares_in2[pid] = data['share2']
                    blinds_in2[pid] = data['blind2']
                if 'commitment1' in data:
                    commits_in1[pid] = data['commitment1']
                    commits_in2[pid] = data['commitment2']
        t_recv = self._clock()
        tl['receive_second_level_share_ms'] = (t_recv - t_wait) * 1000

        # local verdicts: A. the share matches the dealer's commitments; B. the dealer re-shared
        # the first-level value this client dealt to it (owner check, at this client's slot)
        verdict = {pid: True for pid in target_ids}
        k2 = self.config2.k
        my_index = self._pos(self.id)
        block_idx, idx_in_block = my_index // k2, my_index % k2
        x = self.config2.secretIndices[idx_in_block]
        for pid in target_ids:
            try:
                if not (vpss.PedersenVPSS.verify_batch_share(self.pp, shares_in1[pid], commits_in1[pid],
                                                             thread_count=thread_count)
                        and vpss.PedersenVPSS.verify_batch_share(self.pp, shares_in2[pid], commits_in2[pid],
                                                                 thread_count=thread_count)):
                    verdict[pid] = False
                    continue
                pos = self._pos(pid)
                if not (vpss.verify_Pedersen(self.pp, x, first_level_shares_by_dealer[0][pos],
                                             blinds_in1[pid], commits_in1[pid][block_idx])
                        and vpss.verify_Pedersen(self.pp, x, first_level_shares_by_dealer[1][pos],
                                                 blinds_in2[pid], commits_in2[pid][block_idx])):
                    verdict[pid] = False
            except Exception as e:
                print(f"[Client {self.id}] verification of dealer {pid} raised: {e}")
                verdict[pid] = False
        t_verified = self._clock()
        tl['verify_second_level_share_ms'] = (t_verified - t_recv) * 1000

        self._sync('client')                    # region 4: the clients exchange their verdicts
        t_verified = self._clock()
        self.broadcast({'verdict_vector': verdict}, logging_info='verification_result',
                       targets=self.participants)
        tl['send_verify_result_ms'] = (self._clock() - t_verified) * 1000
        yield

        t_wait = self._clock()
        global_verdict = {pid: True for pid in target_ids}
        received = set()
        while len(received) < len(target_ids):
            for pid in target_ids:
                if pid in received:
                    continue
                data = self.receive(pid, True, 1.0)
                if data is not None and 'verdict_vector' in data:
                    for tid in global_verdict:
                        if tid in data['verdict_vector']:
                            global_verdict[tid] = global_verdict[tid] and data['verdict_vector'][tid]
                    received.add(pid)
        t_recv = self._clock()
        tl['receive_verify_result_ms'] = (t_recv - t_wait) * 1000

        qualified = sorted(pid for pid, ok in global_verdict.items() if ok)
        honest1 = {pid: shares_in1[pid] for pid in qualified if pid in shares_in1}
        honest2 = {pid: shares_in2[pid] for pid in qualified if pid in shares_in2}
        tl['filter_honest_clients_ms'] = (self._clock() - t_recv) * 1000
        return qualified, honest1, honest2

    def transposed_degree_reduction(self, honest_set, honest_second_level_shares):
        """Reduce the degree-2(t1 - 1) product shares of the accepted dealers to this client's
        shares of the k1 packed results: S_tilde = D_j * T with T the first k1 columns of
        (V_e * V_alpha^-1)^T (paper, Transposed Degree Reduction). Returns a blocks x k1 matrix."""
        t1, k1, q = self.config1.t, self.config1.k, self.pp.q
        n_honest = len(honest_set)
        # a product of two degree-(t1 - 1) polynomials needs 2 t1 - 1 points; with fewer the
        # interpolation still runs but returns wrong values
        if n_honest < 2 * t1 - 1:
            raise ValueError(f"Not enough honest parties for degree reduction: need 2t-1 = "
                             f"{2 * t1 - 1} (t={t1}), got {n_honest}")
        alpha_vec = vpss.BigIntVector()
        for pid in honest_set:
            alpha_vec.append(self.config1.shareIndices[self._pos(pid)])
        e_vec = vpss.BigIntVector()
        for i in range(t1):
            e_vec.append(self.config1.secretIndices[i])
        V_alpha = vpss.create_vandermonde_matrix(alpha_vec, n_honest - 1, q)     # |I| x |I|
        V_e = vpss.create_vandermonde_matrix(e_vec, n_honest - 1, q)             # t1 x |I|
        M_T = vpss.multiply_matrix(V_e, vpss.invert_matrix(V_alpha, q), q)
        M = vpss.transpose_matrix(M_T)                                            # |I| x t1
        T = vpss.BigIntMatrix()
        for r in range(n_honest):
            row = vpss.BigIntVector()
            for c in range(k1):
                row.append(M[r][c])
            T.append(row)
        num_blocks = len(honest_second_level_shares[honest_set[0]].s_values)
        D = vpss.BigIntMatrix()
        for blk in range(num_blocks):
            row = vpss.BigIntVector()
            for pid in honest_set:
                row.append(honest_second_level_shares[pid].s_values[blk])
            D.append(row)
        return vpss.multiply_matrix(D, T, q)

    def verification_phase(self, thread_count: int = 1):
        tl = self._time_logs['verification'] = {}
        t_start = self._clock()
        if not self.participating:
            # Sitting the round out: the same barriers and yields as a participant, no traffic.
            self.TS = {}
            self._sync('client')                # region 3
            yield
            self._sync('client')                # region 4
            yield
            self._sync('all')                   # region 5
            yield
            self._sync('all')                   # region 6
            yield
            return

        # this client's shares of <g_S, g_i_perp> and <g_i_perp, g_i_perp> for every dealer i,
        # and the same products of the shares it dealt (for the owner check)
        with_server, itself = vpss.BigIntVector(), vpss.BigIntVector()
        with_server_by_dealer, itself_by_dealer = vpss.BigIntVector(), vpss.BigIntVector()
        my_pos = self._pos(self.id)
        for pid in sorted(self.first_level_shares.keys()):
            peer_pos = self._pos(pid)
            with_server.append(vpss.compute_dot_product(
                self.g_server_shares[my_pos].s_values, self.first_level_shares[pid].s_values, self.pp.q))
            with_server_by_dealer.append(vpss.compute_dot_product(
                self.g_server_shares[peer_pos].s_values, self.shares[peer_pos].s_values, self.pp.q))
            itself.append(vpss.compute_dot_product(
                self.first_level_shares[pid].s_values, self.first_level_shares[pid].s_values, self.pp.q))
            itself_by_dealer.append(vpss.compute_dot_product(
                self.shares[peer_pos].s_values, self.shares[peer_pos].s_values, self.pp.q))
        self.shares = None
        self.g_server_shares = None
        tl['calc_partial_dots_ms'] = (self._clock() - t_start) * 1000

        qualified, honest1, honest2 = yield from self.pairwise_verification(
            (with_server, itself), (with_server_by_dealer, itself_by_dealer), thread_count=thread_count)
        t_after = self._clock()

        S_with_server = self.transposed_degree_reduction(qualified, honest1)
        S_itself = self.transposed_degree_reduction(qualified, honest2)
        ones = vpss.BigIntVector()
        for _ in range(self.config1.k):
            ones.append(vpss.BigInt(1))
        final1, final2 = vpss.BatchShareNone(), vpss.BatchShareNone()
        final1.x = self.first_level_shares[self.id].x
        final1.s_values = vpss.multiply_matrix_vec(S_with_server, ones, self.pp.q)
        final2.x = self.first_level_shares[self.id].x
        final2.s_values = vpss.multiply_matrix_vec(S_itself, ones, self.pp.q)
        t_done = self._clock()
        tl['calc_s_tilde_ms'] = (t_done - t_after) * 1000

        data = {'s_tilde_with_server': final1, 's_tilde_itself': final2}
        if not self.cos_at_enrollment:
            data['cos'] = self.cos
        self._sync('all')                       # region 5: send s_tilde to the server
        t_done = self._clock()
        self.unicast(data, SERVER_ID, logging_info='s_tilde')
        tl['send_s_tilde_ms'] = (self._clock() - t_done) * 1000
        yield

        self._sync('all')                       # region 6: the server publishes the trust scores
        t_wait = self._clock()
        self.TS = self._receive_key(SERVER_ID, 'TS', 1.0)['TS']
        tl['receive_TS_ms'] = (self._clock() - t_wait) * 1000
        yield

    # --- aggregation ------------------------------------------------------------------------
    def aggregation_phase(self):
        tl = self._time_logs['aggregation'] = {}
        t_start = self._clock()
        if not self.participating:
            # no share to send, but the new global model still has to arrive
            self._sync('all')                   # region 7a
            yield
            self._sync('all')                   # region 7b
            self.weights = self._receive_key(SERVER_ID, 'global_weights', 3.0)['global_weights']
            yield
            return

        aggregated = vpss.BatchShareNone()
        aggregated.x = self.first_level_shares[self.id].x
        aggregated.s_values = self.first_level_shares[self.id].s_values
        vpss.scaleVector(aggregated.s_values, vpss.BigInt("0"), self.pp.q)
        for sender_id, share in self.first_level_shares.items():
            ts = self.TS.get(sender_id, 0.0)
            if ts <= 0.0:                       # rejected (-1) or zero weight
                continue
            vpss.scaleVector(share.s_values, vpss.BigInt(str(quantize_trust_score(ts))), self.pp.q)
            vpss.addVectors(aggregated.s_values, share.s_values, self.pp.q)
        t_done = self._clock()
        tl['calc_agg_share_ms'] = (t_done - t_start) * 1000

        self._sync('all')                       # region 7a: send the aggregated share
        t_done = self._clock()
        self.unicast({'aggregated_share': aggregated}, SERVER_ID, logging_info='agg_share')
        tl['send_agg_share_ms'] = (self._clock() - t_done) * 1000
        yield

        self._sync('all')                       # region 7b: the server publishes the new model
        t_wait = self._clock()
        self.weights = self._receive_key(SERVER_ID, 'global_weights', 3.0)['global_weights']
        tl['receive_global_weights_ms'] = (self._clock() - t_wait) * 1000
        yield
