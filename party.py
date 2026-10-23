import socket
import selectors
import threading
import queue
import pickle
import struct
import time
import sys
import math
import random
from typing import List, Tuple, Any, Dict, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.synchronize import Lock
import collections
import numpy as np
import vpss_module as vpss
import torch.optim as optim

from models import get_model
from trainer import LocalTrainer
from dataloader import CustomDataLoader

SERVER_ID = -1
EXP_MARGIN = 5
TS_EXP = 8
FLOAT_PRECISION = 0.00000001
BREA_EXP = 40

# Network backend for party communication.
#   backend="inmemory"      -> original in-memory shared-memory queues (published runs; default).
#   backend="udp_throttled"  -> real loopback UDP sockets with bandwidth/latency throttling
#                               (for the end-to-end WAN wall-clock experiment).
# main.py updates this dict before parties are constructed.
NETWORK_CONFIG = {
    "backend": "inmemory",
    "bandwidth_bps": 125_000_000,   # bytes/sec (1 Gbps = 125 MB/s)
    "latency_s": 0.025,             # one-way propagation delay; RTT = 2 * latency_s
}

# --- E2E execution controls ---
# Compute serialization: in the E2E parallel run all parties run in their own threads,
# but heavy LOCAL COMPUTATION is serialized (one party at a time) via _COMPUTE_LOCK so a
# single machine does not oversubscribe CPU. Only COMMUNICATION (send/recv) runs
# concurrently: the comm methods release the lock around the actual network op. This
# reproduces the contention-free per-party compute of the sequential runs while letting
# all parties transmit at the same time.
_COMPUTE_LOCK = threading.RLock()
_SERIALIZE_COMPUTE = False          # set True by the E2E parallel driver

def _acquire_compute():
    if _SERIALIZE_COMPUTE:
        _COMPUTE_LOCK.acquire()

def _release_compute():
    if _SERIALIZE_COMPUTE:
        try:
            _COMPUTE_LOCK.release()
        except (RuntimeError, AssertionError):
            pass

# How the lock and the timers interact in the E2E run.
#
# A comm method DROPS the lock before its network operation and does not take it back: the
# next timer stamp (Party._clock) does, and counts how long that took. Every phase timer is
# a difference of _clock() values, and _clock() returns the wall clock minus the total time
# this party has spent waiting for the lock, so an interval never contains a wait for another
# party's serialized computation. Before this, each comm call re-took the lock on its own and
# the stamp came after, so a receive loop over 47 peers ended only when this party got the
# machine back from whoever was computing -- and every "communication" figure carried the
# other parties' computation. At 10 Gbit/s that was about 400 s of a 26 MB exchange.
#
# Dropping once per comm region rather than per call also stops the lock from bouncing
# between 48 parties' send and receive loops, which is what made those loops slow in real
# time as well.

# Broadcast model:
#   False -> serial single-uplink (honest N x payload/bandwidth on one server link; DEFAULT).
#   True  -> per-link parallel (assumes a well-provisioned server with N x bandwidth uplink).
BROADCAST_PARALLEL = False

# Who computes the g_server shares during enrollment:
#   True  -> the server deals and broadcasts the shares (DEFAULT; no C++ rebuild needed).
#   False -> the server broadcasts only g_server (+exp) and each client recomputes the shares
#            itself, deterministically, so all honest clients agree. Requires a config with
#            deterministicRand=True (rebuilt C++). g_server is public, so zeroing the [k,t)
#            randomness is safe. Trades lower server comm for repeated per-client computation.
SERVER_COMPUTES_SHARES = True

# Multi-Krum for BREA
class MultiKrum:
    def __init__(self, pairwise_distances: List[List[float]], m: int, a: int) -> None:
        self.pairwise_distances: List[List[float]] = pairwise_distances
        self.m: int = m
        self.a: int = a
        self.n: int = len(pairwise_distances)
        self.selected_indices: List[int] = []
        self.best_candidate: Optional[int] = None
        self.min_score: float = float('inf')
        self.current_score: float = 0.0
        self.num_closest: int = 0
        self.distances: List[float] = []
        self.k: int = 0
        self.j: int = 0
        self.u: int = 0

    def select(self) -> List[int]:
        self.selected_indices = []

        for self.k in range(1, self.m + 1):
            self.best_candidate = None
            self.min_score = float('inf')

            for self.j in range(self.n):
                if self.j in self.selected_indices:
                    continue

                self.distances = []
                
                for self.u in range(self.n):
                    if self.u in self.selected_indices:
                        continue
                    
                    if self.u == self.j:
                        continue

                    self.distances.append(self.pairwise_distances[self.j][self.u])

                self.distances.sort()

                self.num_closest = (self.n - self.k + 1) - self.a - 2

                if self.num_closest <= 0:
                    raise ValueError("The number of closest distances to sum must be greater than 0.")

                self.current_score = sum(self.distances[:self.num_closest])

                if self.current_score < self.min_score:
                    self.min_score = float(self.current_score)
                    self.best_candidate = self.j

            if self.best_candidate is not None:
                self.selected_indices.append(self.best_candidate)

        return self.selected_indices


class _Inboxes(dict):
    """Sender address -> queue.Queue of that sender's frames, created on first use.

    The receiving thread (_deliver) and the protocol thread (receive) can both be the first
    to touch a sender's address. collections.defaultdict(queue.Queue) is not safe for that:
    building the Queue runs Python code, so the other thread can run in between, build a
    second Queue and store it; one of the two is then overwritten. If the overwritten one
    already holds the sender's first frame, that frame is gone and receive() waits on the
    other queue for good (seen as an E2E round hanging in the first-level share exchange,
    with every byte read off the sockets). Creating the queue under a lock makes both threads
    get the same one."""

    def __init__(self):
        super().__init__()
        self._guard = threading.Lock()

    def __missing__(self, key):
        with self._guard:
            q = dict.get(self, key)
            if q is None:
                q = queue.Queue()
                dict.__setitem__(self, key, q)
            return q


class InMemoryCommsManager:
    _manager = None
    _inboxes = None
    _lock = None
    
    @classmethod
    def _initialize_manager(cls) -> None:
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
            cls._inboxes = cls._manager.dict()
            cls._lock = cls._manager.Lock()

    def __init__(self, ip: str, port: int):
        self._initialize_manager()
        self.my_addr = (ip, port)
        self._queue_cache: Dict[Tuple[Tuple, Tuple], multiprocessing.Queue] = {}
        
        self._active_shms: List[Tuple[shared_memory.SharedMemory, Any]] = []

    def _get_queue(self, target_addr: Tuple[str, int], sender_addr: Tuple[str, int]):
        key = (target_addr, sender_addr)
        if key in self._queue_cache:
            return self._queue_cache[key]

        with self._lock:
            if key not in self._inboxes:
                self._inboxes[key] = self._manager.Queue()
            q = self._inboxes[key]
            self._queue_cache[key] = q
            return q

    def _cleanup_shms(self) -> None:
        alive_shms = []
        for shm, ref_count in self._active_shms:
            if ref_count.value == 0:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            else:
                alive_shms.append((shm, ref_count))
        self._active_shms = alive_shms

    def send_bytes(self, data: bytes, target_addr: Tuple[str, int]) -> None:
        q = self._get_queue(target_addr=target_addr, sender_addr=self.my_addr)
        q.put(data)

    def broadcast_bytes(self, data: bytes, target_addrs: List[Tuple[str, int]]) -> None:
        if not target_addrs:
            return

        self._cleanup_shms()

        data_len = len(data)
        shm = shared_memory.SharedMemory(create=True, size=data_len)
        
        shm.buf[:data_len] = data
        
        ref_count = self._manager.Value('i', len(target_addrs))
        
        meta_msg = ("SHM_BROADCAST", shm.name, data_len, ref_count)

        for addr in target_addrs:
            q = self._get_queue(target_addr=addr, sender_addr=self.my_addr)
            q.put(meta_msg)

        self._active_shms.append((shm, ref_count))

    def receive(self, sender_addr: Tuple[str, int], block: bool = True, timeout: Optional[float] = 30) -> Optional[bytes]:
        q = self._get_queue(target_addr=self.my_addr, sender_addr=sender_addr)
        try:
            raw_data = q.get(block=block, timeout=timeout)
            
            if isinstance(raw_data, tuple) and len(raw_data) == 4 and raw_data[0] == "SHM_BROADCAST":
                _, shm_name, size, ref_count = raw_data
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                try:
                    data = bytes(existing_shm.buf[:size])
                    return data
                finally:
                    existing_shm.close()
                    with self._lock:
                        ref_count.value -= 1
            
            return raw_data
        except (queue.Empty, FileNotFoundError):
            return None

    def close(self) -> None:
        for shm, _ in self._active_shms:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._active_shms.clear()

    def __del__(self) -> None:
        self.close()

    @classmethod
    def clear_directory(cls) -> None:
        if cls._lock is not None:
            with cls._lock:
                cls._inboxes.clear()


class ThrottledUDPCommsManager:
    """Real loopback-UDP comms with WAN throttling (bandwidth + one-way latency).

    Same interface as InMemoryCommsManager (send_bytes / broadcast_bytes / receive).

    Transport: chunks are fired *paced* at the link bandwidth (each chunk sleeps
    chunk_size/bandwidth), so a message occupies its link for exactly len/bandwidth --
    there is NO per-chunk stop-and-wait stall. Reliability uses a completion
    handshake: after firing, the sender POLLs; the receiver replies DONE (all chunks
    present) or MISS (missing indices) and the sender retransmits only those. On
    loopback with pacing, drops are rare so this almost always finishes in one round.

    Latency: a fully-received message is delivered to the receiver's inbox only after
    latency_s (one-way propagation); control packets are not delayed, so protocol
    round-trips incur ~2*latency_s = RTT naturally.

    Unicast occupies one uplink (serial per sender); broadcast pushes to all
    recipients concurrently (per-link model -> single len/bandwidth), matching how
    broadcast volume is counted once in Tables 4-6.
    """
    HEADER_FORMAT = "!I I I I"   # (type, msg_id, total_chunks, chunk_idx)
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    MAX_PAYLOAD_SIZE = 60000
    MAX_MISS_PER_PKT = 14000
    TYPE_DATA = 0
    TYPE_DONE = 2
    TYPE_POLL = 3
    TYPE_MISS = 4
    POLL_TIMEOUT = 0.5
    MAX_ROUNDS = 200

    def __init__(self, ip: str, port: int,
                 bandwidth_bps: float = 125_000_000, latency_s: float = 0.0):
        self.bandwidth_bps = float(bandwidth_bps)
        self.latency_s = float(latency_s)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 64)
        self.sock.bind((ip, port))

        self.my_addr = (ip, port)
        self.party_inboxes = _Inboxes()
        self._buffer = {}            # (addr,msg_id) -> {chunk_idx: payload}
        self._delivered = set()      # (addr,msg_id) already delivered to inbox
        self._recv_lock = threading.Lock()
        self._ctrl_events = {}       # (addr,msg_id) -> Event   (sender waits on it)
        self._miss_map = {}          # (addr,msg_id) -> missing idx list ([] = DONE)
        self._ctrl_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._running = True
        self.msg_counter = 0

        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    def _listen_loop(self):
        while self._running:
            try:
                self.sock.settimeout(1.0)
                raw_data, addr = self.sock.recvfrom(65535)
                self._process_packet(raw_data, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[{self.my_addr}] Receive Error: {e}")

    def _deliver(self, addr, full_data):
        # Apply one-way propagation latency to DATA delivery only.
        if self.latency_s > 0:
            threading.Timer(self.latency_s, self.party_inboxes[addr].put, args=(full_data,)).start()
        else:
            self.party_inboxes[addr].put(full_data)

    def _process_packet(self, raw_data: bytes, addr: Tuple[str, int]):
        if len(raw_data) < self.HEADER_SIZE:
            return
        ptype, msg_id, total, idx = struct.unpack(self.HEADER_FORMAT, raw_data[:self.HEADER_SIZE])
        payload = raw_data[self.HEADER_SIZE:]

        if ptype == self.TYPE_DATA:
            with self._recv_lock:
                self._buffer.setdefault((addr, msg_id), {})[idx] = payload

        elif ptype == self.TYPE_POLL:
            key = (addr, msg_id)
            deliver_data = None
            with self._recv_lock:
                buf = self._buffer.get(key, {})
                missing = [i for i in range(total) if i not in buf]
                if not missing and key not in self._delivered:
                    self._delivered.add(key)
                    deliver_data = b"".join(buf[i] for i in range(total))
                    self._buffer.pop(key, None)
            if not missing:
                if deliver_data is not None:
                    self._deliver(addr, deliver_data)
                self.sock.sendto(struct.pack(self.HEADER_FORMAT, self.TYPE_DONE, msg_id, total, 0), addr)
            else:
                body = b"".join(struct.pack("!I", i) for i in missing[:self.MAX_MISS_PER_PKT])
                self.sock.sendto(struct.pack(self.HEADER_FORMAT, self.TYPE_MISS, msg_id, total, 0) + body, addr)

        elif ptype == self.TYPE_DONE:
            with self._ctrl_lock:
                self._miss_map[(addr, msg_id)] = []
                ev = self._ctrl_events.get((addr, msg_id))
            if ev:
                ev.set()

        elif ptype == self.TYPE_MISS:
            miss = [struct.unpack_from("!I", payload, k * 4)[0] for k in range(len(payload) // 4)]
            with self._ctrl_lock:
                self._miss_map[(addr, msg_id)] = miss
                ev = self._ctrl_events.get((addr, msg_id))
            if ev:
                ev.set()

    def _next_msg_id(self) -> int:
        with self._counter_lock:
            self.msg_counter += 1
            return self.msg_counter

    def _send_reliable(self, data: bytes, target_addr: Tuple[str, int], pace: bool = True) -> None:
        # Reliable transfer: fire chunks + completion handshake (POLL/DONE/MISS).
        # pace=True paces the fire at the link bandwidth (unicast: sender uplink busy).
        # pace=False fires unthrottled (used by broadcast, which charges the server
        # uplink once for the payload afterwards -> per-link model, counted once).
        msg_id = self._next_msg_id()
        cs = self.MAX_PAYLOAD_SIZE
        total = max(1, math.ceil(len(data) / cs))
        key = (target_addr, msg_id)
        ev = threading.Event()
        with self._ctrl_lock:
            self._ctrl_events[key] = ev
            self._miss_map.pop(key, None)
        to_send = list(range(total))
        try:
            for _ in range(self.MAX_ROUNDS):
                # Deadline-based pacing: keep the average send rate at the link bandwidth
                # (sleep only when ahead of schedule) to avoid sub-ms sleep granularity.
                _t0 = time.perf_counter()
                _sent = 0
                for i in to_send:
                    chunk = data[i * cs:(i + 1) * cs]
                    pkt = struct.pack(self.HEADER_FORMAT, self.TYPE_DATA, msg_id, total, i) + chunk
                    self.sock.sendto(pkt, target_addr)
                    if pace and self.bandwidth_bps > 0:
                        _sent += len(chunk)
                        _dt = _t0 + _sent / self.bandwidth_bps - time.perf_counter()
                        if _dt > 0:
                            time.sleep(_dt)
                ev.clear()
                with self._ctrl_lock:
                    self._miss_map.pop(key, None)
                self.sock.sendto(struct.pack(self.HEADER_FORMAT, self.TYPE_POLL, msg_id, total, 0), target_addr)
                if ev.wait(timeout=self.POLL_TIMEOUT):
                    with self._ctrl_lock:
                        miss = self._miss_map.get(key)
                    if not miss:               # DONE (empty list)
                        return
                    to_send = miss             # retransmit only the missing chunks
                else:
                    to_send = []               # no reply -> just re-poll
            print(f"[{self.my_addr}] send to {target_addr} not confirmed after {self.MAX_ROUNDS} rounds")
        finally:
            with self._ctrl_lock:
                self._ctrl_events.pop(key, None)
                self._miss_map.pop(key, None)

    def send_bytes(self, data: bytes, target_addr: Tuple[str, int]) -> None:
        # Unicast: one uplink, serial per sender.
        self._send_reliable(data, target_addr)

    def broadcast_bytes(self, data: bytes, target_addrs: List[Tuple[str, int]]) -> None:
        if not target_addrs:
            return
        if BROADCAST_PARALLEL:
            # Per-link parallel: push to all recipients concurrently and charge the uplink
            # once (len/bandwidth). Assumes a well-provisioned server (aggregate N x uplink).
            senders = [threading.Thread(target=self._send_reliable, args=(data, addr), kwargs={"pace": False})
                       for addr in target_addrs]
            for t in senders:
                t.start()
            for t in senders:
                t.join()
            if self.bandwidth_bps > 0:
                time.sleep(len(data) / self.bandwidth_bps)
        else:
            # Serial single-uplink (DEFAULT): honest N x payload/bandwidth over one link
            # (no WAN multicast). The paced transport makes this reflect real bandwidth
            # only, with no per-chunk stop-and-wait stall.
            for addr in target_addrs:
                self._send_reliable(data, addr, pace=True)

    def receive(self, sender_addr: Tuple[str, int], block: bool = True,
                timeout: Optional[float] = 30) -> Optional[bytes]:
        try:
            return self.party_inboxes[sender_addr].get(block=block, timeout=timeout if block else 0)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._running = False

    @classmethod
    def clear_directory(cls) -> None:
        pass



class PlainTCPCommsManager:
    """Reliable TCP transport over real (per-namespace) links.

    Unlike the throttled managers, this does NO application-level shaping: the kernel
    plus `tc` on the veth links handle bandwidth, propagation delay, and reliability.
    The transport only provides message framing and per-sender delivery. Same interface
    as InMemoryCommsManager (send_bytes / broadcast_bytes / receive).

    Framing: each message is a 4-byte big-endian length prefix + body. The first frame a
    party sends on a new outbound connection is a handshake carrying its logical "ip:port",
    so the receiver tags its inbox by the sender's stable address (not the ephemeral TCP
    source port). A single select()-based reader services all inbound connections, keeping
    the thread count O(1) per party."""

    LEN_FMT = "!I"
    LEN_SIZE = struct.calcsize("!I")
    SEND_RETRIES = 5             # reconnect+resend attempts on a transient reset before giving up

    def __init__(self, ip: str, port: int, **kwargs):
        self.my_addr = (ip, port)
        self.party_inboxes = _Inboxes()
        self._running = True
        self._reset_log_path = None   # set via set_reset_log(); diagnostic append-log for net faults

        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((ip, port))
        self.lsock.listen(1024)
        self.lsock.setblocking(False)

        self._in_state = {}          # conn -> {'buf': bytearray, 'sender': (ip,port)|None}
        self._out = {}               # target_addr -> socket
        self._out_locks = {}         # target_addr -> Lock
        self._out_guard = threading.Lock()

        # One epoll-based event loop handles accept + all inbound reads. epoll (via
        # selectors.DefaultSelector) has no FD_SETSIZE(1024) limit, unlike select(),
        # so a single process holding thousands of sockets (N up to 100) works.
        self._sel = selectors.DefaultSelector()
        self._sel.register(self.lsock, selectors.EVENT_READ, data="listen")
        self._loop_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._loop_thread.start()

    def _event_loop(self):
        while self._running:
            try:
                events = self._sel.select(timeout=0.5)
            except OSError:
                break
            for key, _ in events:
                if key.data == "listen":
                    self._accept_ready()
                else:
                    self._read_available(key.fileobj)

    def _accept_ready(self):
        while True:
            try:
                conn, _ = self.lsock.accept()
            except (BlockingIOError, OSError):
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setblocking(False)
            self._in_state[conn] = {'buf': bytearray(), 'sender': None}
            self._sel.register(conn, selectors.EVENT_READ, data="conn")

    def _deliver(self, sender_addr, body):
        self.party_inboxes[sender_addr].put(body)

    def set_reset_log(self, path):
        """Point the diagnostic net-fault log at `path` (append mode, created lazily on first fault)."""
        self._reset_log_path = path

    def _log_net(self, msg):
        line = f"[NET t={time.perf_counter():.3f}] {self.my_addr} {msg}\n"
        p = self._reset_log_path
        if p:
            try:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                pass
        else:
            sys.stderr.write(line)

    def _drop_out_addr(self, target_addr):
        """Discard (and close) the cached outbound socket for target_addr so the next send reconnects."""
        with self._out_guard:
            sock = self._out.pop(target_addr, None)
            self._out_locks.pop(target_addr, None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _drop_conn(self, s, reason=""):
        st = self._in_state.pop(s, None)
        if reason:
            who = st.get('sender') if st else None
            self._log_net(f"DROP-IN reason={reason} sender={who}")
        try:
            self._sel.unregister(s)
        except (KeyError, ValueError):
            pass
        try:
            s.close()
        except OSError:
            pass

    def _read_available(self, s):
        st = self._in_state.get(s)
        if st is None:
            return
        try:
            data = s.recv(1 << 20)
        except BlockingIOError:
            return
        except OSError as e:
            self._drop_conn(s, f"recv-oserr {e!r}")
            return
        if not data:
            self._drop_conn(s, "recv-fin(empty)")
            return
        buf = st['buf']
        buf.extend(data)
        while True:
            if len(buf) < self.LEN_SIZE:
                break
            n = struct.unpack(self.LEN_FMT, bytes(buf[:self.LEN_SIZE]))[0]
            if len(buf) < self.LEN_SIZE + n:
                break
            frame = bytes(buf[self.LEN_SIZE:self.LEN_SIZE + n])
            del buf[:self.LEN_SIZE + n]
            if st['sender'] is None:
                try:
                    hip, hport = frame.decode().split(":")
                    st['sender'] = (hip, int(hport))
                except Exception:
                    self._drop_conn(s, "bad-handshake")
                    return
            else:
                self._deliver(st['sender'], frame)

    def _connect(self, target_addr):
        s = self._out.get(target_addr)
        if s is not None:
            return s, self._out_locks[target_addr]
        with self._out_guard:
            s = self._out.get(target_addr)
            if s is not None:
                return s, self._out_locks[target_addr]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for _ in range(600):                     # peer's listener may not be up yet
                try:
                    sock.connect(target_addr)
                    break
                except (ConnectionRefusedError, OSError):
                    time.sleep(0.05)
            hs = f"{self.my_addr[0]}:{self.my_addr[1]}".encode()
            try:
                sock.sendall(struct.pack(self.LEN_FMT, len(hs)) + hs)
            except OSError:
                try:
                    sock.close()
                except OSError:
                    pass
                raise
            lock = threading.Lock()
            self._out[target_addr] = sock
            self._out_locks[target_addr] = lock
            return sock, lock

    def _send_framed(self, data: bytes, target_addr):
        # Resilient send: on a transient reset/broken-pipe, drop the cached socket, reconnect
        # (fresh handshake), and resend the WHOLE frame. Safe because a partial frame lives only
        # in the dropped connection's per-socket buffer and is never delivered (delivery fires on
        # a fully-assembled frame), so no duplicate/torn delivery on the peer.
        packet = struct.pack(self.LEN_FMT, len(data)) + data
        last_err = None
        for attempt in range(self.SEND_RETRIES):
            try:
                sock, lock = self._connect(target_addr)
                with lock:
                    sock.sendall(packet)
                if attempt:
                    self._log_net(f"SEND-OK -> {target_addr} after {attempt} retr{'y' if attempt == 1 else 'ies'}")
                return
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                last_err = e
                self._log_net(f"SEND-RESET -> {target_addr} attempt={attempt} err={e!r}")
                self._drop_out_addr(target_addr)
                time.sleep(0.1 * (attempt + 1))
        self._log_net(f"SEND-GIVEUP -> {target_addr} after {self.SEND_RETRIES} attempts err={last_err!r}")
        raise last_err if last_err is not None else RuntimeError("send failed")

    def send_bytes(self, data: bytes, target_addr):
        self._send_framed(data, target_addr)

    def broadcast_bytes(self, data: bytes, target_addrs):
        if not target_addrs:
            return
        if BROADCAST_PARALLEL:
            senders = [threading.Thread(target=self._send_framed, args=(data, addr))
                       for addr in target_addrs]
            for t in senders:
                t.start()
            for t in senders:
                t.join()
        else:
            for addr in target_addrs:
                self._send_framed(data, addr)

    def receive(self, sender_addr, block: bool = True, timeout: Optional[float] = 30):
        try:
            return self.party_inboxes[sender_addr].get(block=block, timeout=timeout if block else 0)
        except queue.Empty:
            return None

    def close(self):
        self._running = False
        time.sleep(0.02)                     # let the event loop notice and exit its select
        for s in list(self._out.values()):
            try:
                s.close()
            except OSError:
                pass
        for s in list(self._in_state.keys()):
            try:
                s.close()
            except OSError:
                pass
        self._in_state.clear()
        try:
            self._sel.close()
        except Exception:
            pass
        try:
            self.lsock.close()
        except OSError:
            pass

    @classmethod
    def clear_directory(cls):
        pass


# --- Party (Base Class) ---
class Party:
    def __init__(self, 
                 id: int, 
                 ip: str, 
                 port: int, 
                 model_name: str, 
                 dataset_name: str,
                 model: torch.nn.Module,
                 trainer: LocalTrainer,
                 partition_path: str,
                 device: str = 'cpu',
                 dataset_path: str = './dataset',
                 batch_size: int = 32,
                 val_max: int = 0
                 ):
        self.id = id
        # Per-round participant filter. Off (the default) nothing below is consulted: every
        # client takes part in every round, share lists are indexed by client id and the
        # enrollment phase has no cosine round trip. main.py / worker.py turn it on.
        self.participant_filter = False
        self.participants = None       # this round's client ids, sorted, when the filter is on
        self.pos_of = None             # client id -> position in self.participants
        self.participating = True
        _backend = NETWORK_CONFIG.get("backend")
        if _backend == "netns_tcp":
            # Real TCP over per-namespace veth links; bandwidth/latency shaped by tc.
            self.comms = PlainTCPCommsManager(ip, port)
        elif _backend == "udp_throttled":
            self.comms = ThrottledUDPCommsManager(
                ip, port,
                bandwidth_bps=NETWORK_CONFIG.get("bandwidth_bps", 125_000_000),
                latency_s=NETWORK_CONFIG.get("latency_s", 0.0),
            )
        else:
            self.comms = InMemoryCommsManager(ip, port)
        self.device = device
        
        self.model_name = model_name
        self.model = model
        self.trainer = trainer
        
        self.dataset_name = dataset_name
        # Store loader args so the dataloader can be rebuilt with a poisoning config
        # (used only by opt-in attack experiments; no effect on default runs).
        self._partition_path = partition_path
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self.dataloader = CustomDataLoader(dataset_name, model_name, id, partition_path, dataset_path, batch_size)

        self.val_max = val_max

        self._init_logs()

    def _init_logs(self):
        if not hasattr(self, '_time_logs'):
            self._time_logs = {}
        if not hasattr(self, '_comm_logs'):
            self._comm_logs = {'unicast': {}, 'broadcast': {}}

    def extract_and_clear_logs(self, iteration: int) -> Dict[str, Any]:
        self._init_logs()
        result = {
            'party_id': self.id,
            'iteration': iteration,
            'processing_time_ms': self._time_logs,
            'communication_cost_bytes': self._comm_logs
        }
        self._time_logs = {}
        self._comm_logs = {'unicast': {}, 'broadcast': {}}
        return result

    @staticmethod
    def quantize(update_tensor: Tensor, mult) -> List[int]:
        # update_array = update_tensor.tolist()
        # quantized_update_array = [math.floor(x * mult + 0.5) for x in update_array]
        update_tensor = update_tensor.to(torch.float64)
        quantized_update_array = torch.round(update_tensor * float(mult)).tolist()
        quantized_update_array = [int(val) for val in quantized_update_array]
        return quantized_update_array
    
    @staticmethod
    def l2norm(update_array: List[int]) -> int:
        sum_of_squares = 0
        for number in update_array:
            sum_of_squares += number ** 2
        return math.isqrt(sum_of_squares)

    # --- compute lock, lazily re-taken (see the note above _acquire_compute) ---------------
    def _drop_compute(self):
        """Give up the machine before a network operation. Nothing re-takes the lock until the
        next timer stamp, so a whole send or receive loop runs without it."""
        if _SERIALIZE_COMPUTE and not getattr(self, '_lock_dropped', False):
            _release_compute()
            self._lock_dropped = True

    def _clock(self):
        """Wall clock minus the time this party has spent waiting for the compute lock. Also
        the point where a dropped lock is re-taken, so that the wait lands between two stamps
        and is subtracted from whichever interval it fell in."""
        if _SERIALIZE_COMPUTE and getattr(self, '_lock_dropped', False):
            t0 = time.perf_counter()
            _acquire_compute()
            self._lock_dropped = False
            self._lock_wait_s = getattr(self, '_lock_wait_s', 0.0) + (time.perf_counter() - t0)
        return time.perf_counter() - getattr(self, '_lock_wait_s', 0.0)

    def unicast(self, data: Dict[str, Any], receiver_id: int, echo: bool = False, logging: bool = False, logging_info: str = ''):
        if echo:
            if self.id == SERVER_ID:
                print(f"[Server] Unicasting data...")
            else:
                print(f"[Client {self.id}] Unicasting data...")
        self._drop_compute()        # serialization is single-threaded: no need for the machine
        raw_bytes = pickle.dumps(data)

        if logging:
            self._comm_logs['unicast'].setdefault(logging_info, {})[receiver_id] = {
                'size_bytes': len(raw_bytes)
            }

        addr = self.party_map[receiver_id]
        self.comms.send_bytes(raw_bytes, addr)

    def broadcast(self, data: Dict[str, Any], without_server: bool = True, echo: bool = False, logging: bool = False, logging_info: str = '', targets=None):
        if echo:
            if self.id == SERVER_ID:
                print(f"[Server] Broadcasting data...")
            else:
                print(f"[Client {self.id}] Broadcasting data...")
        
        self._drop_compute()        # serialization is single-threaded: no need for the machine
        raw_bytes = pickle.dumps(data)

        # targets=None: everyone in party_map, as before. With the participant filter on, the
        # client-to-client broadcasts pass the round's participant list so a client sitting
        # the round out is neither sent to nor counted in the logged receiver count.
        candidate_ids = list(self.party_map.keys()) if targets is None else list(targets)
        receiver_ids = [id for id in candidate_ids if not (without_server and id == SERVER_ID)]

        if logging:
            self._comm_logs['broadcast'][logging_info]={
                'num_receivers': len(receiver_ids),
                'size_bytes': len(raw_bytes)
            }

        target_addrs = [self.party_map[id] for id in receiver_ids]

        self.comms.broadcast_bytes(raw_bytes, target_addrs)

    def receive(self, id: int, block: bool = True, timeout: float = 30) -> Dict[str, Any]|None:
        addr = self.party_map[id]
        self._drop_compute()        # blocked on the network, then single-threaded unpickling
        data = self.comms.receive(addr, block=block, timeout=timeout)
        if data:
            data = pickle.loads(data)
        return data

    def _sync(self, kind: str = 'all'):
        """E2E-only barrier at a compute->comm boundary. Parks until every participant of the
        upcoming communication region has finished its (serialized) local computation, so all of
        them transmit at the same wall-clock instant instead of the finished ones firing early.
          kind='all'    -> server + N clients (N+1)  : server<->client regions.
          kind='client' -> N clients only (N)        : client<->client regions (server absent).
        No-op outside the E2E parallel run (barriers are unset). The compute lock is released
        while parked so the last still-computing party can finish and reach the barrier."""
        barrier = getattr(self, '_all_barrier', None) if kind == 'all' \
            else getattr(self, '_client_barrier', None)
        if barrier is None:
            return
        self._drop_compute()
        barrier.wait()

    def _peer_send_order(self):
        """Round-robin (rotated) order of the client recipients for all-to-all sends: at step s,
        client i targets recipient (i+s), so every client hits a *distinct* receiver each step.
        Avoids the naive 0..N-1 order in which all clients pile onto one receiver's downlink at
        once (serializing the exchange ~N x). Self is INCLUDED (the protocol sends each client its
        own share too, and the receive loop counts it) and lands at step 0 -- a local loopback
        send, no contention. Correctness is unaffected (order only)."""
        if self.participant_filter and self.participants is not None:
            ids = list(self.participants)                                  # this round's clients incl self
        else:
            ids = sorted(pid for pid in self.party_map if pid != SERVER_ID)   # all N clients incl self
        if self.id in ids:
            k = ids.index(self.id)
            return ids[k:] + ids[:k]         # rotate to start at self; keeps self in the set
        return ids

    def set_VPSS_params(self, pp: vpss.PublicParameters, config1: vpss.VPSSConfig, config2: vpss.VPSSConfig, config1_det: vpss.VPSSConfig = None):
        self.pp = pp
        self.config1 = config1
        self.config2 = config2
        # Deterministic-randomness variant of config1, used only when clients recompute the
        # g_server shares themselves (SERVER_COMPUTES_SHARES=False). Falls back to config1 so
        # the default path is unaffected and no rebuilt C++ config is required.
        self.config1_det = config1_det if config1_det is not None else config1

    # ------------------------------------------------------------------------------
    # Per-round participant set (participant_filter = True only).
    # A client whose cosine with the root update is non-positive gets trust score 0 and so
    # contributes exactly zero to the aggregate; it can sit the round out. The VPSS threshold
    # is derived from the number of participants, so config1/config2 are rebuilt every round
    # from the participant list instead of staying fixed at setup time. With the filter off
    # none of this is called and the configs from set_VPSS_params are used as they are.
    # ------------------------------------------------------------------------------

    def _client_ids(self):
        """The clients of this round: the published participant list with the filter on,
        otherwise everyone in party_map except the server (the original expression)."""
        if self.participant_filter and self.participants is not None:
            return list(self.participants)
        return [id for id in self.party_map if id != SERVER_ID]

    def _pos(self, client_id):
        """Index of a client in the per-round share lists and in config.shareIndices. With
        the filter off that is the client id itself (ids run 0..N-1 in share-index order);
        with it on, the client's position in this round's participant list."""
        if self.participant_filter and self.pos_of is not None:
            return self.pos_of[client_id]
        return client_id

    VARK_MODES = ("", "margin", "opt", "full")

    def set_vpss_shape(self, n_total: int, k1: int, k2: int, threshold_div: int = 3,
                       corrupt_bound: int = 0, vark: str = ""):
        """Remember what is needed to rebuild the configs once a round's participants are
        known. The share index of client i is fixed at n_total + 1 + i (the same points
        main.py hands set_VPSS_params), so a client keeps its evaluation point in every round
        regardless of who else takes part. corrupt_bound is c = ceil(alpha * N), the number of
        malicious clients the round's threshold has to cover; privacy_bound = max(c, 1) is the
        number of share holders that may pool their shares (plan3_cost.privacy_bound). vark chooses the round's k1
        (round_k1_for): "" keeps k1 fixed, "margin" is variable k, plan 1, "opt" is variable k,
        plan 2 (the same k1 rule, with the participant count chosen by choose_round_size),
        "full" is variable k and t, plan 3 (n, k chosen together, t = k + cp; choose_round_shape)."""
        if vark not in Party.VARK_MODES:
            raise ValueError(f"unknown vark mode {vark!r}; one of {Party.VARK_MODES}")
        self.n_total = n_total
        self.k1 = k1
        self.k2 = k2
        self.threshold_div = threshold_div
        self.corrupt_bound = corrupt_bound
        from plan3_cost import privacy_bound
        self.privacy_bound = privacy_bound(corrupt_bound)   # max(c, 1): share holders that may pool
        self.vark = vark

    def round_k1_for(self, n: int) -> int:
        """The first-level packing k1 of a round with n participants (server side; the clients
        get it with the participant list).

        Fixed k1 unless vark == "margin" (variable k, plan 1): the participants are chosen first
        (cosine filter, floor from the fixed k1), then t stays n // 3 and the privacy margin
        left over, t - cp, goes into k1: k1 = max(fixed k1, n // 3 - cp), cp = max(c, 1). The
        robustness margin is left alone, so t never moves away from the filter-off rule.
          N=48, alpha=0.2 (c=cp=10): n=31 -> k1=1 (t=11), n=40 -> k1=3 (t=13), n=48 -> k1=6 (t=16)
          alpha=0 (c=0, cp=1): k1 = n // 3 - 1, e.g. n=48 -> 15 (t=16), n=15 -> 4 (t=5)
        t is what the fixed-k1 run would use at the same n, so only the packing changes.
        Plan 2 ("opt") uses the same rule; what it changes is n (choose_round_size).
        """
        if getattr(self, 'vark', "") not in ("margin", "opt"):
            return self.k1
        base = n // self.threshold_div
        return max(self.k1, base - getattr(self, 'privacy_bound', 1))

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

    def set_round_participants(self, participants, k1=None, t=None):
        """Fix this round's participant list and rebuild the VPSS configs around it. k1 is the
        round's first-level packing as the server chose it (round_k1_for); None keeps the
        fixed k1. t, given only by variable k plan 3, is the round's threshold as the server
        chose it (k1 + cp); None applies the rule below."""
        self.participants = sorted(participants)
        self.pos_of = {pid: idx for idx, pid in enumerate(self.participants)}
        self.participating = (self.id in self.pos_of)
        n = len(self.participants)
        k1 = self.k1 if k1 is None else int(k1)
        self.round_k1 = k1
        k_max = max(k1, self.k2)
        c = getattr(self, 'corrupt_bound', 0)
        cp = getattr(self, 'privacy_bound', max(c, 1))
        # t has to satisfy privacy t - k >= cp (cp = max(c, 1): a holder always sees its own
        # share) and robustness (n - t) / 2 >= c, i.e. lie in [k + cp, n - 2c].
        # n // threshold_div (the filter-off rule, t = N // 3) is used when it lies there and
        # clipped to the nearer end otherwise. k itself may not exceed n // threshold_div. The
        # server's min_participants floor (plan3_cost.filter_floor) makes all of this possible.
        base = n // self.threshold_div
        lo, hi = k_max + cp, n - 2 * c
        if t is not None:
            # plan 3: the server's t, checked against privacy, robustness and the degree bound
            # of the pairwise verification (2t - 1 points for a product of two shares, with up
            # to c clients failing it); plan3_cost.k_max is the same rule seen from k
            t = int(t)
            if not (lo <= t <= hi and 2 * t - 1 <= n - c):
                raise ValueError(f"[Party {self.id}] t={t} is not valid for n={n}, k={k_max}, c={c}, cp={cp}: "
                                 f"needs {lo} <= t <= {hi} and 2t - 1 <= {n - c}")
        elif lo <= hi and k_max <= base:
            t = min(max(max(1, base), lo), hi)
        else:
            t = max(1, base)
            print(f"[Party {self.id}] WARNING: n={n} leaves no t in [k + cp, n - 2c] with k <= n // 3 "
                  f"(k={k_max}, c={c}, cp={cp}).")
            t = max(t, k_max)
        self.round_t = t
        secret_count = max(t, k_max)
        share_indices = [self.n_total + 1 + pid for pid in self.participants]
        self.config1 = Party.build_vpss_config(t, k1, secret_count, share_indices)
        self.config2 = Party.build_vpss_config(t, self.k2, secret_count, share_indices)
        self.config1_det = Party.build_vpss_config(t, k1, secret_count, share_indices,
                                                   deterministic_rand=True)

    def set_poisoning(self, poisoning_config):
        """Opt-in: rebuild this party's dataloader with a poisoning config
        (used only by attack experiments; default/published runs never call this)."""
        self.dataloader = CustomDataLoader(
            self.dataset_name, self.model_name, self.id,
            self._partition_path, self._dataset_path, self._batch_size,
            poisoning_config=poisoning_config,
        )

    def train(self, epoch: int, lr: float):
        # [B5] self.device instead of a hardcoded 'cuda', which fails on a CPU-only host.
        torch.nn.utils.vector_to_parameters(self.weights.to(self.device), self.model.parameters())
        return self.trainer.train_and_get_update(self.dataloader, epoch, lr)

    @staticmethod
    def get_perpendicular_vector(original_vector: torch.Tensor, base_vector: torch.Tensor) -> torch.Tensor:
        original_vector_norm = torch.norm(original_vector)
        base_vector_norm = torch.norm(base_vector)
        if math.isnan(original_vector_norm) or math.isinf(original_vector_norm):
            print("The input vector has abnormal values. Make the return value as zero vector.")
            print(torch.nonzero(torch.isnan(original_vector), as_tuple=False))
            print(torch.nonzero(torch.isinf(original_vector), as_tuple=False))
            return torch.zeros_like(original_vector)

        original_vector = (base_vector_norm/original_vector_norm) * original_vector

        cos = F.cosine_similarity(original_vector, base_vector, dim=0)
        if cos >= 1.0 - FLOAT_PRECISION:
            return torch.zeros_like(original_vector)

        perpendicular_vector = original_vector - cos * base_vector
        perpendicular_vector *= base_vector_norm/torch.norm(perpendicular_vector)
        return perpendicular_vector



# --- Server Class ---
class Server(Party):
    def __init__(self, 
                ip: str,
                port: int, 
                model_name: str, 
                dataset_name: str,
                model: torch.nn.Module,
                trainer: LocalTrainer,
                partition_path: str,
                party_map: Dict[int, Tuple[str, int]] = {}, 
                device: str = "cpu",
                dataset_path: str = './dataset',
                batch_size: int = 32,
                val_max: int = 0
                ):
        super().__init__(SERVER_ID, ip, port, model_name, dataset_name, model, trainer, partition_path, device, dataset_path, batch_size, val_max)
        if party_map == {}:
            self.party_map = {SERVER_ID: (ip, port)}

    def _init_logs(self):
        super()._init_logs()
        if not hasattr(self, '_value_logs'):
            self._value_logs = {}
    
    def extract_and_clear_logs(self, iteration: int) -> Dict[str, Any]:
        result = super().extract_and_clear_logs(iteration)
        result['values'] = self._value_logs
        self._value_logs = {}
        return result

    def train_phase(self, epoch: int, lr: float, logging: bool = False):
        if logging:
            self._init_logs()
            if 'train' not in self._time_logs: self._time_logs['train'] = {}
            t_start = self._clock()

        # print(f"[Server] Train phase starts.")
        self.g_server = self.train(epoch, lr)
        # print(f"[Server] Train finished.")

        if logging:
            t_end = self._clock()
            self._time_logs['train']['process_ms'] = (t_end - t_start) * 1000

    # Participation policy (participant_filter = True only).
    #   min_participants is the lower bound on the round's participant count,
    #   plan3_cost.filter_floor(k, c) = max(k + cp + 2c, 2(k + cp) - 1 + c, 3k) with
    #   k = max(k1, k2), c = ceil(alpha * N) and cp = max(c, 1): the least n for which a
    #   threshold t with t - k >= cp (privacy), (n - t) / 2 >= c (robustness) and
    #   2t - 1 <= n - c (degree reduction) exists, with k within the round's filter-off
    #   threshold n // 3. set_round_participants then picks t = n // 3 clipped into
    #   [k + cp, n - 2c]. Dropping a non-positive-cosine client does not change the aggregate,
    #   so when the cosine filter would leave fewer than that, rejects are added back; they
    #   still contribute zero, they only restore the threshold.
    #
    #   Which of the non-positive-cosine clients are added back is a policy of its own
    #   (addback_strategy). The aggregate does not depend on it, so it decides only who
    #   bears the round's cost, and that is where a malicious client could game a
    #   deterministic rule: with "highest_cos" a client can submit a cosine just below zero
    #   and still be sure to stay in, with an id order the low ids always stay in. The
    #   default is therefore a uniform random draw, from a generator seeded per round
    #   (addback_seed, set by the worker from RVFL_E2E_SEED) so a run can be repeated.
    ADDBACK_STRATEGIES = ("random", "highest_cos")

    def _addback(self, rejects, count, cos_by_id):
        strategy = getattr(self, 'addback_strategy', 'random')
        if strategy == "random":
            seed = getattr(self, 'addback_seed', None)
            rng = random.Random(seed) if seed is not None else random.Random()
            return rng.sample(sorted(rejects), count)
        if strategy == "highest_cos":
            return sorted(rejects, key=lambda cid: (-cos_by_id[cid], cid))[:count]
        raise ValueError(f"unknown addback_strategy {strategy!r}; one of {self.ADDBACK_STRATEGIES}")

    def select_participants(self, cos_by_id):
        if not self.participant_filter:
            return sorted(cos_by_id.keys())
        chosen = sorted(cid for cid, c in cos_by_id.items() if c > 0)
        floor = getattr(self, 'min_participants', 0)
        target = max(len(chosen), floor)
        if getattr(self, 'vark', "") == "opt":
            target = self.choose_round_size(target, len(cos_by_id))
        elif getattr(self, 'vark', "") == "full":
            target = self.choose_round_shape(target, len(cos_by_id))
        if len(chosen) < target:
            rejects = [cid for cid, c in cos_by_id.items() if c <= 0]
            count = min(target - len(chosen), len(rejects))
            chosen = sorted(chosen + self._addback(rejects, count, cos_by_id))
        return chosen

    # Variable k, plan 2 (vark == "opt").
    #   Every positive-cosine client takes part, since leaving one out changes the aggregate.
    #   A non-positive-cosine client contributes zero whether it takes part or not, so adding
    #   such clients back changes only the round's cost. Plan 2 therefore lets the round size n
    #   run from max(positives, floor) up to every client, gives each n the plan-1 packing
    #   k1 = max(k1, n // 3 - cp) with t = n // 3, and takes the n whose predicted round time is
    #   the least (the smallest such n on a tie). The prediction comes from cost_fit, measured
    #   filter-off rounds fitted per client count as  time = A / k + B  (make_cost_model.py):
    #   A and B are interpolated linearly in n between the measured counts and held at the
    #   end values outside them.
    #     e.g. alpha = 0.2 (c = 10), 31 positives: n = 31 -> k1 = 1, n = 36 -> k1 = 2,
    #          n = 48 -> k1 = 6; the n with the least A(n) / k1 + B(n) is used.

    def predict_round_s(self, n, k):
        fit = getattr(self, 'cost_fit', None)
        if not fit:
            raise RuntimeError("variable k plan 2 needs cost_fit (make_cost_model.py output)")
        pts = sorted((int(e[0]), float(e[1]), float(e[2])) for e in fit)
        if n <= pts[0][0]:
            a, b = pts[0][1], pts[0][2]
        elif n >= pts[-1][0]:
            a, b = pts[-1][1], pts[-1][2]
        else:
            for (n0, a0, b0), (n1, a1, b1) in zip(pts, pts[1:]):
                if n0 <= n <= n1:
                    w = (n - n0) / (n1 - n0)
                    a, b = a0 + w * (a1 - a0), b0 + w * (b1 - b0)
                    break
        return a / k + b

    def choose_round_size(self, n_lo, n_hi):
        """The round size plan 2 uses: the n in [n_lo, n_hi] with the least predicted round
        time at its plan-1 packing, the smallest on a tie."""
        best_n, best_s, preds = n_lo, None, {}
        for n in range(n_lo, n_hi + 1):
            s = self.predict_round_s(n, self.round_k1_for(n))
            preds[n] = round(s, 3)
            if best_s is None or s < best_s - 1e-9:
                best_n, best_s = n, s
        self._round_size_choice = {"n_lo": n_lo, "n": best_n,
                                   "pred_s_lo": preds[n_lo], "pred_s": preds[best_n]}
        return best_n

    # Variable k and t, plan 3 (vark == "full").
    #   As in plan 2, every positive-cosine client takes part and non-positive ones may be added
    #   back, which changes only the cost. Plan 3 also moves t: for each round size n in
    #   [n_lo, n_hi] and each k valid at n (plan3_cost.k_max: t = k + cp <= n - 2c,
    #   2t - 1 <= n - c, cp = max(c, 1), no n // 3 cap), the round time is predicted from the
    #   measured plan-3 grid of this c (cost_points, see plan3_cost.predict), and the (n, k)
    #   with the least prediction is used, with t = k + cp (the smallest n, then the smallest
    #   k, on a tie).
    #     e.g. alpha = 0.2 (c = cp = 10), 31 positives: n = 31 allows k = 1 only (t = 11);
    #          n = 48 allows k up to 9 (t = 19); every such pair is compared.

    def choose_round_shape(self, n_lo, n_hi):
        """The round size plan 3 uses; the chosen k and t are kept in _round_shape."""
        import plan3_cost
        pts = getattr(self, 'cost_points', None)
        if not pts:
            raise RuntimeError("variable k plan 3 needs cost_points (make_cost_model.py --plan3 output)")
        points = {(int(e[0]), int(e[1])): float(e[2]) for e in pts}
        c = getattr(self, 'corrupt_bound', 0)
        best, lo_best, unpredicted = None, None, 0
        for n in range(n_lo, n_hi + 1):
            for k in range(max(1, self.k1), plan3_cost.k_max(n, c) + 1):
                s = plan3_cost.predict(points, n, k)
                if s is None:
                    unpredicted += 1
                    continue
                key = (s, n, k)
                if best is None or key < best:
                    best = key
                if n == n_lo and (lo_best is None or key < lo_best):
                    lo_best = key
        if best is None:
            raise RuntimeError(f"variable k plan 3: no predictable (n, k) for n in [{n_lo}, {n_hi}], c={c}")
        s, n, k = best
        cp = plan3_cost.privacy_bound(c)
        self._round_shape = (k, k + cp)
        self._round_size_choice = {"n_lo": n_lo, "n": n, "k": k, "t": k + cp, "pred_s": round(s, 3),
                                   "pred_s_lo": round(lo_best[0], 3) if lo_best else None,
                                   "k_lo": lo_best[2] if lo_best else None,
                                   "unpredicted": unpredicted}
        return n

    def enrollment_phase(self, thread_count: int = 1, logging: bool = False):
        if self.participant_filter:
            # The filtered enrollment is a generator (two exchanges with the clients in
            # between); the caller drives it. This function stays a plain call otherwise.
            return self._enrollment_phase_filtered(thread_count, logging)

        if logging:
            self._time_logs['enrollment'] = {}
            t_start = self._clock()

        # print(f"[Server] Enrollment phase starts.")

        # calculate maximal scaler for preventing overflow during verification phase
        # sum of parameter**2 = ||g||**2 = ||g_server||**2  <= q/2
        # => ||g_server|| <= sqrt(q/2) = val_max
        # put some margin for floating point error
        param_max = torch.norm(self.g_server)
        self.exp = math.log10(self.val_max) - math.log10(param_max) - EXP_MARGIN
        if math.isinf(self.exp):
            raise(f"[Server] power is infinity.\tparam_max: {param_max}")
        if math.isnan(self.exp):
            raise(f"[Server] power is NaN.\tparam_max: {param_max}")
        self.exp = int(self.exp)
        mult = 10 ** self.exp
        # print(f"[Server] l2-norm of g_server(float) : {param_max}")
        # print(f"[Server] exp : {self.exp}")


        # Note: In actual FL scenarios, you can provide only g_server and let the clients compute the shares.
        #       Since g_server is a public value, setting all t_1 - k_1 random values to 0 to calculate
        #       the server's share allows every client to generate the identical share.
        #       In this case, while the computation time for each client during the enrollment phase
        #       approximately doubles, the server's communication cost can be reduced even more significantly.
        #       However, since our experimental environment simulates all clients on a single device, 
        #       this increase in computation cost is a more significant drawback. 
        #       Therefore, we opted for a method where the server performs the dealing directly.
        g_server_quantize = Party.quantize(self.g_server, mult)
        self.g_server_quantize_norm = Party.l2norm(g_server_quantize)
        # print(f"[Server] 12-norm of quantized g_server : {self.g_server_quantize_norm}")

        if SERVER_COMPUTES_SHARES:
            # Default: the server deals the g_server shares and broadcasts them.
            secrets_vector = vpss.BigIntVector()
            for val in g_server_quantize:
                secrets_vector.append(vpss.BigInt(str(val)))
            # print(f"[Server] Making shares.")
            shares, _ = vpss.NoneVPSS.distribute(self.pp, self.config1, secrets_vector, thread_count = thread_count)
        else:
            # Clients recompute the shares themselves from public g_server; the server sends none.
            shares = None

        if logging:
            t_make_server_data = self._clock()
            self._time_logs['enrollment']['make_server_data_ms'] = (t_make_server_data - t_start) * 1000

        # send data to clients
        # print(f"[Server] Send data.")
        data = {
            'g_server'          : self.g_server,
            'exp'               : self.exp,
        }
        if SERVER_COMPUTES_SHARES:
            data['g_server_shares'] = shares
        # print(f"[Server] data size: {len(pickle.dumps(data))/(10**6)} MB")
        self._sync('all')                       # region 1: server + N clients enter comm together
        if logging:
            t_make_server_data = self._clock()   # re-base: exclude sync wait from send timing
        self.broadcast(data, logging=logging, logging_info='server_data')

        if logging:
            t_send_server_data = self._clock()
            self._time_logs['enrollment']['send_server_data_ms'] = (t_send_server_data - t_make_server_data) * 1000


        # sample code for receiving cosine values in advance
        """
        yield

        if logging:
            t_after_yield = self._clock()
        
        # print("[Server] Wait for clients' response.")
        # receive cosine similarity
        target_ids = [id for id in self.party_map if id != SERVER_ID]
        self.cosine_list = {}
        while len(self.cosine_list) < len(target_ids):
            for id in target_ids:
                if id in self.cosine_list:
                    continue

                data = self.receive(id, block=True, timeout=3.0)
                
                if data is not None:
                    if 'cos' in data:
                        self.cosine_list[id] = data['cos']
                    else:
                        pass
        
        if logging:
            t_recv_cos = self._clock()
            self._time_logs['enrollment']['receive_cos_ms'] = (t_recv_cos - t_after_yield) * 1000
            self._value_logs['cos'] = self.cosine_list

        # print(f"[Server] cosine list: {self.cosine_list}")

        yield
        """

    def _enrollment_phase_filtered(self, thread_count: int = 1, logging: bool = False):
        """Enrollment with the participant filter on. Same scaler and g_server broadcast as
        enrollment_phase, then one more round trip: every client reports its cosine, the
        server picks the participants, rebuilds the configs and publishes the list (with the
        g_server shares when the server deals them, since config1 only exists now)."""
        if logging:
            self._time_logs['enrollment'] = {}
            t_start = self._clock()

        param_max = torch.norm(self.g_server)
        self.exp = math.log10(self.val_max) - math.log10(param_max) - EXP_MARGIN
        if math.isinf(self.exp):
            raise ValueError(f"[Server] power is infinity.\tparam_max: {param_max}")
        if math.isnan(self.exp):
            raise ValueError(f"[Server] power is NaN.\tparam_max: {param_max}")
        self.exp = int(self.exp)
        mult = 10 ** self.exp

        g_server_quantize = Party.quantize(self.g_server, mult)
        self.g_server_quantize_norm = Party.l2norm(g_server_quantize)

        if logging:
            t_make_server_data = self._clock()
            self._time_logs['enrollment']['make_server_data_ms'] = (t_make_server_data - t_start) * 1000

        data = {
            'g_server'          : self.g_server,
            'exp'               : self.exp,
        }
        self._sync('all')                       # region 1: server broadcasts g_server, clients receive
        if logging:
            t_make_server_data = self._clock()   # re-base: exclude sync wait from send timing
        self.broadcast(data, logging=logging, logging_info='server_data')

        if logging:
            t_send_server_data = self._clock()
            self._time_logs['enrollment']['send_server_data_ms'] = (t_send_server_data - t_make_server_data) * 1000

        yield

        # --- collect every client's cosine, then publish the participant list -------------
        target_ids = [id for id in self.party_map if id != SERVER_ID]
        self.cos_list = {}
        self._sync('all')                       # region 1b: clients send cos, server receives
        if logging:
            t_recv_cos_start = self._clock()
        while len(self.cos_list) < len(target_ids):
            for id in target_ids:
                if id in self.cos_list:
                    continue
                data = self.receive(id, block=True, timeout=3.0)
                if data is not None and 'cos' in data:
                    self.cos_list[id] = data['cos']

        if logging:
            t_recv_cos = self._clock()
            self._time_logs['enrollment']['receive_cos_ms'] = (t_recv_cos - t_recv_cos_start) * 1000
            self._value_logs['cos_reported'] = dict(self.cos_list)

        participants = self.select_participants(self.cos_list)
        if getattr(self, 'vark', "") == "full":
            round_k1, round_t = self._round_shape          # plan 3: chosen with the round size
        else:
            round_k1, round_t = self.round_k1_for(len(participants)), None
        self.set_round_participants(participants, round_k1, round_t)

        payload = {'participants': participants, 'k1': round_k1, 't': round_t}
        if SERVER_COMPUTES_SHARES:
            secrets_vector = vpss.BigIntVector()
            for val in g_server_quantize:
                secrets_vector.append(vpss.BigInt(str(val)))
            shares, _ = vpss.NoneVPSS.distribute(self.pp, self.config1, secrets_vector,
                                                 thread_count=thread_count)
            payload['g_server_shares'] = shares

        if logging:
            t_select = self._clock()
            self._time_logs['enrollment']['select_participants_ms'] = (t_select - t_recv_cos) * 1000

        self._sync('all')                       # region 1c: server broadcasts the participant list
        if logging:
            t_select = self._clock()       # re-base: exclude sync wait from send timing
        self.broadcast(payload, logging=logging, logging_info='participant_list')

        if logging:
            t_send_list = self._clock()
            self._time_logs['enrollment']['send_participant_list_ms'] = (t_send_list - t_select) * 1000
            self._value_logs['num_participants'] = len(participants)
            self._value_logs['round_t'] = self.round_t
            self._value_logs['round_k1'] = self.round_k1
            self._value_logs['num_positive'] = sum(1 for c in self.cos_list.values() if c > 0)
            if getattr(self, 'vark', "") in ("opt", "full"):
                self._value_logs['round_size_choice'] = self._round_size_choice

        yield

    def verification_phase(self, use_perp: bool = True, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['verification'] = {}
            t_start = self._clock()

        # print(f"[Server] Verification phase starts.")
        
        # print("[Server] Wait for clients' response.")
        # receive the dot product shares    
        target_ids = self._client_ids()
        final_shares_vec1 = vpss.BatchShareNoneVector()
        final_shares_vec2 = vpss.BatchShareNoneVector()
        self.cos_list = {}
        cnt = 0
        self._sync('all')                       # region 5: clients send s_tilde, server receives
        if logging:
            t_start = self._clock()        # re-base: exclude sync wait from receive timing
        while cnt < len(target_ids):
            for id in target_ids:
                if id in self.cos_list:
                    continue

                data = self.receive(id, block=True, timeout=3.0)
                
                if data is not None:
                    if 's_tilde_with_server' in data:
                        final_shares_vec1.append(data['s_tilde_with_server'])
                        final_shares_vec2.append(data['s_tilde_itself'])
                        if use_perp:
                            self.cos_list[id] = data['cos']
                        cnt += 1
                    else:
                        pass

        if logging:
            t_recv_s_tilde = self._clock()
            self._time_logs['verification']['receive_s_tilde_ms'] = (t_recv_s_tilde - t_start) * 1000
        
        # reconstruct and make Trust Score list
        try:
            # reconstruct with error correction
            dot_products_with_server = vpss.NoneVPSS.reconstruct(
                self.pp,
                self.config2,
                final_shares_vec1,
                useErrorCorrection=True,
                thread_count=thread_count
            )
            dot_products_itself = vpss.NoneVPSS.reconstruct(
                self.pp,
                self.config2,
                final_shares_vec2,
                useErrorCorrection=True,
                thread_count=thread_count
            )
            # print(f"[Server] s_dot_i : {dot_products_with_server}\n[Server] i_dot_i : {dot_products_itself}")
            
            # print("[Server] Final Reconstruction Successful!")


            self.TS = {}
            raw_cos = {}          # [B4] cosine before ReLU, for the value log
            ratios_i_dot_i = {}
            ratios_s_dot_i = {}
            s_dot_s = self.g_server_quantize_norm**2
            threshold1 = 0.00005
            threshold2 = 0.00005
            q_int = self.pp.q.to_int()
            q_half = q_int // 2
            # dot_products_* are in participant order; with the filter off that is 0..N-1
            for pos, i in enumerate(self._client_ids()):
                # the result is not from original vector; it's quantized with multiplier
                #  -> should be checked relatively with ||g_server|| and mult
                i_dot_i = dot_products_itself[pos].to_int()
                s_dot_i = dot_products_with_server[pos].to_int()
                # BigInt for VPSS is unsigned int -> number larger than q/2 is negative number
                s_dot_i = s_dot_i - q_int if s_dot_i > q_half else s_dot_i

                ratio1 = i_dot_i/s_dot_s
                ratio2 = s_dot_i/s_dot_s
                ratios_i_dot_i[i] = ratio1
                ratios_s_dot_i[i] = ratio2

                valid = abs(ratio1-1)<threshold1    # ||v|| ?= ||g_server||
                if use_perp:
                    # if used perpendicular vector g_i_perp, check verticality
                    valid = valid and abs(ratio2)<threshold2
                else:
                    # if used original vector g_i, calculate the cosine similarity
                    # cos = <g_i, g_s> / (||g_i||*||g_s||) = <mult*g_i, mult*g_s> / (sqrt(<mult*g_i,mult*g_i>)*||mult*g_s||)
                    cos = s_dot_i / math.isqrt(i_dot_i) / self.g_server_quantize_norm
                    self.cos_list[i] = cos
                raw_cos[i] = self.cos_list[i]
                if(valid):
                    # [B4] ReLU applied IN PLACE, so sum_cos / sum_cos2 in aggregation_phase
                    # use the same convention as the trust score. A client with cos <= 0 then
                    # adds 1 + (2*0 - 1) = 0 to the numerator and 0 to the denominator, which
                    # is what the plaintext path (FLTrust_quantized_perp_agg) already did.
                    self.cos_list[i] = max(self.cos_list[i], 0)    # RELU
                    cos = self.cos_list[i]
                    if use_perp:
                        self.TS[i] = 2 * cos * math.sqrt(1 - cos**2)
                    else:
                        self.TS[i] = cos
                else:
                    # It functions even when set to 0, but is set to -1.0 for explicit differentiation
                    self.TS[i] = -1.0
            # print(f"[Server] ratio of norm square : {ratios}")
            # print(f"[Server] Trust Score (cosine) : {self.TS}")

            if logging:
                t_calc_TS = self._clock()
                self._time_logs['verification']['calc_TS_ms'] = (t_calc_TS - t_recv_s_tilde) * 1000
                self._value_logs['ratios_i_dot_i'] = ratios_i_dot_i
                self._value_logs['ratios_s_dot_i'] = ratios_s_dot_i
                self._value_logs['cos'] = raw_cos          # [B4] log the pre-ReLU values
                self._value_logs['g_server_norm'] = torch.norm(self.g_server).item()
                self._value_logs['exp'] = self.exp

            data = {"TS": self.TS}
            self._sync('all')                   # region 6: server broadcasts TS, clients receive
            if logging:
                t_calc_TS = self._clock()  # re-base: exclude sync wait from send timing
            self.broadcast(data, logging=logging, logging_info='TS', targets=self.participants)

            if logging:
                t_send_TS = self._clock()
                self._time_logs['verification']['send_TS_ms'] = (t_send_TS - t_calc_TS) * 1000

        except Exception as e:
            print(f"Reconstruction Failed: {e}")
    
    def aggregation_phase(self, lr_global: float, use_error_correction: bool = False, use_perp: bool = True, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = self._clock()

        # 1. receive the aggregated shares from all clients
        target_ids = self._client_ids()
        collected_shares = vpss.BatchShareNoneVector()
        
        # print("[Server] Collecting aggregated shares...")
        cnt = 0
        self._sync('all')                       # region 7a: clients send agg_share, server receives
        if logging:
            t_start = self._clock()        # re-base: exclude sync wait from receive timing
        while cnt < len(target_ids):
            for id in target_ids:
                data = self.receive(id, True, None)
                if data and 'aggregated_share' in data:
                    collected_shares.append(data['aggregated_share'])
                    cnt += 1

        if logging:
            t_recv_agg_share = self._clock()
            self._time_logs['aggregation']['receive_agg_share_ms'] = (t_recv_agg_share - t_start) * 1000

        # 2. Reconstruction -> g_sum (Quantized)
        try:
            # reconstruct vector
            reconstructed_vec = vpss.NoneVPSS.reconstruct(
                self.pp, self.config1, collected_shares, use_error_correction, thread_count=thread_count
            )

        except Exception as e:
            print(f"[Server] Reconstruction failed: {e}")
            return

        # 3. dequantization (BigInt -> Float Tensor)
        # TS_SCALE should be lower than math.log10(self.val_max/len(self.TS))
        TS_SCALE = 10**TS_EXP
        QUANT_SCALE = 10 ** self.exp
        TOTAL_SCALE = TS_SCALE * QUANT_SCALE
        
        q_int = self.pp.q.to_int()
        q_half = q_int // 2
        g_sum_list = []

        for val_bigint in reconstructed_vec:
            val_int = val_bigint.to_int()
            if val_int > q_half:
                val_int -= q_int
            
            g_sum_list.append(val_int / TOTAL_SCALE)

        g_sum = torch.tensor(g_sum_list, device='cpu')
        g_sum = g_sum[:self.g_server.size(0)]

        #print(f"[Server] cosine similarity bet. g_server and g_sum: {F.cosine_similarity(self.g_server, g_sum, dim=0)}")
        #print(f"[Server] l2-norm of g_server: {torch.linalg.norm(self.g_server, ord=2)}")
        #print(f"[Server] l2-norm of g_sum: {torch.linalg.norm(g_sum, ord=2)}")

        # 4. calculate the global update
        # Formula: g = g_sum / sum_cos = (sum of cos*g_i) / (sum_cos) = ( g_S * (|I| + sum_cos2) + g_perp ) / (2 * sum_cos)
        
        valid_indices = [i for i, s in self.TS.items() if s >= 0]
        I_size = len(valid_indices) # |I|
        # [B4] cos_list has already been passed through ReLU in verification_phase.
        sum_cos = sum(self.cos_list[i] for i in valid_indices)

        # [B4] Guard on sum_cos rather than I_size: every client can pass verification and
        # still have a non-positive cosine, which leaves I_size > 0 with a zero denominator.
        if sum_cos == 0:
            # print("[Server] No positive cosine. Use g_S only.")
            final_g = self.g_server.cpu()
        else:
            if use_perp:
                sum_cos2 = sum((2 * (self.cos_list[i]**2) - 1) for i in valid_indices) 
                
                term1 = self.g_server.cpu() * (I_size + sum_cos2)            
                numerator = term1 + g_sum
                
                denominator = 2 * sum_cos

                final_g = numerator / denominator
            else:
                final_g = g_sum / sum_cos

        print(f"[Server] l2-norm of final gradient: {torch.norm(final_g)}")

        # if torch.isnan(final_g).any():
        #     print("[Server] Final gradient has some NaN parameter. Update has aborted.")
            

        # 5. apply to model
        self.weights += final_g * lr_global
        # print("[Server] Final global model updated.")

        if logging:
            t_update_global_weights = self._clock()
            self._time_logs['aggregation']['update_global_weights_ms'] = (t_update_global_weights - t_recv_agg_share) * 1000
            self._value_logs['final_g_norm'] = torch.norm(final_g).item()

        data = {
            'global_weights': self.weights,
        }

        # print(f"[Server] Broadcasting new global weights to all clients...")
        self._sync('all')                       # region 7b: server broadcasts weights, clients receive
        if logging:
            t_update_global_weights = self._clock()  # re-base: exclude sync wait from send timing
        self.broadcast(data, logging=logging, logging_info='updated_weights')

        if logging:
            t_send_updated_global_weights = self._clock()
            self._time_logs['aggregation']['send_updated_global_weights_ms'] = (t_send_updated_global_weights - t_update_global_weights) * 1000

    def FLTrust_agg(self, lr_global: float, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = self._clock()

        # print(f"[Server] FLTrust aggregation.")

        target_ids = [id for id in self.party_map if id != SERVER_ID]
        
        # print("[Server] Collecting local updates...")
        cnt = 0
        updates = {}
        while cnt < len(target_ids):
            for id in target_ids:
                data = self.receive(id, True, None)
                if data and 'g_i' in data:
                    updates[id] = data['g_i']
                    cnt += 1

        if logging:
            t_recv_local_update = self._clock()
            self._time_logs['aggregation']['receive_local_update_ms'] = (t_recv_local_update - t_start) * 1000

        cos_sum = 0
        g_server_norm = torch.norm(self.g_server)
        final_g = torch.zeros_like(self.g_server)
        cos_list = []
        for id in updates:
            cos = float(F.cosine_similarity(updates[id], self.g_server, dim=0))
            cos_list.append(cos)
        #     if cos > 0 :
        #         cos_sum += cos
        #         final_g += (g_server_norm / torch.norm(updates[id]) * cos) * updates[id]
        # final_g /= cos_sum
            if cos > 0 :
                cos_sum += cos
                final_g += (g_server_norm / (torch.norm(updates[id]) + 1e-8) * cos) * updates[id]
        if cos_sum > 1e-8:
            final_g /= cos_sum
        else:
            final_g = torch.zeros_like(self.g_server)
        # print(f"cosine similarities : {cos_list}")

        self.weights += final_g * lr_global

        if logging:
            t_calc_updated_weights = self._clock()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000
            self._value_logs['cos'] = cos_list
            self._value_logs['final_g_norm'] = torch.norm(final_g).item()
            self._value_logs['g_server_norm'] = torch.norm(self.g_server).item()

        # print("[Server] Final global model updated.")
        data = {
            'global_weights': self.weights,
        }        
        # print(f"[Server] Broadcasting new global weights to all clients...")
        self.broadcast(data, logging=logging, logging_info='updated_weights')

        if logging:
            t_calc_updated_weights = self._clock()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000

    def FedAvg_agg(self, lr_global: float, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = self._clock()
    
        # print(f"[Server] FedAvg aggregation.")

        target_ids = [id for id in self.party_map if id != SERVER_ID]
        
        # print("[Server] Collecting local updates...")
        cnt = 0
        updates = {}
        while cnt < len(target_ids):
            for id in target_ids:
                data = self.receive(id, True, None)
                if data and 'g_i' in data:
                    updates[id] = data['g_i']
                    cnt += 1

        if logging:
            t_recv_local_update = self._clock()
            self._time_logs['aggregation']['receive_local_update_ms'] = (t_recv_local_update - t_start) * 1000

        final_g = torch.zeros_like(self.g_server)
        for id in updates:
            final_g += updates[id]
        final_g /= cnt

        cos_list = []
        for id in updates:
            cos = float(F.cosine_similarity(updates[id], final_g, dim=0))
            cos_list.append(cos)
        # norm = torch.norm(final_g).item()
        # print(f"norm of g_avg : {norm}\ncosine similarities : {cos_list}")

        self.weights += final_g * lr_global

        if logging:
            t_calc_updated_weights = self._clock()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000
            self._value_logs['cos'] = cos_list
            self._value_logs['final_g_norm'] = torch.norm(final_g).item()
            self._value_logs['g_server_norm'] = torch.norm(self.g_server).item()

        # print("[Server] Final global model updated.")
        data = {
            'global_weights': self.weights,
        }        
        # print(f"[Server] Broadcasting new global weights to all clients...")
        self.broadcast(data, logging=logging, logging_info='updated_weights')

        if logging:
            t_send_updated_weights = self._clock()
            self._time_logs['aggregation']['send_updated_weights_ms'] = (t_send_updated_weights - t_calc_updated_weights) * 1000

    def FLTrust_quantized_agg(self, lr_global: float, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = self._clock()

        target_ids = [id for id in self.party_map if id != SERVER_ID]
        
        cnt = 0
        updates = {}
        while cnt < len(target_ids):
            for id in target_ids:
                data = self.receive(id, True, None)
                if data and 'g_i' in data:
                    updates[id] = data['g_i']
                    cnt += 1

        if logging:
            t_recv_local_update = self._clock()
            self._time_logs['aggregation']['receive_local_update_ms'] = (t_recv_local_update - t_start) * 1000
        
        param_max = torch.norm(self.g_server)
        self.exp = math.log10(self.val_max) - math.log10(param_max) - EXP_MARGIN
        if math.isinf(self.exp):
            raise(f"[Server] power is infinity.\tparam_max: {param_max}")
        if math.isnan(self.exp):
            raise(f"[Server] power is NaN.\tparam_max: {param_max}")
        self.exp = int(self.exp)
        mult = 10 ** self.exp

        g_server_quantize = Party.quantize(self.g_server, mult)

        sum_cos = 0
        g_server_norm = torch.norm(self.g_server)
        final_g = [0] * self.g_server.numel()
        cos_list = {}
        _cos_list = {}
        for id in updates:
            cos = float(F.cosine_similarity(updates[id], self.g_server, dim=0))
            _cos_list[id] = cos
            if cos > 0:
                cos_list[id] = cos
                sum_cos += cos
        
        if len(cos_list) != 0:
            TS_SCALE = 10**TS_EXP
            for id in cos_list:
                scaled_update = (g_server_norm / torch.norm(updates[id])) * updates[id]
                quantized_update = Party.quantize(scaled_update, mult)
                quantized_cos = int(cos_list[id] * TS_SCALE)

                quantized_update = [val * quantized_cos for val in quantized_update]

                final_g = [final_g[i]+quantized_update[i] for i in range(len(final_g))]
            
            q_int = self.pp.q.to_int()
            if max(final_g) > q_int:
                raise ValueError("Too large scaler. An element in final_g exceeds q.")
            div = sum_cos * mult * TS_SCALE
            final_g = [val / div for val in final_g]
            final_g = torch.tensor(final_g)
        else:
            final_g = self.g_server

        self.weights += final_g * lr_global

        if logging:
            t_calc_updated_weights = self._clock()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000
            self._value_logs['g_server_norm'] = torch.norm(self.g_server).item()
            self._value_logs['final_g_norm'] = torch.norm(final_g).item()
            self._value_logs['cos'] = _cos_list

        # print("[Server] Final global model updated.")
        data = {
            'global_weights': self.weights,
        }        
        # print(f"[Server] Broadcasting new global weights to all clients...")
        self.broadcast(data, logging=logging, logging_info='updated_weights')

        if logging:
            t_calc_updated_weights = self._clock()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000

    def FLTrust_quantized_perp_agg(self, lr_global: float, attack=None, malicious_ids=None,
                                   logging: bool = False):
        # Unlike plaintext FLTrust (which sees every g_i and normalizes/cosine/ReLU directly),
        # OUR server never sees a plaintext update -- it can only check DOT PRODUCTS on the
        # client-submitted perpendicular vector v: magnitude <v,v> ?= ||g_S||^2 and orthogonality
        # <v,g_S> ?= 0. A submission failing either is DISCARDED (trust score -1), exactly as the
        # RVFL verification phase does. That is why an over-scaled Scaling submission (||v||=lambda*||g_S||)
        # is rejected here rather than silently normalized.
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = self._clock()

        target_ids = [id for id in self.party_map if id != SERVER_ID]
        
        cnt = 0
        updates = {}
        while cnt < len(target_ids):
            for id in target_ids:
                data = self.receive(id, True, None)
                if data and 'g_i' in data:
                    updates[id] = data['g_i']
                    cnt += 1

        if logging:
            t_recv_local_update = self._clock()
            self._time_logs['aggregation']['receive_local_update_ms'] = (t_recv_local_update - t_start) * 1000
        
        param_max = torch.norm(self.g_server)
        self.exp = math.log10(self.val_max) - math.log10(param_max) - EXP_MARGIN
        if math.isinf(self.exp):
            raise(f"[Server] power is infinity.\tparam_max: {param_max}")
        if math.isnan(self.exp):
            raise(f"[Server] power is NaN.\tparam_max: {param_max}")
        self.exp = int(self.exp)
        mult = 10 ** self.exp

        g_server_quantize = Party.quantize(self.g_server, mult)

        sum_cos = 0
        g_server_norm = torch.norm(self.g_server)
        g_perp = [0] * self.g_server.numel()
        cos_list = {}
        _cos_list = {}
        I_size = 0
        # --- dot-product verification (server sees only inner products, not plaintext) ---
        VERIFY_THR = 1e-3                       # tolerance on ratio1/ratio2 (plaintext float)
        gS_sq = float(g_server_norm ** 2)
        _mal = set(malicious_ids or [])
        _is_scaling = (attack == "scaling_fltrust")
        _rejected = []
        for id in updates:
            cos = float(F.cosine_similarity(updates[id], self.g_server, dim=0))
            _cos_list[id] = cos
            perp = Party.get_perpendicular_vector(updates[id], self.g_server)   # client-normalized to ||g_S||
            if _is_scaling and id in _mal:
                # Scaling attacker submits an UN-normalized (x lambda=n) vector to try to dominate.
                perp = perp * len(target_ids)
            updates[id] = perp
            ratio1 = float(torch.dot(perp, perp)) / gS_sq              # ||v||^2 / ||g_S||^2  (1 if valid)
            ratio2 = float(torch.dot(perp, self.g_server)) / gS_sq     # <v,g_S> / ||g_S||^2  (0 if valid)
            if abs(ratio1 - 1.0) > VERIFY_THR or abs(ratio2) > VERIFY_THR:
                _rejected.append(id)            # TS = -1: fails magnitude/orthogonality -> discarded
                continue
            if cos > 0:
                cos_list[id] = cos
                sum_cos += cos
        if _rejected:
            print(f"[Server] verification rejected {len(_rejected)} update(s) "
                  f"(magnitude/orthogonality fail): {_rejected}")

        if len(cos_list) != 0:
            TS_SCALE = 10**TS_EXP
            for id in cos_list:
                scaled_update = (g_server_norm / torch.norm(updates[id])) * updates[id]
                quantized_update = Party.quantize(scaled_update, mult)
                sin2 = 2 * cos_list[id] * math.sqrt(1 - cos_list[id]**2)
                quantized_sin2 = int(sin2 * TS_SCALE)

                quantized_update = [val * quantized_sin2 for val in quantized_update]

                g_perp = [g_perp[i]+quantized_update[i] for i in range(len(g_perp))]
            
            q_int = self.pp.q.to_int()
            if max(g_perp) > q_int:
                raise ValueError("Too large scaler. An element in g_perp exceeds q.")
            div = mult * TS_SCALE
            g_perp = [val / div for val in g_perp]
            g_perp = torch.tensor(g_perp)

            I_size = len(cos_list)
            sum_cos2 = sum((2 * (cos_list[id]**2) - 1) for id in cos_list)
            final_g = self.g_server * (I_size + sum_cos2) + g_perp
            final_g /= 2 * sum_cos
        else:
            final_g = self.g_server

        self.weights += final_g * lr_global

        if logging:
            t_calc_updated_weights = self._clock()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000
            self._value_logs['g_server_norm'] = torch.norm(self.g_server).item()
            self._value_logs['final_g_norm'] = torch.norm(final_g).item()
            self._value_logs['cos'] = _cos_list

        # print("[Server] Final global model updated.")
        data = {
            'global_weights': self.weights,
        }        
        # print(f"[Server] Broadcasting new global weights to all clients...")
        self.broadcast(data, logging=logging, logging_info='updated_weights')

        if logging:
            t_calc_updated_weights = self._clock()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000

    def BREA(self, lr_global: float, byzantine_rate: float = 0.3, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['BREA'] = {}
            t_start = self._clock()

        # receive the share of pairwise distances 
        target_ids = [id for id in self.party_map if id != SERVER_ID]
        x = {}
        d = {}
        cnt = 0
        n = len(target_ids)
        while cnt < n:
            for id in target_ids:
                if id in x:
                    continue

                data = self.receive(id, block=True, timeout=3.0)
                
                if data is not None:
                    if 'd' in data:
                        x[id] = data['x']
                        d[id] = data['d']
                        cnt += 1
                    else:
                        pass
        
        if logging:
            t_recv_share = self._clock()
            self._time_logs['BREA']['receive_share_ms'] = (t_recv_share - t_start) * 1000
        
        # reconstruct
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
                    except:
                        print(f"Party {i} dont have ({j},{k})")
                        d_i.s_values.append(vpss.BigInt('0'))
            d_vec.append(d_i)
        d_bar = vpss.NoneVPSS.reconstruct(
            self.pp,
            self.config2,
            d_vec,
            useErrorCorrection=True,
            thread_count=thread_count
        )

        if logging:
            t_reconstruct_d = self._clock()
            self._time_logs['BREA']['reconstruct_d_ms'] = (t_reconstruct_d - t_recv_share) * 1000

        # dequantization
        mult = 10 ** (BREA_EXP*2)
        q_int = self.pp.q.to_int()
        q_half = q_int // 2
        d_bar_list = []

        for val_bigint in d_bar:
            val_int = val_bigint.to_int()
            if val_int > q_half:
                val_int -= q_int
            d_bar_list.append(val_int / mult)

        if logging:
            t_dequantize_d = self._clock()
            self._time_logs['BREA']['dequantize_d_ms'] = (t_dequantize_d - t_reconstruct_d) * 1000
            self._value_logs['d_bar'] = d_bar_list


        # select by Multi-Krum
        pairwise_distances = [[0.0]*n for _ in range(n)]
        index = 0
        for j in range(n):
            for k in range(j+1,n):
                if j==k:
                    continue
                pairwise_distances[j][k] = d_bar_list[index]
                pairwise_distances[k][j] = d_bar_list[index]
                index += 1
        a = math.floor(n * byzantine_rate)
        m = n - 2*a - 3
        multi_krum = MultiKrum(pairwise_distances, m, a)
        selected = multi_krum.select()
        S = [target_ids[i] for i in selected]
        # print(f"pairwise_distance:\n{pairwise_distances}\nselected by Multi-KRUM: {selected}")

        if logging:
            t_multi_krum = self._clock()
            self._time_logs['BREA']['multi_krum_ms'] = (t_multi_krum - t_dequantize_d) * 1000
            self._value_logs['A'] = a
            self._value_logs['m'] = m
            self._value_logs['selected set'] = selected

        # send selected set
        data = {'S': S}
        self.broadcast(data, logging = logging, logging_info='S')

        if logging:
            t_send_set = self._clock()
            self._time_logs['BREA']['send_set_ms'] = (t_send_set - t_multi_krum) * 1000

        yield

        if logging:
            t_after_yield_from = self._clock()

        # receive share of aggregated update
        aggregated_share = vpss.BatchShareNoneVector()
        cnt = 0
        while cnt < n:
            for id in target_ids:
                data = self.receive(id, block=True, timeout=None)
                if data is not None:
                    if 'aggregated_share' in data:
                        aggregated_share.append(data['aggregated_share'])
                        cnt += 1
                    else:
                        pass

        if logging:
            t_receive_agg = self._clock()
            self._time_logs['BREA']['receive_agg_ms'] = (t_receive_agg - t_after_yield_from) * 1000
        
        # reconstruct
        reconstructed_vec = vpss.NoneVPSS.reconstruct(
            self.pp,
            self.config1,
            aggregated_share,
            useErrorCorrection=True,
            thread_count=thread_count
        )

        if logging:
            t_reconstruct_agg = self._clock()
            self._time_logs['BREA']['reconstruct_agg_ms'] = (t_reconstruct_agg - t_receive_agg) * 1000

        # dequantize and update weights
        mult = 10 ** BREA_EXP
        g_sum_list = []
        for val_bigint in reconstructed_vec:
            val_int = val_bigint.to_int()
            if val_int > q_half:
                val_int -= q_int
            g_sum_list.append(val_int / mult)
        g_sum = torch.tensor(g_sum_list, device='cpu')
        g_sum = g_sum[:self.weights.size(0)]

        self.weights += g_sum * lr_global

        if logging:
            t_dequantize_and_update = self._clock()
            self._time_logs['BREA']['dequantize_and_update_ms'] = (t_dequantize_and_update - t_reconstruct_agg) * 1000
            self._value_logs['final_g_norm'] = torch.norm(g_sum).item()

        data = {
            'global_weights': self.weights,
        }
        self.broadcast(data, logging=logging, logging_info='updated_weights')

        if logging:
            t_send_updated_global_weights = self._clock()
            self._time_logs['BREA']['send_updated_global_weights_ms'] = (t_send_updated_global_weights - t_dequantize_and_update) * 1000

        yield


        
        
            



# --- Client Class ---
class Client(Party):
    def __init__(
            self,
            id: int, 
            ip: str, 
            port: int, 
            server_addr: Tuple[str, int], 
            model_name: str, 
            dataset_name: str,
            model: torch.nn.Module,
            trainer: LocalTrainer,
            partition_path: str,
            device: str = "cpu",
            dataset_path: str = './dataset',
            batch_size: int = 32,
            val_max: int = 0
            ):
        super().__init__(id, ip, port, model_name, dataset_name, model, trainer, partition_path, device, dataset_path, batch_size, val_max)
        self.party_map = {SERVER_ID: server_addr}
    
    def pairwise_verification(self, first_level_shares, first_level_shares_by_dealer, thread_count: int = 1, logging: bool = False):
        if logging:
            t_start = self._clock()

        # 1. Deal second-level share

        # print(f"[Client {self.id}] Pairwise Verification Starts.")
        # print(f"[Client {self.id}] first-level-shares: {first_level_shares}")
        shares1, commitments1, blinds1 = vpss.PedersenVPSS.distribute(self.pp, self.config2, first_level_shares[0], thread_count = thread_count, return_blinds = True)
        shares2, commitments2, blinds2 = vpss.PedersenVPSS.distribute(self.pp, self.config2, first_level_shares[1], thread_count = thread_count, return_blinds = True)

        if logging:
            t_make_second_level_share = self._clock()
            self._time_logs['verification']['make_second_level_share_ms'] = (t_make_second_level_share - t_start) * 1000

        # print(f"[Client {self.id}] Send second-level shares and commitments.")
        self._sync('client')                    # region 3: clients exchange 2nd-level shares + commitments (N)
        if logging:
            t_make_second_level_share = self._clock()   # re-base: exclude sync wait from send timing
        target_ids = self._peer_send_order()    # round-robin: avoid downlink pile-up
        data = {'commitment1': commitments1, 'commitment2': commitments2}
        self.broadcast(data, logging = logging, logging_info='commitment', targets=self.participants)
        for id in target_ids:
            if id==SERVER_ID:
                continue
            pos = self._pos(id)
            data = {
                'share1': shares1[pos],
                'blind1': blinds1[pos],
                'share2': shares2[pos],
                'blind2': blinds2[pos],
                }
            self.unicast(data, id, logging = logging, logging_info='second_level_share')

        if logging:
            t_send_second_level_share = self._clock()
            self._time_logs['verification']['send_second_level_share_ms'] = (t_send_second_level_share - t_make_second_level_share) * 1000

        yield

        if logging:
            t_after_yield = self._clock()

        # print(f"[Client {self.id}] Receive second-level shares and commitments.")
        second_level_shares1 = {}
        blinds_for_verification1 = {}
        received_commitments1 = {}
        second_level_shares2 = {}
        blinds_for_verification2 = {}
        received_commitments2 = {}
        target_ids = self._client_ids()
        while len(second_level_shares1) < len(target_ids) or \
              len(received_commitments1) < len(target_ids) or \
              len(second_level_shares2) < len(target_ids) or \
              len(received_commitments2) < len(target_ids):
            
            for id in target_ids:
                if id in second_level_shares1 and id in received_commitments1:
                    continue
                try:
                    data = self.receive(id, True, 1.0)
                    if 'share1' in data:
                        #print(f"[Client {self.id}] Share from client {id} received.")
                        second_level_shares1[id] = data['share1']
                        blinds_for_verification1[id] = data['blind1']
                        second_level_shares2[id] = data['share2']
                        blinds_for_verification2[id] = data['blind2']
                    if 'commitment1' in data:
                        #print(f"[Client {self.id}] Commitment from client {id} received.")
                        received_commitments1[id] = data['commitment1']
                        received_commitments2[id] = data['commitment2']
                except Exception as e:
                    pass

        if logging:
            t_receive_second_level_share = self._clock()
            self._time_logs['verification']['receive_second_level_share_ms'] = (t_receive_second_level_share - t_after_yield) * 1000

        # 2. Local verificaiton

        # print(f"[Client {self.id}] Verify shares.")
        local_verdict_vector = {id: True for id in target_ids}
        k2 = self.config2.k
        my_user_index = self._pos(self.id)     # packed secrets are laid out in participant order
        block_idx = my_user_index // k2
        idx_in_block = my_user_index % k2

        for id in target_ids:
            try:
                # --------------------------------------------------------------
                # A. consistency check
                # --------------------------------------------------------------
                consistency_check = vpss.PedersenVPSS.verify_batch_share(
                    self.pp,
                    second_level_shares1[id],
                    received_commitments1[id],
                    thread_count=thread_count
                ) and vpss.PedersenVPSS.verify_batch_share(
                    self.pp,
                    second_level_shares2[id],
                    received_commitments2[id],
                    thread_count=thread_count
                )

                if not consistency_check:
                    # print(f"[Client {self.id}][Verify Fail] Batch verification failed for Sender {id}")
                    local_verdict_vector[id] = False
                    continue

                # --------------------------------------------------------------
                # B. owner check
                # --------------------------------------------------------------
                
                # target commitment block (Vector<BigInt>)
                target_commitment_poly1 = received_commitments1[id][block_idx]
                target_commitment_poly2 = received_commitments2[id][block_idx]

                # target share (BigInt)
                x = self.config2.secretIndices[idx_in_block]
                s1 = first_level_shares_by_dealer[0][self._pos(id)]
                r1 = blinds_for_verification1[id]
                s2 = first_level_shares_by_dealer[1][self._pos(id)]
                r2 = blinds_for_verification2[id]

                # verify
                owner_check = vpss.verify_Pedersen(
                    self.pp,
                    x,
                    s1,
                    r1,
                    target_commitment_poly1
                ) and vpss.verify_Pedersen(
                    self.pp,
                    x,
                    s2,
                    r2,
                    target_commitment_poly2
                )

                if not owner_check:
                    # print(f"[Client {self.id}][Verify Fail] Consistency check failed for Sender {id}")
                    local_verdict_vector[id] = False
                    continue

            except Exception as e:
                print(f"[Client {self.id}][Error] Verification exception for Sender {id}: {e}")
                local_verdict_vector[id] = False
        
        if logging:
            t_verify_second_level_share = self._clock()
            self._time_logs['verification']['verify_second_level_share_ms'] = (t_verify_second_level_share - t_receive_second_level_share) * 1000

        # 3. Broadcast verdict vector

        # print(f"[Client {self.id}] Open verify results.")
        # print(f"[Client {self.id}] Verification result: {my_verification_result}")
        self._sync('client')                    # region 4: clients exchange verdict vectors (N)
        if logging:
            t_verify_second_level_share = self._clock()   # re-base: exclude sync wait from send timing
        self.broadcast({'verdict_vector': local_verdict_vector}, logging = logging, logging_info='verification_result', targets=self.participants)

        if logging:
            t_send_verify_result = self._clock()
            self._time_logs['verification']['send_verify_result_ms'] = (t_send_verify_result - t_verify_second_level_share) * 1000

        yield

        if logging:
            t_after_yield = self._clock()

        # 4. Make a global verdict vector

        target_ids = self._client_ids()
        global_verdict_vector = {id: True for id in target_ids}
        received_ids = set()
        while len(received_ids) < len(target_ids):
            for id in target_ids:
                if id in received_ids:                 # skip already-collected peers so an
                    continue                           # empty inbox never re-blocks the timeout
                data = self.receive(id, True, 1.0)
                if data is not None:
                    if 'verdict_vector' in data:
                        other_vec = data['verdict_vector']
                        for target_id in global_verdict_vector:
                            if target_id in other_vec:
                                global_verdict_vector[target_id] = global_verdict_vector[target_id] and other_vec[target_id]
                        received_ids.add(id)
        # if self.id == 0:
        #     print(f"[Client {self.id}] Integrated verification result: {global_verdict_vector}")

        if logging:
            t_receive_verify_result = self._clock()
            self._time_logs['verification']['receive_verify_result_ms'] = (t_receive_verify_result - t_after_yield) * 1000

        qualified_set = [id for id, is_valid in global_verdict_vector.items() if is_valid]
        qualified_set.sort()
        
        # 5. filter honest second-level share
        honest_second_level_shares1 = {
            id: second_level_shares1[id] 
            for id in qualified_set 
            if id in second_level_shares1
        }
        honest_second_level_shares2 = {
            id: second_level_shares2[id] 
            for id in qualified_set 
            if id in second_level_shares2
        }

        if logging:
            t_filter_honest_clients = self._clock()
            self._time_logs['verification']['filter_honest_clients_ms'] = (t_filter_honest_clients - t_receive_verify_result) * 1000

        return qualified_set, honest_second_level_shares1, honest_second_level_shares2
    
    def transposed_degree_reduction(self, honest_set, honest_second_level_shares):
        """
        Algorithm: Transposed Degree Reduction
        Output: Reduced Share Matrix S_tilde (Block_size x k1)
        """

        # print(f"[Client {self.id}] Transposed Degree Reduction Starts.")
        # 1. Configuration & Constants
        t1 = self.config1.t
        k1 = self.config1.k
        q = self.pp.q
        I_len = len(honest_set)
        
        # 2. Check the number of honest parties. The rows being reduced are shares of a product
        # of two degree-(t1 - 1) polynomials, degree 2(t1 - 1), so 2*t1 - 1 points are needed;
        # with fewer the interpolation below still runs but returns wrong values (seen at n=7,
        # t=5 and n=9, t=6), so this stops the round instead.
        if I_len < 2 * t1 - 1:
            raise ValueError(f"Not enough honest parties for degree reduction: need 2t-1 = "
                             f"{2 * t1 - 1} (t={t1}), got {I_len}")
        
        # 3. Prepare Coordinates (alpha and e)
        # alpha: x-coordinates of the selected honest parties
        # Note: config1.shareIndices is 0-indexed vector corresponding to Party 1..N
        # If party ID starts at 1, verify config indices mapping. 
        # Assuming: Party ID k maps to shareIndices[k-1] (or check your VPSS config logic)
        alpha_vec = vpss.BigIntVector()
        for id in honest_set:
            # shareIndices holds this round's clients in order: position, not id, when filtering
            alpha_vec.append(self.config1.shareIndices[self._pos(id)])
            
        # e: secret indices (where secrets are embedded)
        # Use first t1 indices to create a square matrix for V_e (we will truncate later)
        # Or simply use all k1 indices if creating rectangular.
        # VPSS createVandermonde creates N x (degree+1).
        # We need V_e to be t1 x t1 for consistency in matrix chain, then project.
        # So we pass t1 secret indices (padding if k1 < t1).
        e_vec = vpss.BigIntVector()
        for i in range(t1):
            e_vec.append(self.config1.secretIndices[i])

        # 4. Construct Transformation Matrices (C++ Operations)

        # V_alpha (Size: |I| x |I|)
        V_alpha = vpss.create_vandermonde_matrix(alpha_vec, I_len-1, q)
        
        # V_e (Size: t1 x |I|)
        V_e = vpss.create_vandermonde_matrix(e_vec, I_len-1, q)

        # Invert (V_alpha^-1)
        V_alpha_inv = vpss.invert_matrix(V_alpha, q)

        # Multiply: M^T = V_e * V_alpha^-1
        # Result Size: t1 x |I|
        M_T_temp = vpss.multiply_matrix(V_e, V_alpha_inv, q)

        # Transpose (M = (V_alpha^T)^-1 * V_e^T)
        # Result Size: |I| x t1
        M_temp = vpss.transpose_matrix(M_T_temp)
        
        # 5. Apply Projection (Select first k1 columns)
        # Formula: ... * [ I_k1 ; 0 ]
        # This is equivalent to taking the first k1 columns of M_temp.
        # Transform Matrix T (Size: |I| x k1)
        Transform_T = vpss.BigIntMatrix()
        for r in range(I_len):
            row_vec = vpss.BigIntVector()
            for c in range(k1):
                row_vec.append(M_temp[r][c])
            Transform_T.append(row_vec)
            
        # 6. Construct Data Matrix D_j
        # D_j rows: Blocks (L)
        # D_j cols: Honest Parties (in basis_indices order)
        
        # Determine number of blocks from the first share
        num_blocks = len(honest_second_level_shares[honest_set[0]].s_values)
        
        D_j = vpss.BigIntMatrix()
        for blk in range(num_blocks):
            row_vec = vpss.BigIntVector()
            for id in honest_set:
                # Extract share value for this block from this party
                val = honest_second_level_shares[id].s_values[blk]
                row_vec.append(val)
            D_j.append(row_vec)
            
        # 7. Final Computation: S_tilde = D_j * Transform_T
        # D_j (Blocks x t1) * T (t1 x k1) = (Blocks x k1)
        S_tilde = vpss.multiply_matrix(D_j, Transform_T, q)
        
        return S_tilde
                
    def train_phase(self, epoch: int, lr: float, logging: bool = False):
        if logging:
            self._init_logs()
            if 'train' not in self._time_logs: self._time_logs['train'] = {}
            t_start = self._clock()

        # print(f"[Client {self.id}] Train phase starts.")
        self.g_i = self.train(epoch, lr)
        # print(f"[Client {self.id}] Train finished.")

        if logging:
            t_end = self._clock()
            self._time_logs['train']['process_ms'] = (t_end - t_start) * 1000

    def enrollment_phase(self, use_perp: bool = True, thread_count: int = 1, logging: bool = False):
        if self.participant_filter:
            yield from self._enrollment_phase_filtered(use_perp, thread_count, logging)
            return

        if logging:
            self._time_logs['enrollment'] = {}
            t_start = self._clock()

        # print(f"[Client {self.id}] Enrollment phase starts.")
        self._sync('all')                       # region 1: server broadcasts server_data, clients receive
        if logging:
            t_start = self._clock()        # re-base: exclude sync wait from receive timing
        received = False
        data = None
        while not received:
            data = self.receive(SERVER_ID, True, timeout=3.0)
            if data is not None:
                if 'g_server' in data:
                    received = True
                else:
                    pass
        g_server = data['g_server']
        mult = 10 ** data['exp']
        # print(f"[Client {self.id}] Server's data has received.")

        if logging:
            t_recv_server = self._clock()
            self._time_logs['enrollment']['receive_server_ms'] = (t_recv_server - t_start) * 1000

        if SERVER_COMPUTES_SHARES:
            # Default: shares were dealt and broadcast by the server.
            self.g_server_shares = data['g_server_shares']
        else:
            # Recompute g_server shares locally from public g_server. Because g_server is public,
            # the [k,t) randomness is zeroed (config1_det has deterministicRand=True) so every
            # honest client derives the identical share set.
            g_server_quantize = Party.quantize(g_server, mult)
            gs_secrets_vector = vpss.BigIntVector()
            for val in g_server_quantize:
                gs_secrets_vector.append(vpss.BigInt(str(val)))
            self.g_server_shares, _ = vpss.NoneVPSS.distribute(
                self.pp, self.config1_det, gs_secrets_vector, thread_count=thread_count)

        if logging:
            t_recompute_server = self._clock()
            self._time_logs['enrollment']['recompute_server_share_ms'] = (t_recompute_server - t_recv_server) * 1000
            t_recv_server = t_recompute_server

        if use_perp:
            self.cos = float(F.cosine_similarity(self.g_i, g_server, dim=0))
            if self.cos >= 1.0 - FLOAT_PRECISION:
                self.cos = 1.0
            g_i_perp = Party.get_perpendicular_vector(self.g_i, g_server)
            g_i_perp_quantize = Party.quantize(g_i_perp, mult)
        else:
            self.g_i *= torch.norm(g_server) / torch.norm(self.g_i)
            g_i_quantize = Party.quantize(self.g_i, mult)

        # print(f"[Client {self.id}] Making shares.")
        secrets_vector = vpss.BigIntVector()
        if use_perp:
            for val in g_i_perp_quantize:
                secrets_vector.append(vpss.BigInt(str(val)))
        else:
            for val in g_i_quantize:
                secrets_vector.append(vpss.BigInt(str(val)))
        self.shares, _ = vpss.NoneVPSS.distribute(self.pp, self.config1, secrets_vector, thread_count = thread_count)

        if logging:
            t_calc_first_level_share = self._clock()
            self._time_logs['enrollment']['calc_first_level_share_ms'] = (t_calc_first_level_share - t_recv_server) * 1000

        # print(f"[Client {self.id}] Send first-level shares.")
        self._sync('client')                    # region 2: clients exchange first-level shares (N)
        if logging:
            t_calc_first_level_share = self._clock()   # re-base: exclude sync wait from send timing
        target_ids = self._peer_send_order()    # round-robin: avoid downlink pile-up
        for id in target_ids:
            if id==SERVER_ID:
                continue
            data = { 'share': self.shares[id] }
            # print(f"[Client {self.id}] data size: {len(pickle.dumps(data))/(10**6)} MB")
            # print(f"[Client {self.id}] Send to {id}")
            self.unicast(data, id, logging = logging, logging_info='first_level_share')
        # data = {'cos': self.cos}
        # self.unicast(data, SERVER_ID, logging = logging, logging_info='cos')

        if logging:
            t_send_first_level_share = self._clock()
            self._time_logs['enrollment']['send_first_level_share_ms'] = (t_send_first_level_share - t_calc_first_level_share) * 1000

        yield

        if logging:
            t_wait_shares_start = self._clock()

        self.first_level_shares = {}
        target_ids = [id for id in self.party_map if id != SERVER_ID]
        while len(self.first_level_shares) < len(target_ids):
            for id in target_ids:
                if id in self.first_level_shares:
                    continue
                data = self.receive(id, block=True, timeout=3.0)
                if data is not None:
                    if 'share' in data:
                        #print(f"[Client {self.id}] Receive from {id}")
                        self.first_level_shares[id] = data['share']
                    else:
                        pass

        if logging:
            t_recv_first_level_share = self._clock()
            self._time_logs['enrollment']['receive_first_level_share_ms'] = (t_recv_first_level_share - t_wait_shares_start) * 1000

        yield

    def _enrollment_phase_filtered(self, use_perp: bool = True, thread_count: int = 1, logging: bool = False):
        """Enrollment with the participant filter on: three yields instead of two. The cosine
        is computed and reported before any share is dealt, the server's participant list
        decides whether this client deals at all, and a client sitting the round out still
        enters every barrier (with no traffic) so the fixed-arity barriers stay balanced."""
        if logging:
            self._time_logs['enrollment'] = {}
            t_start = self._clock()

        self._sync('all')                       # region 1: server broadcasts server_data, clients receive
        if logging:
            t_start = self._clock()        # re-base: exclude sync wait from receive timing
        received = False
        data = None
        while not received:
            data = self.receive(SERVER_ID, True, timeout=3.0)
            if data is not None:
                if 'g_server' in data:
                    received = True
                else:
                    pass
        g_server = data['g_server']
        mult = 10 ** data['exp']

        if logging:
            t_recv_server = self._clock()
            self._time_logs['enrollment']['receive_server_ms'] = (t_recv_server - t_start) * 1000

        # The cosine decides whether this client takes part, so it is needed before the shares.
        self.cos = float(F.cosine_similarity(self.g_i, g_server, dim=0))
        if self.cos >= 1.0 - FLOAT_PRECISION:
            self.cos = 1.0

        if logging:
            t_calc_cos = self._clock()
            self._time_logs['enrollment']['calc_cos_ms'] = (t_calc_cos - t_recv_server) * 1000

        self._sync('all')                       # region 1b: clients send cos, server receives
        if logging:
            t_calc_cos = self._clock()     # re-base: exclude sync wait from send timing
        self.unicast({'cos': self.cos}, SERVER_ID, logging=logging, logging_info='cos')

        if logging:
            t_send_cos = self._clock()
            self._time_logs['enrollment']['send_cos_ms'] = (t_send_cos - t_calc_cos) * 1000

        yield

        # --- participant list -------------------------------------------------------------
        self._sync('all')                       # region 1c: server broadcasts the participant list
        if logging:
            t_wait_list = self._clock()
        received = False
        data = None
        while not received:
            data = self.receive(SERVER_ID, True, timeout=3.0)
            if data is not None:
                if 'participants' in data:
                    received = True
                else:
                    pass
        self.set_round_participants(data['participants'], data.get('k1'), data.get('t'))

        if logging:
            t_recv_list = self._clock()
            self._time_logs['enrollment']['receive_participant_list_ms'] = (t_recv_list - t_wait_list) * 1000

        if self.participating:
            if SERVER_COMPUTES_SHARES:
                self.g_server_shares = data['g_server_shares']
            else:
                # Recompute the g_server shares locally. g_server is public, so the [k,t)
                # randomness is zeroed (config1_det) and every honest client derives the
                # identical share set.
                g_server_quantize = Party.quantize(g_server, mult)
                gs_secrets_vector = vpss.BigIntVector()
                for val in g_server_quantize:
                    gs_secrets_vector.append(vpss.BigInt(str(val)))
                self.g_server_shares, _ = vpss.NoneVPSS.distribute(
                    self.pp, self.config1_det, gs_secrets_vector, thread_count=thread_count)

            if logging:
                t_recompute_server = self._clock()
                self._time_logs['enrollment']['recompute_server_share_ms'] = (t_recompute_server - t_recv_list) * 1000
                t_recv_list = t_recompute_server

            if use_perp:
                g_i_perp = Party.get_perpendicular_vector(self.g_i, g_server)
                quantized = Party.quantize(g_i_perp, mult)
            else:
                self.g_i *= torch.norm(g_server) / torch.norm(self.g_i)
                quantized = Party.quantize(self.g_i, mult)

            secrets_vector = vpss.BigIntVector()
            for val in quantized:
                secrets_vector.append(vpss.BigInt(str(val)))
            self.shares, _ = vpss.NoneVPSS.distribute(self.pp, self.config1, secrets_vector,
                                                      thread_count=thread_count)

            if logging:
                t_calc_first_level_share = self._clock()
                self._time_logs['enrollment']['calc_first_level_share_ms'] = (t_calc_first_level_share - t_recv_list) * 1000
        else:
            self.shares = None
            self.g_server_shares = None
            self.first_level_shares = {}

        self._sync('client')                    # region 2: clients exchange first-level shares (N)
        if logging:
            t_calc_first_level_share = self._clock()   # re-base: exclude sync wait from send timing
        if self.participating:
            for id in self._peer_send_order():  # round-robin: avoid downlink pile-up
                if id == SERVER_ID:
                    continue
                self.unicast({'share': self.shares[self.pos_of[id]]}, id,
                             logging=logging, logging_info='first_level_share')

        if logging:
            t_send_first_level_share = self._clock()
            self._time_logs['enrollment']['send_first_level_share_ms'] = (t_send_first_level_share - t_calc_first_level_share) * 1000

        yield

        if logging:
            t_wait_shares_start = self._clock()

        if self.participating:
            self.first_level_shares = {}
            while len(self.first_level_shares) < len(self.participants):
                for id in self.participants:
                    if id in self.first_level_shares:
                        continue
                    data = self.receive(id, block=True, timeout=3.0)
                    if data is not None:
                        if 'share' in data:
                            self.first_level_shares[id] = data['share']
                        else:
                            pass

        if logging:
            t_recv_first_level_share = self._clock()
            self._time_logs['enrollment']['receive_first_level_share_ms'] = (t_recv_first_level_share - t_wait_shares_start) * 1000

        yield

    def verification_phase(self, use_perp: bool = True, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['verification'] = {}
            t_start = self._clock()

        if not self.participating:
            # Sitting the round out: mirror the barrier and yield schedule of a participant
            # exactly, with no traffic, so the fixed-arity barriers of the parallel run stay
            # balanced and the driver's next() count is unchanged.
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

        # print(f"[Client {self.id}] Verification phase starts.")
                
        partial_dot_shares_with_server = vpss.BigIntVector()
        partial_dot_shares_itself = vpss.BigIntVector()
        partial_dot_shares_with_server_by_dealer = vpss.BigIntVector()
        partial_dot_shares_itself_by_dealer = vpss.BigIntVector()
        sorted_user_ids = sorted([uid for uid in self.first_level_shares.keys()])
        # self.shares / self.g_server_shares are lists in share-index order: position == id
        # with the filter off, position in the participant list with it on (see _pos).
        my_pos = self._pos(self.id)
        for id in sorted_user_ids:
            peer_pos = self._pos(id)
            # s_dot_i
            d = vpss.compute_dot_product(self.g_server_shares[my_pos].s_values, self.first_level_shares[id].s_values, self.pp.q)
            partial_dot_shares_with_server.append(d)
            d = vpss.compute_dot_product(self.g_server_shares[peer_pos].s_values, self.shares[peer_pos].s_values, self.pp.q)
            partial_dot_shares_with_server_by_dealer.append(d)

            # i_dot_i
            d = vpss.compute_dot_product(self.first_level_shares[id].s_values, self.first_level_shares[id].s_values, self.pp.q)
            partial_dot_shares_itself.append(d)
            d = vpss.compute_dot_product(self.shares[peer_pos].s_values, self.shares[peer_pos].s_values, self.pp.q)
            partial_dot_shares_itself_by_dealer.append(d)
        self.shares = None
        self.g_server_shares = None
        partial_dot_shares = (partial_dot_shares_with_server, partial_dot_shares_itself)
        partial_dot_shares_by_dealer =(partial_dot_shares_with_server_by_dealer, partial_dot_shares_itself_by_dealer)

        if logging:
            t_calc_partial_dots = self._clock()
            self._time_logs['verification']['calc_partial_dots_ms'] = (t_calc_partial_dots - t_start) * 1000

        qualified_set, honest_second_level_shares_with_server, honest_second_level_shares_itself = yield from self.pairwise_verification(partial_dot_shares, partial_dot_shares_by_dealer, thread_count=thread_count, logging = logging)
        partial_dot_shares_with_server = None
        partial_dot_shares_itself = None
        partial_dot_shares_with_server_by_dealer = None
        partial_dot_shares_itself_by_dealer = None
        partial_dot_shares = None
        partial_dot_shares_by_dealer = None

        if logging:
            t_after_yield_from = self._clock()

        S_tilde_with_server = self.transposed_degree_reduction(qualified_set, honest_second_level_shares_with_server)
        S_tilde_itself = self.transposed_degree_reduction(qualified_set, honest_second_level_shares_itself)
        qualified_set = None
        honest_second_level_shares_with_server = None
        honest_second_level_shares_itself = None

        # sum the shares to make dot product share
        k1 = self.config1.k
        ones_vec = vpss.BigIntVector()
        one_val = vpss.BigInt(1)        
        for _ in range(k1):
            ones_vec.append(one_val)
        s_tilde_with_server = vpss.multiply_matrix_vec(S_tilde_with_server, ones_vec, self.pp.q)
        s_tilde_itself = vpss.multiply_matrix_vec(S_tilde_itself, ones_vec, self.pp.q)

        if logging:
            t_calc_s_tilde = self._clock()
            self._time_logs['verification']['calc_s_tilde_ms'] = (t_calc_s_tilde - t_after_yield_from) * 1000

        final_share1 = vpss.BatchShareNone()
        final_share2 = vpss.BatchShareNone()
        final_share1.x = self.first_level_shares[self.id].x
        final_share1.s_values = s_tilde_with_server
        final_share2.x = self.first_level_shares[self.id].x
        final_share2.s_values = s_tilde_itself
        data = {
            's_tilde_with_server' : final_share1,
            's_tilde_itself' : final_share2
        }
        if use_perp:
            data['cos'] = self.cos
        self._sync('all')                       # region 5: clients send s_tilde, server receives
        if logging:
            t_calc_s_tilde = self._clock()   # re-base: exclude sync wait from send timing
        self.unicast(data, SERVER_ID, logging = logging, logging_info='s_tilde')

        if logging:
            t_send_s_tilde = self._clock()
            self._time_logs['verification']['send_s_tilde_ms'] = (t_send_s_tilde - t_calc_s_tilde) * 1000

        yield

        if logging:
            t_wait_TS_start = self._clock()

        self._sync('all')                       # region 6: server broadcasts TS, clients receive
        if logging:
            t_wait_TS_start = self._clock()   # re-base: exclude sync wait from receive timing
        received = False
        data = None
        while not received:
            data = self.receive(SERVER_ID, True, 1.0)            
            if data is not None:
                if 'TS' in data:
                    received = True
                    self.TS = data['TS']
                else:
                    pass

        if logging:
            t_recv_TS = self._clock()
            self._time_logs['verification']['receive_TS_ms'] = (t_recv_TS - t_wait_TS_start) * 1000
        
        yield

    def aggregation_phase(self, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = self._clock()

        if not self.participating:
            # No share to send, but the new global model still has to arrive.
            self._sync('all')                   # region 7a
            yield
            self._sync('all')                   # region 7b
            received = False
            data = None
            while not received:
                data = self.receive(SERVER_ID, True, 3.0)
                if data is not None and 'global_weights' in data:
                    received = True
            self.weights = data['global_weights']
            yield
            return

        # scaler for quantizing TS (=cos or sin2)
        # maximal scaler = sqrt(q/2)/N
        TS_SCALE = 10**TS_EXP
        
        aggregated_share = vpss.BatchShareNone()
        aggregated_share.x = self.first_level_shares[self.id].x
        aggregated_share.s_values = self.first_level_shares[self.id].s_values
        vpss.scaleVector(aggregated_share.s_values, vpss.BigInt("0"), self.pp.q)

        for sender_id, share_obj in self.first_level_shares.items():
            ts_val = self.TS.get(sender_id, 0.0)
            
            # don't need to calculate if TS<=0
            if ts_val <= 0.0:
                continue
            
            # quantize TS
            TS_int = int(ts_val * TS_SCALE)
            TS_bigint = vpss.BigInt(str(TS_int))

            vpss.scaleVector(share_obj.s_values, TS_bigint, self.pp.q)
            vpss.addVectors(aggregated_share.s_values, share_obj.s_values, self.pp.q)

        if logging:
            t_calc_agg_share = self._clock()
            self._time_logs['aggregation']['calc_agg_share_ms'] = (t_calc_agg_share - t_start) * 1000

        self._sync('all')                       # region 7a: clients send agg_share, server receives
        if logging:
            t_calc_agg_share = self._clock()   # re-base: exclude sync wait from send timing
        self.unicast({'aggregated_share': aggregated_share}, SERVER_ID, logging = logging, logging_info='agg_share')
        # print(f"[Client {self.id}] Aggregated share sent.")

        if logging:
            t_send_agg_share = self._clock()
            self._time_logs['aggregation']['send_agg_share_ms'] = (t_send_agg_share - t_calc_agg_share) * 1000

        yield

        if logging:
            t_wait_weights_start = self._clock()

        # print(f"[Client {self.id}] Waiting for updated global weights...")
        self._sync('all')                       # region 7b: server broadcasts weights, clients receive
        if logging:
            t_wait_weights_start = self._clock()   # re-base: exclude sync wait from receive timing
        received = False
        data = None
        while not received:
            data = self.receive(SERVER_ID, True, 3.0)            
            if data is not None:
                if 'global_weights' in data:
                    received = True
                else:
                    pass
        self.weights = data['global_weights']
        # print(f"[Client {self.id}] Local model synchronized with global weights.")

        if logging:
            t_recv_weights = self._clock()
            self._time_logs['aggregation']['receive_global_weights_ms'] = (t_recv_weights - t_wait_weights_start) * 1000

        yield

    def plaintext_agg(self, logging: bool = False):
        if logging:
            self._time_logs['plaintext_agg'] = {}
            t_start = self._clock()

        # print(f"[Client {self.id}] FLTrust aggregation.")
        
        # print(f"[Client {self.id}] Send update g_i to server.")
        data = { 'g_i' : self.g_i }
        self.unicast(data, SERVER_ID, logging = logging, logging_info='g_i')

        if logging:
            t_send_g_i = self._clock()
            self._time_logs['plaintext_agg']['send_g_i_ms'] = (t_send_g_i - t_start) * 1000

        yield

        if logging:
            t_wait_weights_start = self._clock()

        # print(f"[Client {self.id}] Waiting for new global weights...")
        received = False
        data = None
        while not received:
            data = self.receive(SERVER_ID, True, 3.0)            
            if data is not None:
                if 'global_weights' in data:
                    received = True
                else:
                    pass
        self.weights = data['global_weights']
        # print(f"[Client {self.id}] Local model synchronized with global weights.")

        if logging:
            t_recv_weights = self._clock()
            self._time_logs['plaintext_agg']['receive_global_weights_ms'] = (t_recv_weights - t_wait_weights_start) * 1000

        yield

    def BREA(self, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['BREA'] = {}
            t_start = self._clock()

        # quantize and make shares
        mult = 10 ** BREA_EXP
        g_i_quantize = Party.quantize(self.g_i, mult)
        for i in range(len(g_i_quantize)):
            if g_i_quantize[i] < 0:
                g_i_quantize[i] = 0

        secrets_vector = vpss.BigIntVector()
        for val in g_i_quantize:
            v = vpss.BigInt(str(val))
            # print(f"{val}\t{v}")
            secrets_vector.append(v)
        shares, commitments = vpss.PedersenVPSS.distribute(self.pp, self.config1, secrets_vector, thread_count = thread_count, return_blinds = False)

        if logging:
            t_calc_share = self._clock()
            self._time_logs['BREA']['calc_share_ms'] = (t_calc_share - t_start) * 1000

        # send to other parties
        target_ids = [id for id in self.party_map if id != SERVER_ID]
        data = {'commitments': commitments}
        self.broadcast(data, logging = logging, logging_info='commitment')
        for id in target_ids:
            if id==SERVER_ID:
                continue
            data = { 'share': shares[id] }
            self.unicast(data, id, logging = logging, logging_info='share')
        commitments = None

        if logging:
            t_send_share = self._clock()
            self._time_logs['BREA']['send_share_ms'] = (t_send_share - t_calc_share) * 1000

        yield

        if logging:
            t_wait_shares_start = self._clock()

        # receive shares of other parties
        s = {}
        C = {}
        target_ids = [id for id in self.party_map if id != SERVER_ID]
        while len(s) < len(target_ids) or len(C) < len(target_ids):
            for id in target_ids:
                if id in s and C:
                    continue
                data = self.receive(id, block=True, timeout=3.0)
                if data is not None:
                    if 'share' in data:
                        s[id] = data['share']
                    elif 'commitments' in data:
                        C[id] = data['commitments']
                    else:
                        pass

        if logging:
            t_recv_share = self._clock()
            self._time_logs['BREA']['receive_share_ms'] = (t_recv_share - t_wait_shares_start) * 1000
        
        # verify shares
        I = {}
        for id in target_ids:
            I[id] = vpss.PedersenVPSS.verify_batch_share(
                self.pp,
                s[id],
                C[id],
                thread_count=thread_count
            )
        C = None

        if logging:
            t_verification = self._clock()
            self._time_logs['BREA']['verification_ms'] = (t_verification - t_recv_share) * 1000

        # calculate share of pairwise distance
        # Formula : ||v_j - v_k||^2 = ||v_j||^2 + ||v_k||^2 - 2<v_j,v_k>
        valid_ids = sorted([key for key, is_valid in I.items() if is_valid])
        norm_squares = {}
        d = {}
        # calculate share of ||v_j||^2
        for id in valid_ids:
            norm_squares[id] = vpss.compute_dot_product(s[id].s_values, s[id].s_values, self.pp.q)
        n = len(valid_ids)
        for j in range(n):
            for k in range(j + 1, n):
                id_j: int = valid_ids[j]
                id_k: int = valid_ids[k]
                
                # calculate share of <v_j,v_k>
                j_dot_k = vpss.compute_dot_product(s[id_j].s_values, s[id_k].s_values, self.pp.q)   # <v_j,v_k>
                # calculate final share
                j_dot_k = (j_dot_k * 2) % self.pp.q                                                 # 2<v_j,v_k>
                d_jk = (norm_squares[id_j] + norm_squares[id_k]) % self.pp.q                        # ||v_j||^2 + ||v_k||^2
                d_jk = (d_jk - j_dot_k) % self.pp.q                                                 # ||v_j||^2 + ||v_k||^2 - 2<v_j,v_k>
                if id_j not in d:
                    d[id_j] = {}
                d[id_j][id_k] = d_jk
                
        
        if logging:
            t_calc_pairwise_distance = self._clock()
            self._time_logs['BREA']['calc_pairwise_distance_ms'] = (t_calc_pairwise_distance - t_verification) * 1000
        
        # send shares of pairwise distance
        data = {
            'x' : s[self.id].x,
            'd' : d,
        }
        self.unicast(data, SERVER_ID, logging = logging, logging_info='d')

        if logging:
            t_send_pairwise_distance = self._clock()
            self._time_logs['BREA']['send_pairwise_distance_ms'] = (t_send_pairwise_distance - t_calc_pairwise_distance) * 1000

        yield

        if logging:
            t_wait_set_start = self._clock()

        # receive selected set
        S = []
        received = False
        while not received:
            data = self.receive(SERVER_ID, block=True, timeout=None)
            if 'S' in data:
                S = data['S']
                received = True
            else:
                pass
        
        if logging:
            t_recv_set = self._clock()
            self._time_logs['BREA']['recv_set_ms'] = (t_recv_set - t_wait_set_start) * 1000
        
        # add shares of all parties in set
        aggregated_share = vpss.BatchShareNone()
        aggregated_share.x = s[self.id].x
        aggregated_share.s_values = s[self.id].s_values
        vpss.scaleVector(aggregated_share.s_values, vpss.BigInt("0"), self.pp.q)
        for id in S:
            vpss.addVectors(aggregated_share.s_values, s[id].s_values, self.pp.q)

        if logging:
            t_aggregate_shares = self._clock()
            self._time_logs['BREA']['aggregate_shares_ms'] = (t_aggregate_shares - t_recv_set) * 1000
        
        # send shares of aggregated update
        data = {'aggregated_share': aggregated_share}
        self.unicast(data, SERVER_ID, logging = logging, logging_info='aggregated_share')

        if logging:
            t_send_aggregated_shares = self._clock()
            self._time_logs['BREA']['send_aggregated_shares_ms'] = (t_send_aggregated_shares - t_aggregate_shares) * 1000

        yield

        if logging:
            t_wait_weights_start = self._clock()

        # receive updated weights
        received = False
        data = None
        while not received:
            data = self.receive(SERVER_ID, True, 3.0)            
            if data is not None:
                if 'global_weights' in data:
                    received = True
                else:
                    pass
        self.weights = data['global_weights']

        if logging:
            t_recv_weights = self._clock()
            self._time_logs['BREA']['receive_global_weights_ms'] = (t_recv_weights - t_wait_weights_start) * 1000

        yield