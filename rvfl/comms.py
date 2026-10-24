"""Message transports between parties, and the compute lock of the end-to-end run.

Two transports with the same interface (send_bytes / broadcast_bytes / receive / close):

  InMemoryCommsManager   in-process queues (one process runs every party). Used by the
                         cost measurement, where only computation time and byte counts matter.
  PlainTCPCommsManager   real TCP between processes, one per party, each in its own network
                         namespace (rvfl.testbed). Bandwidth and delay are enforced by `tc` on
                         the namespace links; the transport only frames messages.

make_comms() picks one from BACKEND, which the runner sets before any party is built.

Compute lock (end-to-end run only). Every party runs in its own process on one machine, so
their local computations would compete for the same cores and slow each other down. The
worker processes therefore share one lock: a party holds it while computing and drops it
for every network operation, so computation is serialized and communication overlaps, as on
separate machines. A communication method drops the lock and does not take it back; the next
timer stamp (Party._clock) does, and subtracts the time spent waiting for it. Every phase
timer is a difference of _clock() values, so no interval contains another party's
computation.
"""
import multiprocessing
import queue
import selectors
import socket
import struct
import sys
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

# "inmemory" (one process, every party), "netns_tcp" (one process per party) or "none" (no
# messages: the parties of an accuracy run only train).
BACKEND = "inmemory"

# How the server broadcasts over TCP:
#   False -> one uplink, the receivers served one after another (N x payload / bandwidth; default)
#   True  -> one connection per receiver in parallel (a server with N x the bandwidth)
BROADCAST_PARALLEL = False

_COMPUTE_LOCK = threading.RLock()
_SERIALIZE_COMPUTE = False


def use_compute_lock(lock):
    """Serialize local computation across the processes sharing `lock` (end-to-end run)."""
    global _COMPUTE_LOCK, _SERIALIZE_COMPUTE
    _COMPUTE_LOCK = lock
    _SERIALIZE_COMPUTE = True


def compute_lock_enabled():
    return _SERIALIZE_COMPUTE


def acquire_compute():
    if _SERIALIZE_COMPUTE:
        _COMPUTE_LOCK.acquire()


def release_compute():
    if _SERIALIZE_COMPUTE:
        try:
            _COMPUTE_LOCK.release()
        except (RuntimeError, AssertionError):
            pass


def make_comms(ip: str, port: int):
    if BACKEND == "netns_tcp":
        return PlainTCPCommsManager(ip, port)
    if BACKEND == "inmemory":
        return InMemoryCommsManager(ip, port)
    if BACKEND == "none":
        return None
    raise ValueError(f"unknown comms backend {BACKEND!r}")


class _Inboxes(dict):
    """Sender address -> queue.Queue of that sender's frames, created on first use.

    The receiving thread and the protocol thread can both be the first to touch a sender's
    address. collections.defaultdict(queue.Queue) is not safe for that: building the Queue runs
    Python code, so the other thread can build a second Queue in between, and a frame put into
    the overwritten one is lost. Creating the queue under a lock gives both threads the same one.
    """

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
    """Queues shared through a multiprocessing.Manager; a broadcast puts the payload in shared
    memory once and sends every receiver a reference to it."""
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
        self._queue_cache: Dict[Tuple[Tuple, Tuple], Any] = {}
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
        alive = []
        for shm, ref_count in self._active_shms:
            if ref_count.value == 0:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            else:
                alive.append((shm, ref_count))
        self._active_shms = alive

    def send_bytes(self, data: bytes, target_addr: Tuple[str, int]) -> None:
        self._get_queue(target_addr=target_addr, sender_addr=self.my_addr).put(data)

    def broadcast_bytes(self, data: bytes, target_addrs: List[Tuple[str, int]]) -> None:
        if not target_addrs:
            return
        self._cleanup_shms()
        shm = shared_memory.SharedMemory(create=True, size=len(data))
        shm.buf[:len(data)] = data
        ref_count = self._manager.Value('i', len(target_addrs))
        meta = ("SHM_BROADCAST", shm.name, len(data), ref_count)
        for addr in target_addrs:
            self._get_queue(target_addr=addr, sender_addr=self.my_addr).put(meta)
        self._active_shms.append((shm, ref_count))

    def receive(self, sender_addr: Tuple[str, int], block: bool = True,
                timeout: Optional[float] = 30) -> Optional[bytes]:
        q = self._get_queue(target_addr=self.my_addr, sender_addr=sender_addr)
        try:
            raw = q.get(block=block, timeout=timeout)
            if isinstance(raw, tuple) and len(raw) == 4 and raw[0] == "SHM_BROADCAST":
                _, name, size, ref_count = raw
                shm = shared_memory.SharedMemory(name=name)
                try:
                    return bytes(shm.buf[:size])
                finally:
                    shm.close()
                    with self._lock:
                        ref_count.value -= 1
            return raw
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


class PlainTCPCommsManager:
    """Reliable TCP transport over real (per-namespace) links.

    No application-level shaping: the kernel and `tc` on the veth links handle bandwidth,
    propagation delay and reliability. Each message is a 4-byte big-endian length prefix and
    the body. The first frame on a new outbound connection is a handshake carrying the
    sender's logical "ip:port", so the receiver files frames by the sender's stable address
    rather than the ephemeral source port. One epoll loop serves every inbound connection,
    so the thread count per party stays constant (no FD_SETSIZE limit, unlike select()).
    """

    LEN_FMT = "!I"
    LEN_SIZE = struct.calcsize("!I")
    SEND_RETRIES = 5             # reconnect-and-resend attempts on a transient reset

    def __init__(self, ip: str, port: int):
        self.my_addr = (ip, port)
        self.party_inboxes = _Inboxes()
        self._running = True
        self._reset_log_path = None   # set_reset_log(): append-only log of network faults

        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((ip, port))
        self.lsock.listen(1024)
        self.lsock.setblocking(False)

        self._in_state = {}          # conn -> {'buf': bytearray, 'sender': (ip, port) | None}
        self._out = {}               # target_addr -> socket
        self._out_locks = {}         # target_addr -> Lock
        self._out_guard = threading.Lock()

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

    def set_reset_log(self, path):
        """Write network faults to `path` (append mode, created on the first fault)."""
        self._reset_log_path = path

    def _log_net(self, msg):
        line = f"[NET t={time.perf_counter():.3f}] {self.my_addr} {msg}\n"
        if self._reset_log_path:
            try:
                with open(self._reset_log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                pass
        else:
            sys.stderr.write(line)

    def _drop_out_addr(self, target_addr):
        """Close the cached outbound socket for target_addr so the next send reconnects."""
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
            self._log_net(f"DROP-IN reason={reason} sender={st.get('sender') if st else None}")
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
        while len(buf) >= self.LEN_SIZE:
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
                self.party_inboxes[st['sender']].put(frame)

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
            for _ in range(600):                     # the peer's listener may not be up yet
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
        # On a transient reset, drop the cached socket, reconnect (new handshake) and resend
        # the whole frame. A partial frame lives only in the dropped connection's buffer on the
        # peer and is never delivered, so nothing arrives twice or torn.
        packet = struct.pack(self.LEN_FMT, len(data)) + data
        last_err = None
        for attempt in range(self.SEND_RETRIES):
            try:
                sock, lock = self._connect(target_addr)
                with lock:
                    sock.sendall(packet)
                if attempt:
                    self._log_net(f"SEND-OK -> {target_addr} after {attempt} retries")
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
        time.sleep(0.02)                     # let the event loop leave its select
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
