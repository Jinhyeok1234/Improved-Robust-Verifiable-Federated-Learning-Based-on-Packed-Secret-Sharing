"""One party of the end-to-end run, as its own process.

run.py starts one process per party (server and N clients) with worker_main. The process
enters its network namespace first, so every socket it opens is bound to the namespace's
shaped link (rvfl.testbed); then it builds its party from the plain settings in `spec`, takes
part in the barriers and the compute lock shared with the others, and runs the rounds.
Bandwidth and delay come from `tc` on the links; the transport (comms.PlainTCPCommsManager)
does no shaping of its own.
"""
import json
import os
import warnings
from pathlib import Path

# torch warns that it cannot initialize NVML when sudo has dropped the environment (CUDA works),
# and torchvision's CIFAR-10 loader trips a NumPy 2.4 deprecation warning on every load.
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")


def _enter_netns(ns_name: str):
    CLONE_NEWNET = 0x40000000
    fd = os.open(f"/run/netns/{ns_name}", os.O_RDONLY)
    try:
        if hasattr(os, "setns"):
            os.setns(fd, CLONE_NEWNET)
        else:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            if libc.setns(fd, CLONE_NEWNET) != 0:
                e = ctypes.get_errno()
                raise OSError(e, os.strerror(e))
    finally:
        os.close(fd)


def die_with_parent():
    """Linux: have the kernel send this process SIGKILL when its parent ends, however it ends
    (so no party is left behind when run.py is killed). Elsewhere a no-op."""
    import ctypes
    import signal
    parent = os.getppid()
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        return
    if os.getppid() != parent:            # the parent ended before the request took effect
        os._exit(1)


def _exhaust(gen):
    for _ in gen:
        pass


def worker_main(spec, all_barrier, client_barrier, compute_lock, start_gate, end_gate):
    die_with_parent()
    _enter_netns(spec["netns"])

    import torch
    import vpss_module as vpss
    from . import comms, crypto_params
    from .experiment import resolve_device, seed_round, vpss_configs
    from .filtering import FilterPolicy
    from .models import get_model
    from .party import Client, Server
    from .trainer import LocalTrainer

    is_server = spec["party_id"] == -1
    device = resolve_device(spec["device"])
    threads = spec["threads"]

    comms.BACKEND = "netns_tcp"
    comms.BROADCAST_PARALLEL = spec["broadcast_parallel"]
    comms.use_compute_lock(compute_lock)

    pp = crypto_params.pp_from_spec(vpss, spec["crypto"])
    configs = vpss_configs(spec["n"], spec["t1"], spec["k1"], spec["t2"], spec["k2"])

    model = get_model(spec["model"]).to(device)
    common = dict(model_name=spec["model"], dataset_name=spec["dataset"], model=model,
                  trainer=LocalTrainer(model, device=device), partition_path=spec["partition_path"],
                  device=device, dataset_path=spec["data_dir"], batch_size=spec["batch_size"],
                  val_max=spec["val_max"])
    addr_map = {int(k): tuple(v) for k, v in spec["addr_map"].items()}
    ip, port = addr_map[spec["party_id"]]
    if is_server:
        party = Server(ip=ip, port=port, **common)
    else:
        party = Client(id=spec["party_id"], ip=ip, port=port, server_addr=addr_map[-1], **common)
    party.party_map = addr_map
    party.comms.set_reset_log(str(Path(spec["out_dir"]) / f"netfaults_{spec['party_id']}.log"))
    party.set_VPSS_params(pp, *configs)
    party.cos_at_enrollment = spec["cos_reveal"] == "enroll"
    if spec["filter"] != "none":
        f = spec["filter_settings"]
        policy = None
        if is_server:
            policy = FilterPolicy(spec["filter"], spec["n"], spec["k1"], spec["k2"], f["alpha"],
                                  threshold=f["threshold"], addback=f["addback"],
                                  cost_model=f.get("cost_model"))
            party.enable_filter(spec["n"], spec["k1"], spec["k2"], policy.c, policy)
        else:
            party.enable_filter(spec["n"], spec["k1"], spec["k2"], f["c"])
    party.weights = torch.load(spec["weights_path"], map_location="cpu")
    if spec.get("bn_state_path"):
        # the shared vector holds the parameters only; the batch-norm statistics of the
        # starting checkpoint are put back here (they matter for the server's evaluation only)
        bufs = dict(model.named_buffers())
        for name, val in torch.load(spec["bn_state_path"], map_location="cpu").items():
            if name in bufs and bufs[name].dtype.is_floating_point:
                bufs[name].data.copy_(val.to(bufs[name].device))
    party._all_barrier = all_barrier
    party._client_barrier = None if is_server else client_barrier

    if not is_server and spec["party_id"] in spec["malicious_ids"] and spec["poison"]:
        from .attacks import poisoning_config, submission_scale
        cfg = poisoning_config(spec["poison"])
        if cfg is not None:
            party.set_poisoning(cfg)
        party.submission_scale = submission_scale(spec["poison"], spec["n"])

    out_dir = Path(spec["out_dir"])
    fout = (out_dir / f"worker_{spec['party_id']}.jsonl").open("w", encoding="utf-8")
    fval = (out_dir / "values.jsonl").open("w", encoding="utf-8") if is_server else None
    seed = spec["seed"]
    for it in range(1, spec["rounds"] + 1):
        # local training: serialized under the compute lock, not part of the timed protocol
        comms.acquire_compute()
        try:
            seed_round(seed, it, spec["party_id"])
            party.train_phase(epoch=1, lr=spec["lr_local"])
        finally:
            comms.release_compute()
        if is_server:
            party.addback_seed = seed + it        # the filter's add-back draw, per round

        start_gate.wait()                         # the parent starts its round clock
        comms.acquire_compute()
        party._lock_dropped = False               # held now; comm methods drop it, stamps re-take it
        party._lock_wait_s = 0.0
        try:
            if is_server:
                _exhaust(party.enrollment_phase(threads))
                party.verification_phase(threads)
                party.aggregation_phase(spec["lr_global"], threads)
            else:
                _exhaust(party.enrollment_phase(threads))
                _exhaust(party.verification_phase(threads))
                _exhaust(party.aggregation_phase())
        finally:
            if not getattr(party, "_lock_dropped", False):
                comms.release_compute()
            party._lock_dropped = False
        end_gate.wait()                           # the parent stops its round clock

        rec = party.extract_and_clear_logs(it)
        values = rec.pop("values", None)
        rec["lock_wait_ms"] = round(getattr(party, "_lock_wait_s", 0.0) * 1000, 3)
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if fval is not None:
            fval.write(json.dumps({"iteration": it, "values": values}) + "\n")
            fval.flush()
        if is_server and spec["checkpoints"]:
            torch.nn.utils.vector_to_parameters(party.weights.to(device, copy=True), model.parameters())
            ck = out_dir / "checkpoints" / f"iter{it}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ck)
    fout.close()
    if fval is not None:
        fval.close()
    try:
        party.comms.close()
    except Exception:
        pass
