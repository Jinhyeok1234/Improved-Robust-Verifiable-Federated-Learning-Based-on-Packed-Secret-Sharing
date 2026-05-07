import multiprocessing
from multiprocessing.synchronize import Lock, Barrier
import os
import time
import random
import math
import torch
import numpy as np
from typing import Dict, Tuple
import logging
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from pathlib import Path
import json

# os.add_dll_directory(r"C:\dev\vcpkg\installed\x64-windows\bin")
import vpss_module as vpss
import dataset_partitioner
from party import Server, Client
from models import get_model
from trainer import LocalTrainer

import warnings
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")

def is_prime(n, k=5):
    """Miller-Rabin primality test"""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_safe_prime(bits):
    """Generate a safe prime p such that p = 2q + 1, where q is also prime."""
    print(f"Generating {bits}-bit safe primes... (this might take a moment)")
    while True:
        q = random.getrandbits(bits - 1)
        q |= (1 << (bits - 2)) | 1 
        
        if is_prime(q):
            p = 2 * q + 1
            if is_prime(p):
                return p, q

def setup_crypto_params(bits=128, p_int: int|None = None, q_int: int|None = None):
    if p_int is None:
        p_int, q_int = generate_safe_prime(bits)
        print(f"Selected p : {p_int}")
        print(f"Selected q : {q_int}")
    
    p_bi = vpss.BigInt(str(p_int))
    q_bi = vpss.BigInt(str(q_int))
    
    g_bi = vpss.find_generator(p_bi, q_bi)
    
    while True:
        h_bi = vpss.find_generator(p_bi, q_bi)
        if h_bi.to_int() != g_bi.to_int():
            break
            
    return {
        "p": str(p_bi),
        "q": str(q_bi),
        "g": str(g_bi),
        "h": str(h_bi)
    }

def init_public_parameter(p, q, g, h):
    pp = vpss.PublicParameters()
    pp.p = vpss.BigInt(p)
    pp.q = vpss.BigInt(q)
    pp.g = vpss.BigInt(g)
    pp.h = vpss.BigInt(h)
    return pp

def init_config_parameter(t, k, N, secret_indices: vpss.BigIntVector, share_indices: vpss.BigIntVector, padType: str = "Zero"):
    config = vpss.VPSSConfig()
    config.t = t
    config.k = k
    config.N = N
    if padType == "Zero":
        config.padType = vpss.PaddingMode.Zero
    elif padType == "Random":
        config.padType = vpss.PaddingMode.Random
    else:
        # dummy
        config.padType = vpss.PaddingMode.Zero
    config.secretIndices = secret_indices
    config.shareIndices = share_indices
    return config

def init_party(
    node_type: str,
    my_id: int,
    party_addr_map: Dict[int, Tuple[str, int]],
    dataset_name: str,
    dataset_path: str,
    partition_path: str,
    device: int,
    model_name: str,
    model: torch.nn.Module,
    trainer: LocalTrainer,
    batch_size: int,
    val_max: int,
    initial_weights: torch.Tensor,
    pp: vpss.PublicParameters,
    config1: vpss.VPSSConfig,
    config2: vpss.VPSSConfig,
):
    my_ip, my_port = party_addr_map[my_id]
    if node_type == 'Server':
        # Server ID = -1
        party = Server(
            ip=my_ip,
            port=my_port,
            party_map={},
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            partition_path=partition_path,
            model=model,
            trainer=trainer,
            device=device,
            batch_size=batch_size,
            val_max=val_max
        )
    else:
        server_addr = party_addr_map[-1] # ID -1 is Server
        party = Client(
            id=my_id,
            ip=my_ip,
            port=my_port,
            server_addr=server_addr,
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            partition_path=partition_path,
            model=model,
            trainer=trainer,
            device=device,
            batch_size=batch_size,
            val_max=val_max
        )
    party.party_map = party_addr_map
    party.set_VPSS_params(pp, config1, config2)
    party.weights = initial_weights
    return party


if __name__ == "__main__":
    # Linux: fork, Windows: spawn
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # ==========================
    # Configuration
    # ==========================
    # DATASET_NAME = "MNIST"
    DATASET_NAME = "CIFAR10"
    # MODEL_NAME = "MNIST_FLTrust"
    # MODEL_NAME = "CIFAR_CNN"
    MODEL_NAME = "ResNet20"
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SERVER_DATASET_SIZE = 100
    DATASET_PATH = "./dataset"
    PARTITION_ROOT = "./partitions"
    CKPT_DIR = Path("./checkpoints")
    INIT_WEIGHT_PATH = "./init_model.pt"

    # MODE = "FedAvg"
    # MODE = "BREA"
    # MODE = "FLTrust"
    # MODE = "FLTrust_quantized"
    # MODE = "FLTrust_quantized_perp"
    # MODE = "FLTrust_quantized_perp"
    MODE = "RVFL"   # our scheme
    use_perp = True # only for RVFL

    ITER_START = 0
    NUM_ITERATIONS = 600

    CHECKPOINT_INTERVAL       = 1

    SERVER_IP = "127.0.0.1"
    SERVER_PORT = 5000
    BASE_CLIENT_PORT = 5001

    N = 40
    t1 = N // 3     # 1 to N/2; for robustness, 1 to N/3
    t2 = N // 3
    # k1 = t1       # 1 to t
    k1 = 1
    k2 = 1

    if MODE == "BREA":
        k1 = 1          # no packing
        t2 = 2*t1-1     # no degree reduction

    NUM_ITERATIONS += int((N-10)*15)
    NUM_ITERATIONS = 1

    lr_local = 0.1
    lr_global = 1.0

    LOG_DIR = Path("./logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TIME_LOG_PATH = LOG_DIR / MODE / f"{MODE}_N{N}_processing_time_log.jsonl"
    COMM_LOG_PATH = LOG_DIR / MODE / f"{MODE}_N{N}_communication_cost_log.jsonl"
    VALUE_LOG_PATH = LOG_DIR / MODE / f"{MODE}_N{N}_values_log.jsonl"
    # if TIME_LOG_PATH.exists(): TIME_LOG_PATH.unlink()
    # if COMM_LOG_PATH.exists(): COMM_LOG_PATH.unlink()
    # if VALUE_LOG_PATH.exists(): VALUE_LOG_PATH.unlink()
    ENABLE_LOGGING = True

    print("=== 1. Setup Crypto Parameters ===")
    # p,q for 512-bit
    p = int(9681430027940933314117844523556564365100465364126496139331159627462563422327337615390594744138274835627182601092258110206638556852362843066680532343607499)
    q = int(4840715013970466657058922261778282182550232682063248069665579813731281711163668807695297372069137417813591300546129055103319278426181421533340266171803749)
    crypto_params = setup_crypto_params(bits=512, p_int=p, q_int=q)

    vpss_params = {
        **crypto_params, # p, q, g, h
        "t1": t1,
        "k1": k1,
        "t2": t2,
        "k2": k2,
        "N": N
    }
    print(f"Generated Params: N={N}, t1={t1}, k1={k1}, t2={t2}, k2={k2}")

    secret_indices = vpss.BigIntVector()
    # secret_pts: 1 to t_max
    t_max = max(vpss_params["t1"], vpss_params["t2"])
    for i in range(1, t_max + 1):
        secret_indices.append(vpss.BigInt(i))
        
    share_indices = vpss.BigIntVector()
    # share_pts: N+1 to 2N
    for i in range(N + 1, 2 * N + 1):
        share_indices.append(vpss.BigInt(i))
    print(f"e = {secret_indices}\nalpha = {share_indices}")

    pp = init_public_parameter(vpss_params["p"], vpss_params["q"], vpss_params["g"], vpss_params["h"])
    config1 = init_config_parameter(vpss_params["t1"], vpss_params["k1"], vpss_params["N"], secret_indices, share_indices)
    config2 = init_config_parameter(vpss_params["t2"], vpss_params["k2"], vpss_params["N"], secret_indices, share_indices)

    print("\n=== 2. Partitioning Dataset ===")
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        
    PARTITION_PATH, _, ratio = dataset_partitioner.partition_and_save(
        dataset_name=DATASET_NAME,
        num_clients=N,
        save_path=PARTITION_ROOT,
        # is_iid=False,
        is_iid=True,
        q=0.5,
        dataset_path=DATASET_PATH,
        server_samples=SERVER_DATASET_SIZE
    )

    print("\n=== 3. Network Configuration ===")
    # ID -1: Server, ID 0~N-1: Clients
    party_addr_map = {}
    party_addr_map[-1] = (SERVER_IP, SERVER_PORT)
    for i in range(N):
        party_addr_map[i] = (SERVER_IP, BASE_CLIENT_PORT + i)


    print("\n=== 4. Initial Weight Generation ===")

    model = get_model(MODEL_NAME)
    trainer = LocalTrainer(model, device=DEVICE)

    if ITER_START == 0:
        weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        torch.save(weights, INIT_WEIGHT_PATH)
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR, exist_ok=True)
        filename = f"{DATASET_NAME}_{MODEL_NAME}_N{N}_Iter0.pt"
        save_path = CKPT_DIR / filename
        torch.save(model.state_dict(), save_path)
        print("Initial global model saved.")
    M = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_int = int(vpss_params["q"])
    val_max = math.isqrt(q_int//2)
    print(f"val_max : {val_max} \t Model Params M : {M}")

    weights = None
    if ITER_START == 0:
        if os.path.exists(INIT_WEIGHT_PATH):
            weights = torch.load(INIT_WEIGHT_PATH, map_location='cpu')
            ITER_START += 1
        else:
            print(f"Warning: Initial weights file not found.")
    elif ITER_START != 0 and os.path.exists(CKPT_DIR):
        filename = f"{DATASET_NAME}_{MODEL_NAME}_N{N}_Iter{ITER_START-1}.pt"
        if os.path.exists(CKPT_DIR / filename):
            model.load_state_dict(torch.load(CKPT_DIR / filename))
            weights = torch.nn.utils.parameters_to_vector(model.parameters()).cpu().detach().clone()
        else:
            print(f"Warning: Weights file for Iter{ITER_START-1} not found.")


    print("\n=== 5. Spawning Parties ===")

    common_args = (
        party_addr_map,
        DATASET_NAME, DATASET_PATH, PARTITION_PATH,
        DEVICE, MODEL_NAME, model, trainer,
        BATCH_SIZE, val_max, weights,
        pp, config1, config2,
    )

    server = init_party('Server', -1, *common_args)
    clients = []
    for id in range(N):
        clients.append(init_party('Client', id, *common_args))

    THREAD_COUNT = 60
    for iteration in range(ITER_START, NUM_ITERATIONS + 1):
        t_start = time.perf_counter()

        print(f"=== Iteration {iteration} / {NUM_ITERATIONS} Start ===")

        # In the case of BREA, server training is not required.
        if MODE == "BREA":
            print(f"\t[Train Phase]")
            total = len(clients)
            for i, client in enumerate(clients):
                print(f"\r\t\tClients' Train Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="")
                client.train_phase(epoch=1, lr=lr_local, logging=ENABLE_LOGGING)
            print()

            client_processes = [client.BREA(THREAD_COUNT, logging=ENABLE_LOGGING) for client in clients]
            server_process = server.BREA(lr_global=lr_global, byzantine_rate=0.3, thread_count=THREAD_COUNT, logging=ENABLE_LOGGING)
            try:
                # deal share of local update
                print(f"\t[Client - Deal Shares]")
                for cp in client_processes:
                    next(cp)
                # calculate share of pairwise distance
                print(f"\t[Client - Verify Shares and Calculate Pairwise Distance]")
                for cp in client_processes:
                    next(cp)
                # select clients by Multi-Krum
                print(f"\t[Server - Reconstruct and Multi-Krum]")
                next(server_process)
                # make share of aggregated update
                print(f"\t[Client - Calculate Aggregated Shares]")
                for cp in client_processes:
                    next(cp)
                # update weights
                print(f"\t[Server - Reconstruct and Update Model]")
                next(server_process)
                # receive new weights
                print(f"\t[Client - Receive Updated Model]")
                for cp in client_processes:
                    next(cp)
            except StopIteration:
                pass
        else:
            # 0. Train Phase
            print(f"\t[Train Phase]")
            server.train_phase(epoch=1, lr=lr_local, logging=ENABLE_LOGGING)
            total = len(clients)
            for i, client in enumerate(clients):
                print(f"\r\t\tClients' Train Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="")
                client.train_phase(epoch=1, lr=lr_local, logging=ENABLE_LOGGING)
            print()

            if MODE == "RVFL":
                # 1. Enrollment Phase
                print(f"\t[Enrollment Phase]")
                processes = [client.enrollment_phase(use_perp, THREAD_COUNT, logging=ENABLE_LOGGING) for client in clients]
                server.enrollment_phase(THREAD_COUNT, logging=ENABLE_LOGGING)
                while True:
                    try:
                        for process in processes:
                            next(process)
                    except StopIteration:
                        break
                
                # 2. Verification Phase
                print(f"\t[Verification Phase]")
                processes = [client.verification_phase(use_perp, THREAD_COUNT, logging=ENABLE_LOGGING) for client in clients]
                for cp in processes:
                    next(cp)              # send second-level share with Pedersen VSS
                for cp in processes:
                    next(cp)              # send verify list
                for cp in processes:
                    next(cp)              # send s_tilde to server
                server.verification_phase(use_perp, THREAD_COUNT, logging=ENABLE_LOGGING)      # send TS
                for cp in processes:
                    next(cp)              # receive TS
                
                # 3. Aggregation & Final Update Phase
                print(f"\t[Aggregation Phase]")
                processes = [client.aggregation_phase(logging=ENABLE_LOGGING) for client in clients]
                for cp in processes:
                    next(cp)
                server.aggregation_phase(lr_global, use_error_correction=True, use_perp=use_perp, thread_count=THREAD_COUNT, logging=ENABLE_LOGGING)
                for cp in processes:
                    next(cp)
            else:
                print(f"\t[Aggregation Phase]")
                processes = [client.plaintext_agg(logging=ENABLE_LOGGING) for client in clients]
                for cp in processes:
                    next(cp)
                if MODE == "FLTrust":
                    # FLTrust
                    server.FLTrust_agg(lr_global = lr_global, logging=ENABLE_LOGGING)
                elif MODE == "FLTrust_quantized":
                    # FLTrust with quantized vector (simulate RVFL without perp)
                    server.FLTrust_quantized_agg(lr_global = lr_global, logging=ENABLE_LOGGING)
                elif MODE == "FLTrust_quantized_perp":
                    # FLTrust with quantized perpendicular vector (simulate RVFL with perp)
                    server.FLTrust_quantized_perp_agg(lr_global = lr_global, logging=ENABLE_LOGGING)
                elif MODE == "FedAvg":
                    # FedAvg
                    server.FedAvg_agg(lr_global = lr_global, logging=ENABLE_LOGGING)
                for cp in processes:
                    next(cp)
            
        t_end = time.perf_counter()
        print(f"Processing time for 1 iteration: {(t_end - t_start) * 1000:.2f} ms")
            


        if (iteration-1)%CHECKPOINT_INTERVAL == 0:
            # save weights
            # format: {dataset}_{model}_{N}_{iteration}.pt
            torch.nn.utils.vector_to_parameters(server.weights, model.parameters())
            filename = f"{DATASET_NAME}_{MODEL_NAME}_N{N}_Iter{iteration}.pt"
            save_path = CKPT_DIR / filename
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

        if ENABLE_LOGGING:
            with TIME_LOG_PATH.open("a", encoding="utf-8") as f_time, \
                 COMM_LOG_PATH.open("a", encoding="utf-8") as f_comm, \
                 VALUE_LOG_PATH.open("a", encoding="utf-8") as f_value :
                 
                for client in [server] + clients:
                    logs = client.extract_and_clear_logs(iteration)
                    
                    time_entry = {
                        "party_id": logs["party_id"],
                        "iteration": logs["iteration"],
                        "time_logs_ms": logs["processing_time_ms"]
                    }
                    f_time.write(json.dumps(time_entry) + "\n")
                    
                    comm_entry = {
                        "party_id": logs["party_id"],
                        "iteration": logs["iteration"],
                        "comm_logs_bytes": logs["communication_cost_bytes"]
                    }
                    f_comm.write(json.dumps(comm_entry) + "\n")

                    if logs["party_id"] == -1:
                        value_entry = {
                            "iteration": logs["iteration"],
                            "values": logs["values"]
                        }
                        f_value.write(json.dumps(value_entry) + "\n")
        
        print(f"=== Iteration {iteration} Finished ===")

    print(f"=== All Iterations Completed. ===")
    print("\n=== Simulation Finished ===")