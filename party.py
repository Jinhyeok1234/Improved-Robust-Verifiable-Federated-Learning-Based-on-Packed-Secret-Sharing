import socket
import threading
import queue
import pickle
import struct
import time
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
        self.comms = InMemoryCommsManager(ip, port)
        self.device = device
        
        self.model_name = model_name
        self.model = model
        self.trainer = trainer
        
        self.dataset_name = dataset_name
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

    def unicast(self, data: Dict[str, Any], receiver_id: int, echo: bool = False, logging: bool = False, logging_info: str = ''):
        if echo:
            if self.id == SERVER_ID:
                print(f"[Server] Unicasting data...")
            else:
                print(f"[Client {self.id}] Unicasting data...")
        raw_bytes = pickle.dumps(data)

        if logging:
            self._comm_logs['unicast'].setdefault(logging_info, {})[receiver_id] = {
                'size_bytes': len(raw_bytes)
            }

        addr = self.party_map[receiver_id]
        self.comms.send_bytes(raw_bytes, addr)

    def broadcast(self, data: Dict[str, Any], without_server: bool = True, echo: bool = False, logging: bool = False, logging_info: str = ''):
        if echo:
            if self.id == SERVER_ID:
                print(f"[Server] Broadcasting data...")
            else:
                print(f"[Client {self.id}] Broadcasting data...")
        
        raw_bytes = pickle.dumps(data)

        if logging:
            self._comm_logs['broadcast'][logging_info]={
                'num_receivers': len([id for id in self.party_map if not (without_server and id == SERVER_ID)]),
                'size_bytes': len(raw_bytes)
            }

        target_addrs = []
        for id, addr in self.party_map.items():
            #if (id == self.id):
            #    continue
            if without_server and (id == SERVER_ID):
                continue
            target_addrs.append(addr)
            
        self.comms.broadcast_bytes(raw_bytes, target_addrs)
    
    def receive(self, id: int, block: bool = True, timeout: float = 30) -> Dict[str, Any]|None:
        addr = self.party_map[id]
        data = self.comms.receive(addr, block=block, timeout=timeout)
        if data:
            data = pickle.loads(data)
        return data
    
    def set_VPSS_params(self, pp: vpss.PublicParameters, config1: vpss.VPSSConfig, config2: vpss.VPSSConfig):
        self.pp = pp
        self.config1 = config1
        self.config2 = config2

    def train(self, epoch: int, lr: float):
        torch.nn.utils.vector_to_parameters(self.weights.to('cuda'), self.model.parameters())
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
            t_start = time.perf_counter()

        # print(f"[Server] Train phase starts.")
        self.g_server = self.train(epoch, lr)
        # print(f"[Server] Train finished.")

        if logging:
            t_end = time.perf_counter()
            self._time_logs['train']['process_ms'] = (t_end - t_start) * 1000

    def enrollment_phase(self, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['enrollment'] = {}
            t_start = time.perf_counter()

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
        secrets_vector = vpss.BigIntVector()
        for val in g_server_quantize:
            secrets_vector.append(vpss.BigInt(str(val)))
        # print(f"[Server] Making shares.")
        shares, _ = vpss.NoneVPSS.distribute(self.pp, self.config1, secrets_vector, thread_count = thread_count)

        if logging:
            t_make_server_data = time.perf_counter()
            self._time_logs['enrollment']['make_server_data_ms'] = (t_make_server_data - t_start) * 1000

        # send data to clients
        # print(f"[Server] Send data.")
        data = {
            'g_server'          : self.g_server,
            'exp'               : self.exp,
            'g_server_shares'   : shares
        }
        # print(f"[Server] data size: {len(pickle.dumps(data))/(10**6)} MB")
        self.broadcast(data, logging=logging, logging_info='server_data')

        if logging:
            t_send_server_data = time.perf_counter()
            self._time_logs['enrollment']['send_server_data_ms'] = (t_send_server_data - t_make_server_data) * 1000


        # sample code for receiving cosine values in advance
        """
        yield

        if logging:
            t_after_yield = time.perf_counter()
        
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
            t_recv_cos = time.perf_counter()
            self._time_logs['enrollment']['receive_cos_ms'] = (t_recv_cos - t_after_yield) * 1000
            self._value_logs['cos'] = self.cosine_list

        # print(f"[Server] cosine list: {self.cosine_list}")

        yield
        """

    def verification_phase(self, use_perp: bool = True, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['verification'] = {}
            t_start = time.perf_counter()

        # print(f"[Server] Verification phase starts.")
        
        # print("[Server] Wait for clients' response.")
        # receive the dot product shares    
        target_ids = [id for id in self.party_map if id != SERVER_ID]
        final_shares_vec1 = vpss.BatchShareNoneVector()
        final_shares_vec2 = vpss.BatchShareNoneVector()
        self.cos_list = {}
        cnt = 0
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
            t_recv_s_tilde = time.perf_counter()
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
            ratios_i_dot_i = {}
            ratios_s_dot_i = {}
            s_dot_s = self.g_server_quantize_norm**2
            threshold1 = 0.00005
            threshold2 = 0.00005
            q_int = self.pp.q.to_int()
            q_half = q_int // 2
            for i in range(self.config2.N):
                # the result is not from original vector; it's quantized with multiplier
                #  -> should be checked relatively with ||g_server|| and mult
                i_dot_i = dot_products_itself[i].to_int()
                s_dot_i = dot_products_with_server[i].to_int()
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
                if(valid):
                    cos = max(self.cos_list[i],0)    # RELU
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
                t_calc_TS = time.perf_counter()
                self._time_logs['verification']['calc_TS_ms'] = (t_calc_TS - t_recv_s_tilde) * 1000
                self._value_logs['ratios_i_dot_i'] = ratios_i_dot_i
                self._value_logs['ratios_s_dot_i'] = ratios_s_dot_i
                self._value_logs['cos'] = self.cos_list
                self._value_logs['g_server_norm'] = torch.norm(self.g_server).item()
                self._value_logs['exp'] = self.exp

            data = {"TS": self.TS}
            self.broadcast(data, logging=logging, logging_info='TS')

            if logging:
                t_send_TS = time.perf_counter()
                self._time_logs['verification']['send_TS_ms'] = (t_send_TS - t_calc_TS) * 1000

        except Exception as e:
            print(f"Reconstruction Failed: {e}")
    
    def aggregation_phase(self, lr_global: float, use_error_correction: bool = False, use_perp: bool = True, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = time.perf_counter()

        # 1. receive the aggregated shares from all clients
        target_ids = [id for id in self.party_map if id != SERVER_ID]
        collected_shares = vpss.BatchShareNoneVector()
        
        # print("[Server] Collecting aggregated shares...")
        cnt = 0
        while cnt < len(target_ids):
            for id in target_ids:
                data = self.receive(id, True, None)
                if data and 'aggregated_share' in data:
                    collected_shares.append(data['aggregated_share'])
                    cnt += 1

        if logging:
            t_recv_agg_share = time.perf_counter()
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
        
        if I_size == 0:
            # print("[Server] All cosine values are negative. Use g_S only.")
            final_g = self.g_server.cpu()
        else:
            # TS_sum = sum(self.TS[i] for i in valid_indices)
            # final_g = g_sum / TS_sum
            sum_cos = sum(self.cos_list[i] for i in valid_indices)

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
            t_update_global_weights = time.perf_counter()
            self._time_logs['aggregation']['update_global_weights_ms'] = (t_update_global_weights - t_recv_agg_share) * 1000
            self._value_logs['final_g_norm'] = torch.norm(final_g).item()

        data = {
            'global_weights': self.weights,
        }
        
        # print(f"[Server] Broadcasting new global weights to all clients...")
        self.broadcast(data, logging=logging, logging_info='updated_weights')

        if logging:
            t_send_updated_global_weights = time.perf_counter()
            self._time_logs['aggregation']['send_updated_global_weights_ms'] = (t_send_updated_global_weights - t_update_global_weights) * 1000

    def FLTrust_agg(self, lr_global: float, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = time.perf_counter()

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
            t_recv_local_update = time.perf_counter()
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
            t_calc_updated_weights = time.perf_counter()
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
            t_calc_updated_weights = time.perf_counter()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000

    def FedAvg_agg(self, lr_global: float, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = time.perf_counter()
    
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
            t_recv_local_update = time.perf_counter()
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
            t_calc_updated_weights = time.perf_counter()
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
            t_send_updated_weights = time.perf_counter()
            self._time_logs['aggregation']['send_updated_weights_ms'] = (t_send_updated_weights - t_calc_updated_weights) * 1000

    def FLTrust_quantized_agg(self, lr_global: float, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = time.perf_counter()

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
            t_recv_local_update = time.perf_counter()
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
            t_calc_updated_weights = time.perf_counter()
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
            t_calc_updated_weights = time.perf_counter()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000

    def FLTrust_quantized_perp_agg(self, lr_global: float, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = time.perf_counter()

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
            t_recv_local_update = time.perf_counter()
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
        for id in updates:
            cos = float(F.cosine_similarity(updates[id], self.g_server, dim=0))
            _cos_list[id] = cos
            updates[id] = Party.get_perpendicular_vector(updates[id], self.g_server)
            if cos > 0:
                cos_list[id] = cos
                sum_cos += cos
        
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
            t_calc_updated_weights = time.perf_counter()
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
            t_calc_updated_weights = time.perf_counter()
            self._time_logs['aggregation']['calc_updated_weights_ms'] = (t_calc_updated_weights - t_recv_local_update) * 1000

    def BREA(self, lr_global: float, byzantine_rate: float = 0.3, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['BREA'] = {}
            t_start = time.perf_counter()

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
            t_recv_share = time.perf_counter()
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
            t_reconstruct_d = time.perf_counter()
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
            t_dequantize_d = time.perf_counter()
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
            t_multi_krum = time.perf_counter()
            self._time_logs['BREA']['multi_krum_ms'] = (t_multi_krum - t_dequantize_d) * 1000
            self._value_logs['A'] = a
            self._value_logs['m'] = m
            self._value_logs['selected set'] = selected

        # send selected set
        data = {'S': S}
        self.broadcast(data, logging = logging, logging_info='S')

        if logging:
            t_send_set = time.perf_counter()
            self._time_logs['BREA']['send_set_ms'] = (t_send_set - t_multi_krum) * 1000

        yield

        if logging:
            t_after_yield_from = time.perf_counter()

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
            t_receive_agg = time.perf_counter()
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
            t_reconstruct_agg = time.perf_counter()
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
            t_dequantize_and_update = time.perf_counter()
            self._time_logs['BREA']['dequantize_and_update_ms'] = (t_dequantize_and_update - t_reconstruct_agg) * 1000
            self._value_logs['final_g_norm'] = torch.norm(g_sum).item()

        data = {
            'global_weights': self.weights,
        }
        self.broadcast(data, logging=logging, logging_info='updated_weights')

        if logging:
            t_send_updated_global_weights = time.perf_counter()
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
            t_start = time.perf_counter()

        # 1. Deal second-level share

        # print(f"[Client {self.id}] Pairwise Verification Starts.")
        # print(f"[Client {self.id}] first-level-shares: {first_level_shares}")
        shares1, commitments1, blinds1 = vpss.PedersenVPSS.distribute(self.pp, self.config2, first_level_shares[0], thread_count = thread_count, return_blinds = True)
        shares2, commitments2, blinds2 = vpss.PedersenVPSS.distribute(self.pp, self.config2, first_level_shares[1], thread_count = thread_count, return_blinds = True)

        if logging:
            t_make_second_level_share = time.perf_counter()
            self._time_logs['verification']['make_second_level_share_ms'] = (t_make_second_level_share - t_start) * 1000

        # print(f"[Client {self.id}] Send second-level shares and commitments.")
        target_ids = [id for id in self.party_map if id != SERVER_ID]
        data = {'commitment1': commitments1, 'commitment2': commitments2}
        self.broadcast(data, logging = logging, logging_info='commitment')
        for id in target_ids:
            if id==SERVER_ID:
                continue
            data = {
                'share1': shares1[id],
                'blind1': blinds1[id],
                'share2': shares2[id],
                'blind2': blinds2[id],
                }
            self.unicast(data, id, logging = logging, logging_info='second_level_share')

        if logging:
            t_send_second_level_share = time.perf_counter()
            self._time_logs['verification']['send_second_level_share_ms'] = (t_send_second_level_share - t_make_second_level_share) * 1000

        yield

        if logging:
            t_after_yield = time.perf_counter()

        # print(f"[Client {self.id}] Receive second-level shares and commitments.")
        second_level_shares1 = {}
        blinds_for_verification1 = {}
        received_commitments1 = {}
        second_level_shares2 = {}
        blinds_for_verification2 = {}
        received_commitments2 = {}
        target_ids = [id for id in self.party_map if id != SERVER_ID]
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
            t_receive_second_level_share = time.perf_counter()
            self._time_logs['verification']['receive_second_level_share_ms'] = (t_receive_second_level_share - t_after_yield) * 1000

        # 2. Local verificaiton

        # print(f"[Client {self.id}] Verify shares.")
        local_verdict_vector = {id: True for id in target_ids}
        k2 = self.config2.k
        my_user_index = self.id
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
                s1 = first_level_shares_by_dealer[0][id]
                r1 = blinds_for_verification1[id]
                s2 = first_level_shares_by_dealer[1][id]
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
            t_verify_second_level_share = time.perf_counter()
            self._time_logs['verification']['verify_second_level_share_ms'] = (t_verify_second_level_share - t_receive_second_level_share) * 1000

        # 3. Broadcast verdict vector

        # print(f"[Client {self.id}] Open verify results.")
        # print(f"[Client {self.id}] Verification result: {my_verification_result}")
        self.broadcast({'verdict_vector': local_verdict_vector}, logging = logging, logging_info='verification_result')

        if logging:
            t_send_verify_result = time.perf_counter()
            self._time_logs['verification']['send_verify_result_ms'] = (t_send_verify_result - t_verify_second_level_share) * 1000

        yield

        if logging:
            t_after_yield = time.perf_counter()

        # 4. Make a global verdict vector

        target_ids = [id for id in self.party_map if id != SERVER_ID]
        global_verdict_vector = {id: True for id in target_ids}
        received_count = 0
        while received_count < len(target_ids):
            for id in target_ids:
                data = self.receive(id, True, 1.0)
                if data is not None:
                    if 'verdict_vector' in data:
                        other_vec = data['verdict_vector']
                        for target_id in global_verdict_vector:
                            if target_id in other_vec:
                                global_verdict_vector[target_id] = global_verdict_vector[target_id] and other_vec[target_id]
                        received_count += 1
        # if self.id == 0:
        #     print(f"[Client {self.id}] Integrated verification result: {global_verdict_vector}")

        if logging:
            t_receive_verify_result = time.perf_counter()
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
            t_filter_honest_clients = time.perf_counter()
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
        
        # 2. Check the number of honest parties
        if I_len < t1:
            raise ValueError(f"Not enough honest parties. Need {t1}, got {I_len}")
        
        # 3. Prepare Coordinates (alpha and e)
        # alpha: x-coordinates of the selected honest parties
        # Note: config1.shareIndices is 0-indexed vector corresponding to Party 1..N
        # If party ID starts at 1, verify config indices mapping. 
        # Assuming: Party ID k maps to shareIndices[k-1] (or check your VPSS config logic)
        alpha_vec = vpss.BigIntVector()
        for id in honest_set:
            # Party ID to index mapping check needed. Assuming id 1..N -> index 0..N-1
            alpha_vec.append(self.config1.shareIndices[id])
            
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
            t_start = time.perf_counter()

        # print(f"[Client {self.id}] Train phase starts.")
        self.g_i = self.train(epoch, lr)
        # print(f"[Client {self.id}] Train finished.")

        if logging:
            t_end = time.perf_counter()
            self._time_logs['train']['process_ms'] = (t_end - t_start) * 1000

    def enrollment_phase(self, use_perp: bool = True, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['enrollment'] = {}
            t_start = time.perf_counter()

        # print(f"[Client {self.id}] Enrollment phase starts.")
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
        self.g_server_shares = data['g_server_shares']
        mult = 10 ** data['exp']
        # print(f"[Client {self.id}] Server's data has received.")

        if logging:
            t_recv_server = time.perf_counter()
            self._time_logs['enrollment']['receive_server_ms'] = (t_recv_server - t_start) * 1000

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
            t_calc_first_level_share = time.perf_counter()
            self._time_logs['enrollment']['calc_first_level_share_ms'] = (t_calc_first_level_share - t_recv_server) * 1000

        # print(f"[Client {self.id}] Send first-level shares.")
        target_ids = [id for id in self.party_map if id != SERVER_ID]
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
            t_send_first_level_share = time.perf_counter()
            self._time_logs['enrollment']['send_first_level_share_ms'] = (t_send_first_level_share - t_calc_first_level_share) * 1000

        yield

        if logging:
            t_wait_shares_start = time.perf_counter()

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
            t_recv_first_level_share = time.perf_counter()
            self._time_logs['enrollment']['receive_first_level_share_ms'] = (t_recv_first_level_share - t_wait_shares_start) * 1000

        yield
    
    def verification_phase(self, use_perp: bool = True, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['verification'] = {}
            t_start = time.perf_counter()

        # print(f"[Client {self.id}] Verification phase starts.")
                
        partial_dot_shares_with_server = vpss.BigIntVector()
        partial_dot_shares_itself = vpss.BigIntVector()
        partial_dot_shares_with_server_by_dealer = vpss.BigIntVector()
        partial_dot_shares_itself_by_dealer = vpss.BigIntVector()
        sorted_user_ids = sorted([uid for uid in self.first_level_shares.keys()])
        for id in sorted_user_ids:
            # s_dot_i
            d = vpss.compute_dot_product(self.g_server_shares[self.id].s_values, self.first_level_shares[id].s_values, self.pp.q)
            partial_dot_shares_with_server.append(d)
            d = vpss.compute_dot_product(self.g_server_shares[id].s_values, self.shares[id].s_values, self.pp.q)
            partial_dot_shares_with_server_by_dealer.append(d)

            # i_dot_i
            d = vpss.compute_dot_product(self.first_level_shares[id].s_values, self.first_level_shares[id].s_values, self.pp.q)
            partial_dot_shares_itself.append(d)
            d = vpss.compute_dot_product(self.shares[id].s_values, self.shares[id].s_values, self.pp.q)
            partial_dot_shares_itself_by_dealer.append(d)
        self.shares = None
        self.g_server_shares = None
        partial_dot_shares = (partial_dot_shares_with_server, partial_dot_shares_itself)
        partial_dot_shares_by_dealer =(partial_dot_shares_with_server_by_dealer, partial_dot_shares_itself_by_dealer)

        if logging:
            t_calc_partial_dots = time.perf_counter()
            self._time_logs['verification']['calc_partial_dots_ms'] = (t_calc_partial_dots - t_start) * 1000

        qualified_set, honest_second_level_shares_with_server, honest_second_level_shares_itself = yield from self.pairwise_verification(partial_dot_shares, partial_dot_shares_by_dealer, thread_count=thread_count, logging = logging)
        partial_dot_shares_with_server = None
        partial_dot_shares_itself = None
        partial_dot_shares_with_server_by_dealer = None
        partial_dot_shares_itself_by_dealer = None
        partial_dot_shares = None
        partial_dot_shares_by_dealer = None

        if logging:
            t_after_yield_from = time.perf_counter()

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
            t_calc_s_tilde = time.perf_counter()
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
        self.unicast(data, SERVER_ID, logging = logging, logging_info='s_tilde')

        if logging:
            t_send_s_tilde = time.perf_counter()
            self._time_logs['verification']['send_s_tilde_ms'] = (t_send_s_tilde - t_calc_s_tilde) * 1000

        yield

        if logging:
            t_wait_TS_start = time.perf_counter()

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
            t_recv_TS = time.perf_counter()
            self._time_logs['verification']['receive_TS_ms'] = (t_recv_TS - t_wait_TS_start) * 1000
        
        yield

    def aggregation_phase(self, logging: bool = False):
        if logging:
            self._time_logs['aggregation'] = {}
            t_start = time.perf_counter()

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
            t_calc_agg_share = time.perf_counter()
            self._time_logs['aggregation']['calc_agg_share_ms'] = (t_calc_agg_share - t_start) * 1000

        self.unicast({'aggregated_share': aggregated_share}, SERVER_ID, logging = logging, logging_info='agg_share')
        # print(f"[Client {self.id}] Aggregated share sent.")

        if logging:
            t_send_agg_share = time.perf_counter()
            self._time_logs['aggregation']['send_agg_share_ms'] = (t_send_agg_share - t_calc_agg_share) * 1000

        yield

        if logging:
            t_wait_weights_start = time.perf_counter()

        # print(f"[Client {self.id}] Waiting for updated global weights...")
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
            t_recv_weights = time.perf_counter()
            self._time_logs['aggregation']['receive_global_weights_ms'] = (t_recv_weights - t_wait_weights_start) * 1000

        yield

    def plaintext_agg(self, logging: bool = False):
        if logging:
            self._time_logs['plaintext_agg'] = {}
            t_start = time.perf_counter()

        # print(f"[Client {self.id}] FLTrust aggregation.")
        
        # print(f"[Client {self.id}] Send update g_i to server.")
        data = { 'g_i' : self.g_i }
        self.unicast(data, SERVER_ID, logging = logging, logging_info='g_i')

        if logging:
            t_send_g_i = time.perf_counter()
            self._time_logs['plaintext_agg']['send_g_i_ms'] = (t_send_g_i - t_start) * 1000

        yield

        if logging:
            t_wait_weights_start = time.perf_counter()

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
            t_recv_weights = time.perf_counter()
            self._time_logs['plaintext_agg']['receive_global_weights_ms'] = (t_recv_weights - t_wait_weights_start) * 1000

        yield

    def BREA(self, thread_count: int = 1, logging: bool = False):
        if logging:
            self._time_logs['BREA'] = {}
            t_start = time.perf_counter()

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
            t_calc_share = time.perf_counter()
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
            t_send_share = time.perf_counter()
            self._time_logs['BREA']['send_share_ms'] = (t_send_share - t_calc_share) * 1000

        yield

        if logging:
            t_wait_shares_start = time.perf_counter()

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
            t_recv_share = time.perf_counter()
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
            t_verification = time.perf_counter()
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
            t_calc_pairwise_distance = time.perf_counter()
            self._time_logs['BREA']['calc_pairwise_distance_ms'] = (t_calc_pairwise_distance - t_verification) * 1000
        
        # send shares of pairwise distance
        data = {
            'x' : s[self.id].x,
            'd' : d,
        }
        self.unicast(data, SERVER_ID, logging = logging, logging_info='d')

        if logging:
            t_send_pairwise_distance = time.perf_counter()
            self._time_logs['BREA']['send_pairwise_distance_ms'] = (t_send_pairwise_distance - t_calc_pairwise_distance) * 1000

        yield

        if logging:
            t_wait_set_start = time.perf_counter()

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
            t_recv_set = time.perf_counter()
            self._time_logs['BREA']['recv_set_ms'] = (t_recv_set - t_wait_set_start) * 1000
        
        # add shares of all parties in set
        aggregated_share = vpss.BatchShareNone()
        aggregated_share.x = s[self.id].x
        aggregated_share.s_values = s[self.id].s_values
        vpss.scaleVector(aggregated_share.s_values, vpss.BigInt("0"), self.pp.q)
        for id in S:
            vpss.addVectors(aggregated_share.s_values, s[id].s_values, self.pp.q)

        if logging:
            t_aggregate_shares = time.perf_counter()
            self._time_logs['BREA']['aggregate_shares_ms'] = (t_aggregate_shares - t_recv_set) * 1000
        
        # send shares of aggregated update
        data = {'aggregated_share': aggregated_share}
        self.unicast(data, SERVER_ID, logging = logging, logging_info='aggregated_share')

        if logging:
            t_send_aggregated_shares = time.perf_counter()
            self._time_logs['BREA']['send_aggregated_shares_ms'] = (t_send_aggregated_shares - t_aggregate_shares) * 1000

        yield

        if logging:
            t_wait_weights_start = time.perf_counter()

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
            t_recv_weights = time.perf_counter()
            self._time_logs['BREA']['receive_global_weights_ms'] = (t_recv_weights - t_wait_weights_start) * 1000

        yield