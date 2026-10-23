import json
from typing import Dict, Any

def calculate_communication_averages(file_path: str) -> Dict[str, float]:
    total_server_b_bytes: int = 0
    total_server_u_bytes: int = 0
    server_sample_count: int = 0
    
    total_client_b_bytes: int = 0
    total_client_u_bytes: int = 0
    client_sample_count: int = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            data: Dict[str, Any] = json.loads(line)
            party_id: int = data.get("party_id", 0)
            comm_logs: Dict[str, Any] = data.get("comm_logs_bytes", {})

            # 1. Broadcast 계산
            b_bytes: int = sum(
                v.get("size_bytes", 0) 
                for v in comm_logs.get("broadcast", {}).values()
            )

            # 2. Unicast 계산 (구버전/신버전 자동 판별)
            u_bytes: int = 0
            for share_type, share_data in comm_logs.get("unicast", {}).items():
                if "size_bytes" in share_data:
                    # --- 구버전 처리 로직 ---
                    # 덮어쓰기로 인해 마지막 기록만 남은 경우, (r_id + 1)을 곱해 총량을 추정합니다.
                    r_id: int = share_data.get("receiver_id", -1)
                    s_bytes: int = share_data.get("size_bytes", 0)
                    
                    if r_id == -1:
                        u_bytes += s_bytes
                    else:
                        u_bytes += s_bytes * (r_id + 1)
                else:
                    # --- 신버전 처리 로직 ---
                    # 덮어쓰기 오류가 해결되어 0, 1, 2... 각 수신자별로 딕셔너리가 존재하는 경우,
                    # 임의로 곱셈을 할 필요 없이 각 수신자에게 보낸 size_bytes를 그대로 합산합니다.
                    for receiver_id_str, metrics in share_data.items():
                        if isinstance(metrics, dict):
                            u_bytes += metrics.get("size_bytes", 0)

            # 3. 누적 연산
            if party_id == -1:
                total_server_b_bytes += b_bytes
                total_server_u_bytes += u_bytes
                server_sample_count += 1
            else:
                total_client_b_bytes += b_bytes
                total_client_u_bytes += u_bytes
                client_sample_count += 1

    s_count: int = server_sample_count if server_sample_count > 0 else 1
    c_count: int = client_sample_count if client_sample_count > 0 else 1

    return {
        "server_broadcast_avg": total_server_b_bytes / s_count,
        "server_unicast_avg": total_server_u_bytes / s_count,
        "client_broadcast_avg": total_client_b_bytes / c_count,
        "client_unicast_avg": total_client_u_bytes / c_count
    }


if __name__ == '__main__':
    # [B6] Path from the command line: `python comm_log_analyzer.py <communication_cost_log.jsonl>`.
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "logs/BREA/BREA_N40_communication_cost_log.jsonl"
    averages = calculate_communication_averages(file_path)
    print(json.dumps(averages, indent=4))