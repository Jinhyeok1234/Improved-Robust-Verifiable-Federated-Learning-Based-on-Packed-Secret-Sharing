# Improved Robust Verifiable Federated Learning Based on Packed Secret Sharing

[![Conference](https://img.shields.io/badge/NeurIPS-2026-blue.svg)](https://neurips.cc/)
[![Status](https://img.shields.io/badge/Status-Ready-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Private-red.svg)]()

This repository contains the official implementation of **RVFL (Robust Verifiable Federated Learning)**. This project optimizes degree reduction efficiency in Verifiable Packed Secret Sharing (VPSS) and integrates it with a Byzantine-robust Federated Learning framework (FLTrust) to defend against both inference and Byzantine attacks.

---

## 1. Project Overview

- **Optimized Degree Reduction**
  - Re-engineered VPSS primitives in C++ to reduce computational overhead in Byzantine-robust settings.
- **Robust Aggregation**
  - Integration with FLTrust to ensure robustness against malicious model updates.
- **Privacy Preservation**
  - Utilizes Packed Secret Sharing (PSS) to protect local updates from inference attacks.
- **Multi-Protocol Support**
  - Includes implementations for `RVFL`, `FLTrust`, `BREA`, and `FedAvg` for benchmarking.

---

## 2. Directory Structure
```text
.
├── vpss/                  # Core VPSS C++ Implementation & Python Bindings
│   ├── VPSS.cpp           # VPSS primitive logic
│   ├── bindings.cpp       # Pybind11 interface
│   └── setup.py           # Build script for the extension
├── dataloader.py          # Data loading logic with poisoning hooks
├── dataset_partitioner.py # Dataset splitting for IID/Non-IID settings
├── models.py              # Model definitions (Primary: ResNet20)
├── party.py               # FL Party (Server/Client) behaviors
├── trainer.py             # Local training and global update application
├── main.py                # Main experiment entry point
└── requirements.txt       # Software dependencies
```

---

## 3. Getting Started

### Hardware & Software Requirements
- **OS**: Ubuntu 24.04.4 LTS
- **GPU**: NVIDIA RTX A6000 (CUDA 12.x recommended)
- **Compiler**: GCC 13.3.0
- **Python**: Version 3.12.3 with PyTorch 2.10.0

### Installation & Build
- **Install Dependencies**
  - `pip install -r requirements.txt`
- **Build C++ Extension**
  - `cd vpss`
  - `python setup.py build_ext --inplace`
  - `cp *.so ../` (Copy the compiled library to the root directory)
  - `cd ..`

### Mandatory Directory Setup
**Warning**: The script requires specific output directories to exist. Please create them manually before running the experiments:
```bash
mkdir -p dataset partitions checkpoints logs/RVFL logs/FLTrust logs/BREA logs/FedAvg
```

---

## 4. Usage

### Configuration
- You can modify the following parameters in `main.py`:
  - `MODE`: Select protocol (`RVFL`, `FLTrust`, `BREA`, `FedAvg`)
  - `N`: Total number of clients (Default: 40)
  - `NUM_ITERATIONS`: Training rounds
  - `use_perp`: Toggle perpendicular vector quantization (RVFL only)

### Execution
```bash
python main.py
```

---

## 5. Experiment Logs & Checkpoints

- **Checkpoints**
  - Stored in `./checkpoints/`
  - Pattern: `{DATASET}_{MODEL}_N{N}_Iter{n}.pt`
- **Performance Logs**
  - Generated in `./logs/{MODE}/` as `.jsonl` files
  - `*_processing_time_log.jsonl`: Computation time breakdown
  - `*_communication_cost_log.jsonl`: Network overhead in bytes
  - `*_values_log.jsonl`: Aggregation values and metrics

---

## 6. Implementation Details

- **Cryptographic Settings**
  - Uses 512-bit prime field parameters for secure multi-party computation.
- **Simulation Environment**
  - Parties communicate via local loopback (127.0.0.1) for accurate cost measurement.
- **Data Partitioning**
  - `dataset_partitioner.py` handles the distribution logic for the server-side proxy dataset and client-side indices.

---

## 7. Anonymous Submission Note

This repository is configured for anonymous review (NeurIPS 2026). All institutional affiliations and author names have been removed. The anonymous version of the repository is maintained at:
[https://anonymous.4open.science/r/Improved-Robust-Verifiable-Federated-Learning-Based-on-Packed-Secret-Sharing-8001/](https://anonymous.4open.science/r/Improved-Robust-Verifiable-Federated-Learning-Based-on-Packed-Secret-Sharing-8001/)