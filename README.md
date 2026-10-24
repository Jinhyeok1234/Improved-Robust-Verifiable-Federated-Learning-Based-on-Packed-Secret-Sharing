# Improved Robust Verifiable Federated Learning Based on Packed Secret Sharing

Code of the NeurIPS 2026 paper *Improved Robust Verifiable Federated Learning Based on Packed
Secret Sharing* (RVFL). RVFL aggregates client updates with FLTrust's cosine trust scores while
the updates stay secret-shared: each client deals packed shares of the component of its update
perpendicular to the server's root update, the clients verify each other's dot-product shares
with Pedersen-committed second-level shares, and the server reconstructs only the two dot
products per client and the weighted aggregate.

## Layout

```text
.
├── vpss/                 C++ VPSS primitives (GMP, OpenSSL elliptic curves) and Python bindings
├── rvfl/                 the protocol and everything around it
│   ├── party.py          server and clients of the RVFL protocol, phase by phase
│   ├── baselines.py      plaintext FLTrust, RVFL's aggregation in the clear, BREA
│   ├── filtering.py      optional per-round participant filter (not used in the paper)
│   ├── comms.py          in-process and TCP transports, compute lock of the end-to-end run
│   ├── testbed.py        network namespaces with tc-shaped links
│   ├── worker.py         one party of the end-to-end run, as its own process
│   ├── experiment.py     seeding, VPSS configurations, round drivers
│   ├── crypto_params.py  commitment groups (NIST curves; the Schnorr groups are legacy)
│   ├── curve_bench.py    per-curve timing of the VPSS primitives
│   ├── attacks.py        poisoning attacks (LF, Trim, Krum, Adaptive, Scaling)
│   ├── data.py, models.py, trainer.py, logs.py
├── run.py                runs the experiments listed in a TOML file
├── experiments/          the experiment lists of the paper
├── make_artifacts.py     the paper's tables and figures from the run folders
└── results/              the measurements behind the paper, in run.py's folder layout
```

## Requirements

- Linux, Python 3.11 or later (the paper used Python 3.12.3, GCC 13.3.0, Ubuntu 24.04)
- GMP (`libgmp-dev`) and OpenSSL headers (`libssl-dev`) for the VPSS extension
- `pip install -r requirements.txt`
- End-to-end runs only: root, `iproute2` with the `netem` queueing discipline
- The paper's timings were taken with jemalloc preloaded (`libjemalloc2`), see below

Build the extension and put it next to `run.py`:

```bash
cd vpss
python setup.py build_ext --inplace
cp vpss_module*.so ..
cd ..
```

`setup.py` enables the elliptic-curve backend when it finds the OpenSSL headers (it prints
which); every experiment uses it.

## Running experiments

An experiment list is a TOML file of `[[run]]` entries. A setting given as a list is varied, and
the entry expands into one point per combination:

```toml
[[run]]
name = "cost"
kind = "cost"            # train | cost | e2e | curve_bench
protocol = "rvfl"
n = [10, 20]             # two points
k1 = "t"                 # k1 = t1 = N // 3
```

```bash
python run.py experiments/cost.toml              # every point; finished ones are skipped
python run.py experiments/cost.toml --list       # the points and whether they are done
python run.py experiments/attacks.toml --jobs 4 --gpus 0 1
```

Each point writes a folder `runs/<name>/<point>/` with its settings (`config.json`), its status,
and its measurements (`rvfl/logs.py` lists the files). Every setting and its default is described
at the top of `run.py`. The kinds:

- `train`: accuracy under attack. RVFL's aggregation is computed in the clear
  (`rvfl.baselines`); it produces the protocol's model without the cryptographic cost per round.
- `cost`: computation time and bytes sent by every party, all parties in one process.
- `e2e`: wall-clock rounds over emulated links, one process per party, each in its own network
  namespace with its link shaped by `tc`. Needs root. Local computation is serialized across the
  parties (one machine plays all of them) and excluded from the communication timers.
- `curve_bench`: the VPSS primitives timed per elliptic curve.

`seed` (default 0) fixes the data split, the initial model, and the random draws of every local
training (one seed per round and party). On the CPU a point run twice gives the same model. On a
GPU the runs differ slightly: cuDNN's convolutions and the backward passes of the pooling layers
add up in an order that changes from run to run.

`--jobs` runs several `train` points at once; timing points always run one at a time.

Timings in the paper were measured with jemalloc:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python run.py experiments/cost.toml
sudo -E LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 "$(which python)" run.py experiments/e2e.toml
```

## Reproducing the paper

| Artifact | Experiments | Make it |
|---|---|---|
| Accuracy comparison, cosine dynamics | `experiments/accuracy.toml` | `python make_artifacts.py figures --only accuracy cosine` |
| Attack sweep, Scaling attack | `experiments/attacks.toml` | `python make_artifacts.py figures --only attack_sweep scaling` |
| Measured costs by N and by k1, comparison with BREA | `experiments/cost.toml` | `python make_artifacts.py tables --only cost_n cost_k1 brea` |
| LAN / WAN end-to-end, communication breakdown | `experiments/e2e.toml` | `python make_artifacts.py tables --only e2e_lan e2e_wan comm_breakdown` |
| Elliptic-curve parameters | `experiments/curve_bench.toml` | `python make_artifacts.py tables --only curves` |

Tables are written as LaTeX by default (`--format latex csv md` for the others), figures as PDF
(`--fmt pdf png`), into `artifacts/`. The cost and end-to-end tables leave out each point's first
round (connection set-up and first allocations; `--drop-first`).

The measurements behind the paper are in `results/`, in the same layout (the per-party logs
gzip-compressed), so the artifacts can be made from them directly:

```bash
python make_artifacts.py tables --runs results
python make_artifacts.py figures --runs results
```

## Participant filter

`rvfl/filtering.py` implements an optional filter that drops the clients with a non-positive
cosine from a round (they would contribute zero to the aggregate) and rebuilds the sharing for
the remaining participants. It is not part of the paper's experiments; `experiments/filter.toml`
shows its settings, and `python make_artifacts.py figures --only filter` plots its rounds against
the filter-off points of the same N and link.
