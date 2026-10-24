# Results of the paper

The measurements behind the paper's tables and figures, in the folder layout `run.py` writes
(one folder per point, see `rvfl/logs.py`; the per-party logs are gzip-compressed). Make the
artifacts from them with

```bash
python make_artifacts.py tables --runs results
python make_artifacts.py figures --runs results
```

| Folder | Experiment list | Artifacts |
|---|---|---|
| `accuracy/` | `experiments/accuracy.toml` | accuracy comparison, cosine dynamics |
| `attacks/` | `experiments/attacks.toml` | attack sweep, Scaling attack |
| `cost/` | `experiments/cost.toml` | measured costs by N and by k1, comparison with BREA |
| `e2e/` | `experiments/e2e.toml` | LAN and WAN end-to-end, communication breakdown |
| `curves/` | `experiments/curve_bench.toml` (N = 100) | elliptic-curve parameters |

These were measured with the code as it stood before it was cleaned up (git tag
`pre-cleanup`), and converted into this layout; their `config.json` says so in `note`. The runs
were not seeded, so `seed` is null. The cleaned code differs from that code in a few computations,
so a rerun does not reproduce these numbers bit for bit:

- the quantization multiplier is the largest power of ten that satisfies the bound of appendix
  G.1 (before: a fixed margin of 10^5 below sqrt(q/2) / ||g_S||);
- trust scores are rounded to the nearest integer after scaling by 10^8 (before: truncated);
- a reported cosine outside [-1, 1] gets no trust score; an update parallel to g_S is sent with a
  random perpendicular vector (before: a zero vector, which failed the norm check);
- BREA shares negative coordinates as they are (before: set to 0);
- every run is seeded.

The smaller multiplier also shows in the bytes when k1 = t1: the first-level sharing then has no
random coefficients, so the encoded size of a share follows the size of the quantized
coordinates (for k1 < t1 it does not). At N = 10, k1 = 3, a client's unicast is about 57.0 MB
with the cleaned code against 57.9 MB here.
