"""
Backdoor / Scaling-attack evaluator: Attack Success Rate (ASR) per checkpoint.

The Scaling attack is TARGETED, so test error is the wrong metric. Following FLTrust, we
report ASR = fraction of trigger-embedded test inputs (whose TRUE label is NOT the target)
that the global model classifies as the attacker's target label. We also report clean test
accuracy so a low ASR can be read alongside a still-working model.

Trigger must match training (dataloader.PoisonedDataset._apply_backdoor): a `size`x`size`
patch set to `color`, at `pos` (negative = from the bottom/right corner). NOTE: the dataloader
default pos=(-3,-3),size=(3,3) yields an EMPTY slice [-3:0] and stamps nothing -- fix it there
(use pos modulo H/W, see --trigger_pos below) so training and this evaluator agree.

Example:
    python eval_backdoor.py --dataset CIFAR10 --model ResNet20 --N 100 \
        --ckpt_dir ./checkpoints/<TAG> --out ./logs/eval/<TAG>_asr.jsonl \
        --target_label 2 --stride 25
"""
import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import get_model
import eval_io


def build_transform(model_name: str):
    if model_name == "ResNet20":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif "ResNet" in model_name:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def get_testset(name: str, transform, data_path: str):
    if name == "CIFAR10":
        return datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    if name == "MNIST":
        return datasets.MNIST(data_path, train=False, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def stamp_trigger(x: torch.Tensor, pos, size: int, color: float) -> torch.Tensor:
    """Stamp a size x size patch of `color` onto a (B,C,H,W) batch. Matches the (fixed)
    dataloader _apply_backdoor: pos taken modulo H/W so negative pos = bottom/right corner."""
    H, W = x.shape[-2], x.shape[-1]
    py, px = pos[0] % H, pos[1] % W
    x = x.clone()
    x[:, :, py:py + size, px:px + size] = color
    return x


@torch.no_grad()
def clean_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def attack_success_rate(model, loader, device, target_label, pos, size, color):
    """Fraction of non-target-true inputs that, with the trigger stamped, are predicted as target."""
    model.eval()
    hit = total = 0
    for data, target in loader:
        keep = target != target_label           # ASR is measured on true-label != target only
        if keep.sum() == 0:
            continue
        data = stamp_trigger(data[keep].to(device), pos, size, color)
        pred = model(data).argmax(dim=1)
        hit += (pred == target_label).sum().item()
        total += data.size(0)
    return hit / max(total, 1)


def discover_iters(ckpt_dir: Path, dataset: str, model: str, N: int):
    pat = re.compile(rf"^{re.escape(dataset)}_{re.escape(model)}_N{N}_Iter(\d+)\.pt$")
    return sorted(int(m.group(1)) for m in (pat.match(f.name) for f in ckpt_dir.iterdir()) if m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR10")
    ap.add_argument("--model", default="ResNet20")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--data", default="./dataset")
    ap.add_argument("--out", default="./logs/eval/backdoor_asr.jsonl")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stride", type=int, default=1, help="Evaluate every k-th checkpoint.")
    ap.add_argument("--target_label", type=int, default=2, help="Attacker target (CIFAR-10 bird=2).")
    ap.add_argument("--trigger_pos", type=int, nargs=2, default=[-3, -3],
                    help="Top-left of the trigger patch (negative = from corner; taken mod H/W).")
    ap.add_argument("--trigger_size", type=int, default=3)
    ap.add_argument("--trigger_color", type=float, default=1.0)
    ap.add_argument("--no_clean", action="store_true", help="Skip clean-accuracy (ASR only).")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    transform = build_transform(args.model)
    test_set = get_testset(args.dataset, transform, args.data)
    loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=2)

    iters = discover_iters(ckpt_dir, args.dataset, args.model, args.N)[:: args.stride]
    if not iters:
        print(f"No checkpoints in {ckpt_dir} for {args.dataset}_{args.model}_N{args.N}_Iter*.pt")
        return

    model = get_model(args.model).to(device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # RESUME + MERGE (single file, kept sorted): load already-measured iterations and fold in any
    # legacy <tag>_asr_peek.jsonl (same eval, old split file); skip measured iters, evaluate new
    # ones, and rewrite SORTED at the end so a changed --stride interleaves in order.
    records, absorbed_peek = eval_io.load_merged(out_path)
    todo = [it for it in iters if it not in records]
    print(f"ASR eval: {len(todo)} new ({len(records)} done), target={args.target_label}, "
          f"trigger pos={tuple(args.trigger_pos)} size={args.trigger_size} -> {out_path}")

    with out_path.open("a" if records else "w", encoding="utf-8") as fout:   # append -> crash-safe
        for it in todo:
            ckpt = ckpt_dir / f"{args.dataset}_{args.model}_N{args.N}_Iter{it}.pt"
            if not ckpt.exists():
                print(f"  [skip] missing {ckpt.name}")
                continue
            model.load_state_dict(torch.load(ckpt, map_location=device))
            asr = attack_success_rate(model, loader, device, args.target_label,
                                      args.trigger_pos, args.trigger_size, args.trigger_color)
            rec = {"iteration": it, "asr": asr}
            if not args.no_clean:
                rec["clean_acc"] = clean_accuracy(model, loader, device)
            records[it] = rec
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            msg = f"  Iter {it:>4} | ASR={asr:.4f}"
            if "clean_acc" in rec:
                msg += f" clean_acc={rec['clean_acc']:.4f}"
            print(msg)

    # rewrite SORTED by iteration (order-correct across differing strides), folding in any absorbed
    # legacy peek records; then retire the migrated peek file.
    eval_io.write_sorted(out_path, records)
    if absorbed_peek is not None:
        absorbed_peek.unlink()
        print(f"[absorbed] folded {absorbed_peek.name} into {out_path.name} and removed it")


if __name__ == "__main__":
    main()