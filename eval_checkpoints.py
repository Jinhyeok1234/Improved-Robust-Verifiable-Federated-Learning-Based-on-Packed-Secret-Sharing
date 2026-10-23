"""
Standalone checkpoint evaluator.

Loads saved global-model checkpoints ({DATASET}_{MODEL}_N{N}_Iter{i}.pt) one by one
and reports test (and optionally train) accuracy / error rate per iteration.

This produces the "testing error rate vs. iteration" curve used to demonstrate that
robustness is preserved (FLTrust-style metric), without retraining.

Example:
    python eval_checkpoints.py --dataset CIFAR10 --model ResNet20 --N 40 \
        --ckpt_dir ./checkpoints --out ./logs/eval_RVFL_LF.jsonl --eval_train
"""
import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
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
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])


def get_dataset(name: str, train: bool, transform, data_path: str):
    if name == "CIFAR10":
        return datasets.CIFAR10(data_path, train=train, download=True, transform=transform)
    elif name == "MNIST":
        return datasets.MNIST(data_path, train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / max(total, 1)


def discover_iters(ckpt_dir: Path, dataset: str, model: str, N: int):
    pat = re.compile(rf"^{re.escape(dataset)}_{re.escape(model)}_N{N}_Iter(\d+)\.pt$")
    iters = []
    for f in ckpt_dir.iterdir():
        m = pat.match(f.name)
        if m:
            iters.append(int(m.group(1)))
    return sorted(iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR10")
    ap.add_argument("--model", default="ResNet20")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--data", default="./dataset")
    ap.add_argument("--out", default="./logs/eval_accuracy.jsonl")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", default=None,
                    help="Comma-separated iters or 'a:b:step'. Default: all found checkpoints.")
    ap.add_argument("--stride", type=int, default=1, help="Evaluate every k-th discovered checkpoint.")
    ap.add_argument("--eval_train", action="store_true", help="Also report train accuracy.")
    ap.add_argument("--train_subset", type=int, default=5000,
                    help="Random train samples for train-acc (0 = full train set).")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    transform = build_transform(args.model)

    test_set = get_dataset(args.dataset, False, transform, args.data)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=2)

    train_loader = None
    if args.eval_train:
        train_set = get_dataset(args.dataset, True, transform, args.data)
        if args.train_subset and args.train_subset < len(train_set):
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(train_set), generator=g)[:args.train_subset].tolist()
            train_set = Subset(train_set, idx)
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=False, num_workers=2)

    if args.iters is None:
        iters = discover_iters(ckpt_dir, args.dataset, args.model, args.N)[:: args.stride]
    elif ":" in args.iters:
        a, b, *s = args.iters.split(":")
        step = int(s[0]) if s else 1
        iters = list(range(int(a), int(b) + 1, step))
    else:
        iters = [int(x) for x in args.iters.split(",")]

    if not iters:
        print(f"No checkpoints found in {ckpt_dir} for {args.dataset}_{args.model}_N{args.N}_Iter*.pt")
        return

    model = get_model(args.model).to(device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # RESUME + MERGE (single file, kept sorted): load already-measured iterations and fold in any
    # legacy <tag>_peek.jsonl (same eval, old split file). Already-measured iters are skipped; NEW
    # ones (e.g. after a stride change) are evaluated and merged; the file is rewritten SORTED at the
    # end, so different-stride runs interleave correctly. Delete the file to force a full re-eval.
    records, absorbed_peek = eval_io.load_merged(out_path)
    todo = [it for it in iters if it not in records]
    print(f"Evaluating {len(todo)} new checkpoint(s) ({len(records)} already done) -> {out_path}")

    with out_path.open("a" if records else "w", encoding="utf-8") as fout:   # append -> crash-safe
        for it in todo:
            ckpt = ckpt_dir / f"{args.dataset}_{args.model}_N{args.N}_Iter{it}.pt"
            if not ckpt.exists():
                print(f"  [skip] missing {ckpt.name}")
                continue
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)

            test_acc = accuracy(model, test_loader, device)
            rec = {"iteration": it, "test_acc": test_acc, "test_err": 1.0 - test_acc}
            if train_loader is not None:
                rec["train_acc"] = accuracy(model, train_loader, device)

            records[it] = rec
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            msg = f"  Iter {it:>4} | test_err={1.0 - test_acc:.4f} test_acc={test_acc:.4f}"
            if "train_acc" in rec:
                msg += f" train_acc={rec['train_acc']:.4f}"
            print(msg)

    # rewrite the whole file SORTED by iteration (merges/orders across differing strides), which
    # also folds in any absorbed legacy peek records; then retire the migrated peek file.
    eval_io.write_sorted(out_path, records)
    if absorbed_peek is not None:
        absorbed_peek.unlink()
        print(f"[absorbed] folded {absorbed_peek.name} into {out_path.name} and removed it")


if __name__ == "__main__":
    main()