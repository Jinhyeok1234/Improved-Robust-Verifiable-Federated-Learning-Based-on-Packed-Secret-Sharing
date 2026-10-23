import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Dict, Any
import json

from models import get_model

# --- 상수 설정 ---
DATASET_NAME    = "CIFAR10"
# MODEL_NAME      = "CIFAR_CNN"
MODEL_NAME      = "ResNet20"
#DATASET_NAME    = "MNIST"
#MODEL_NAME      = "MNIST_FLTrust"
BATCH_SIZE      = 1000
N               = 100
END             = 601 + int((N-10)*15)
ITERATIONS      = range(0,END,10)
ITERATIONS      = range(0,11,1)

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHECKPOINT_DIR  = Path("./checkpoints")
CHECKPOINT_DIR  = Path(f"checkpoints/FLTrust_quantized_perp_label_flip_fltrust_m20")
DATASET_PATH    = Path("./dataset")
SAVE_PATH_TRAIN = CHECKPOINT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_N{N}_train_accuracy_plot.png"
SAVE_PATH_TEST  = CHECKPOINT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_N{N}_test_accuracy_plot.png"
SAVE_PATH_BOTH = CHECKPOINT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_N{N}_train_and_test_accuracy_plot.png"
SAVE_PATH_JSON: Path = CHECKPOINT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_N{N}_accuracy_results.json"

# EVAL_TRAIN_ACC  = True
EVAL_TRAIN_ACC  = False
PLOT_TOGETHER   = True

# --- 데이터셋 및 로더 준비 함수 ---
def get_data_loaders(dataset_name: str, dataset_path: Path, batch_size: int, model_name: str) -> tuple[DataLoader, DataLoader]:
    """데이터셋 이름에 따라 적절한 전처리와 Train/Test DataLoader를 반환하는 함수"""
    if model_name == 'ResNet20':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif 'ResNet' in model_name:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root        = str(dataset_path), 
            train       = True, 
            download    = True, 
            transform   = transform
        )
        test_dataset = datasets.MNIST(
            root        = str(dataset_path), 
            train       = False, 
            download    = True, 
            transform   = transform
        )
        
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root        = str(dataset_path), 
            train       = True, 
            download    = True, 
            transform   = transform
        )
        test_dataset = datasets.CIFAR10(
            root        = str(dataset_path), 
            train       = False, 
            download    = True, 
            transform   = transform
        )
        
    elif dataset_name == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        train_dataset = datasets.ImageNet(
            root        = str(dataset_path), 
            split       = 'train', 
            transform   = transform
        )
        test_dataset = datasets.ImageNet(
            root        = str(dataset_path), 
            split       = 'val', 
            transform   = transform
        )
        
    else:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset_name}")

    train_loader = DataLoader(
        dataset     = train_dataset, 
        batch_size  = batch_size, 
        shuffle     = False
    )
    test_loader = DataLoader(
        dataset     = test_dataset, 
        batch_size  = batch_size, 
        shuffle     = False
    )
    
    return train_loader, test_loader

def evaluate(model, data_loader, device):
    """모델의 정확도를 계산하는 함수"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return (correct / total) * 100

# --- 테스트 루프 준비 ---
train_loader, test_loader = get_data_loaders(
    dataset_name = DATASET_NAME,
    dataset_path = DATASET_PATH,
    batch_size   = BATCH_SIZE,
    model_name   = MODEL_NAME
)

train_results = []
test_results  = []
model = get_model(MODEL_NAME).to(device=DEVICE)

# --- 테스트 루프 실행 ---
for i in ITERATIONS:
    file_path = CHECKPOINT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_N{N}_Iter{i}.pt"
    
    if file_path.exists():
        state_dict = torch.load(f=file_path, map_location=DEVICE)
        model.load_state_dict(state_dict=state_dict)

        test_accuracy  = evaluate(model=model, data_loader=test_loader, device=DEVICE)
        test_results.append(test_accuracy)

        if EVAL_TRAIN_ACC:
            train_accuracy = evaluate(model=model, data_loader=train_loader, device=DEVICE)
            train_results.append(train_accuracy)
            print(f"Iteration {i}: Train Acc = {train_accuracy:.2f}%, Test Acc = {test_accuracy:.2f}%")
        else:
            print(f"Iteration {i}: Test Acc = {test_accuracy:.2f}%")
    else:
        print(f"Warning: {file_path} 파일이 존재하지 않아 평가를 건너뜁니다.")
        train_results.append(None)
        test_results.append(None)

# --- 결과 데이터 JSON 저장 ---
results_to_save: List[Dict[str, Any]] = []

for idx, iteration_step in enumerate(ITERATIONS):
    if test_results[idx] is not None:
        record: Dict[str, Any] = {
            "iteration": iteration_step,
            "test_accuracy": test_results[idx]
        }
        
        if EVAL_TRAIN_ACC and train_results[idx] is not None:
            record["train_accuracy"] = train_results[idx]
            
        results_to_save.append(record)

with open(SAVE_PATH_JSON, "w", encoding="utf-8") as f:
    json.dump(results_to_save, f, indent=4, ensure_ascii=False)

# --- 결과 시각화 준비 ---
valid_indices       = [i for i, res in zip(ITERATIONS, test_results) if res is not None]

# --- 결과 시각화 준비 ---
valid_indices       = [i for i, res in zip(ITERATIONS, test_results) if res is not None]
valid_train_results = [res for res in train_results if res is not None]
valid_test_results  = [res for res in test_results if res is not None]

# --- Train Accuracy 그래프 시각화 및 저장 ---
if PLOT_TOGETHER:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    if EVAL_TRAIN_ACC:
        ax.plot(
            valid_indices, 
            valid_train_results, 
            # marker    = '^', 
            linestyle = '--', 
            color     = 'r', 
            label     = 'Train Accuracy'
        )

    ax.plot(
        valid_indices, 
        valid_test_results, 
        # marker    = 'o', 
        linestyle = '-', 
        color     = 'b', 
        label     = 'Test Accuracy'
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if EVAL_TRAIN_ACC:
        ax.set_title(label=f"{DATASET_NAME} Train and Test Accuracy per Iteration", fontsize=14)
    else:
        ax.set_title(label=f"{DATASET_NAME} Test Accuracy per Iteration", fontsize=14)
    ax.set_xlabel(xlabel="Iteration Step", fontsize=12)
    ax.set_ylabel(ylabel="Accuracy (%)", fontsize=12)
    ax.grid(visible=True, linestyle='--', alpha=0.7)
    ax.legend(loc="lower right")

    if valid_train_results:
        ax.set_ylim(bottom=min(valid_train_results) - 5, top=100)

    save_path = SAVE_PATH_TEST
    if EVAL_TRAIN_ACC:
        save_path = SAVE_PATH_BOTH

    plt.tight_layout()
    plt.savefig(
        fname       = str(save_path),
        dpi         = 300,
        bbox_inches = 'tight',
        format      = 'png'
    )
    plt.close(fig)

else:
    if EVAL_TRAIN_ACC:
        fig_train, ax_train = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        ax_train.plot(
            valid_indices, 
            valid_train_results, 
            # marker    = '^', 
            linestyle = '--', 
            color     = 'r', 
            label     = 'Train Accuracy'
        )

        ax_train.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_train.set_title(label=f"{DATASET_NAME} Train Accuracy per Iteration", fontsize=14)
        ax_train.set_xlabel(xlabel="Iteration Step", fontsize=12)
        ax_train.set_ylabel(ylabel="Accuracy (%)", fontsize=12)
        ax_train.grid(visible=True, linestyle='--', alpha=0.7)
        ax_train.legend(loc="lower right")

        if valid_train_results:
            ax_train.set_ylim(bottom=min(valid_train_results) - 5, top=100)

        plt.tight_layout()
        plt.savefig(
            fname       = str(SAVE_PATH_TRAIN),
            dpi         = 300,
            bbox_inches = 'tight',
            format      = 'png'
        )
        plt.close(fig_train)

    # --- Test Accuracy 그래프 시각화 및 저장 ---
    fig_test, ax_test = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    ax_test.plot(
        valid_indices, 
        valid_test_results, 
        # marker    = 'o', 
        linestyle = '-', 
        color     = 'b', 
        label     = 'Test Accuracy'
    )

    ax_test.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_test.set_title(label=f"{DATASET_NAME} Test Accuracy per Iteration", fontsize=14)
    ax_test.set_xlabel(xlabel="Iteration Step", fontsize=12)
    ax_test.set_ylabel(ylabel="Accuracy (%)", fontsize=12)
    ax_test.grid(visible=True, linestyle='--', alpha=0.7)
    ax_test.legend(loc="lower right")

    if valid_test_results:
        ax_test.set_ylim(bottom=min(valid_test_results) - 5, top=100)

    plt.tight_layout()
    plt.savefig(
        fname       = str(SAVE_PATH_TEST),
        dpi         = 300,
        bbox_inches = 'tight',
        format      = 'png'
    )
    plt.close(fig_test)