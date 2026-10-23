import json
import os
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

class AccuracyVisualizer:
    def __init__(
        self, 
        n_list: List[int],
        data_dir: str = "./", 
        output_filename: str = "comparison_plot.png",
        mode: str = "accuracy"
    ) -> None:
        self.data_dir: str = data_dir
        self.n_list: List[int] = n_list
        self.output_filename: str = output_filename
        self.mode: str = mode  # 'accuracy' or 'error'
        self.algorithms: List[str] = ["Ours", "FLTrust"]
        self.styles: Dict[str, str] = {"Ours": "-", "FLTrust": "--"}
        self.threshold: float = 84.0

    def _load_and_preprocess(self, file_path: str) -> Dict[str, List[float]]:
        processed_data: Dict[str, List[float]] = {
            "iteration": [],
            "train": [],
            "test": []
        }
        
        if not os.path.exists(file_path):
            return processed_data

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data: List[Dict[str, Any]] = json.load(f)

        for entry in raw_data:
            train_acc: float = entry.get("train_accuracy", 0.0)
            test_acc: float = entry.get("test_accuracy", 0.0)
            iter_val: float = float(entry.get("iteration", 0))

            if self.mode == "error":
                processed_data["train"].append(100.0 - train_acc)
                processed_data["test"].append(100.0 - test_acc)
            else:
                processed_data["train"].append(train_acc)
                processed_data["test"].append(test_acc)
            
            processed_data["iteration"].append(iter_val)

            # only use (train_accuracy <= 84)
            if train_acc >= self.threshold:
                print(f"file {file_path} termiated at iteration {iter_val}")
                break
                
        return processed_data

    def plot(self, show_train: bool = True, show_test: bool = True) -> None:
        num_subplots: int = sum([show_train, show_test])
        if num_subplots == 0:
            return

        fig: Figure
        axs: Any
        fig, axs = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 5), squeeze=False)
        
        axes_list: List[Axes] = axs.flatten().tolist()
        current_ax_idx: int = 0
        
        metrics: List[str] = []
        if show_train: metrics.append("train")
        if show_test: metrics.append("test")

        y_label: str = "Accuracy (%)" if self.mode == "accuracy" else "Error (%)"
        colors: List[str] = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for metric in metrics:
            ax: Axes = axes_list[current_ax_idx]
            for i, n in enumerate(self.n_list):
                color: str = colors[i % len(colors)]
                for algo in self.algorithms:
                    file_name: str = f"{algo}_N{n}.json"
                    file_path: str = os.path.join(self.data_dir, file_name)
                    
                    data: Dict[str, List[float]] = self._load_and_preprocess(file_path)
                    
                    if not data["iteration"]:
                        continue
                        
                    ax.plot(
                        data["iteration"], 
                        data[metric], 
                        label=f"{algo} (N={n})",
                        linestyle=self.styles[algo],
                        color=color
                    )
            
            ax.set_xlabel("Iteration")
            ax.set_ylabel(y_label)
            ax.set_title(f"{metric.capitalize()} {self.mode}")
            ax.legend()
            current_ax_idx += 1

        plt.tight_layout()
        plt.savefig(self.output_filename, dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    mode = "accuracy"  # 'accuracy' or 'error'
    visualizer = AccuracyVisualizer(
        data_dir="./", 
        n_list=[10,20,30,40,50],
        output_filename=f"{mode}_comparison.png",
        mode=mode  # 'accuracy' or 'error'
    )
    
    visualizer.plot(show_train=True, show_test=True)