import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import matplotlib.axes as maxes
from typing import List, Dict, Any, Tuple
from enum import Enum

class ProcessStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

class ValueCategory(Enum):
    POSITIVE = "positive"
    NON_POSITIVE = "non_positive"

class LogAnalyzer:
    def __init__(self, file_path: str, ema_span_early: int = 10, ema_span_late: int = 50, transition_iteration: int = 50) -> None:
        self.file_path: str = file_path
        self.ema_span_early: int = ema_span_early
        self.ema_span_late: int = ema_span_late
        self.transition_iteration: int = transition_iteration
        self.raw_data: List[Dict[str, Any]] = []
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.status: ProcessStatus = ProcessStatus.PENDING

    def parse_line(self, json_string: str) -> Dict[str, Any]:
        return dict(json.loads(json_string))

    def extract_cos_metrics(self, cos_data: Dict[str, float]) -> Tuple[float, float]:
        if not cos_data:
            return 0.0, 0.0
        
        total_count: int = len(cos_data)
        sum_cos: float = 0.0
        le_zero_count: int = 0
        
        for key, value in cos_data.items():
            value_float: float = float(value)
            sum_cos += value_float
            
            category: ValueCategory = ValueCategory.POSITIVE if value_float > 0.0 else ValueCategory.NON_POSITIVE
            if category == ValueCategory.NON_POSITIVE:
                le_zero_count += 1
                
        mean_cos: float = sum_cos / total_count
        ratio_le_zero: float = le_zero_count / total_count
        
        return mean_cos, ratio_le_zero

    def process_file(self, iter_end: int = -1) -> None:
        parsed_list: List[Dict[str, Any]] = []
        with open(self.file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    parsed_data: Dict[str, Any] = self.parse_line(line)
                    
                    cos_dict: Dict[str, float] = {}
                    if "cos" in parsed_data.get("values", {}):
                        cos_dict = parsed_data["values"]["cos"]
                        
                    mean_cos: float
                    ratio_le_zero: float
                    mean_cos, ratio_le_zero = self.extract_cos_metrics(cos_dict)
                    
                    iteration_val: int = parsed_data.get("iteration", 0)
                    
                    row: Dict[str, Any] = {
                        "iteration": iteration_val,
                        "cos_mean": mean_cos,
                        "cos_ratio_le_zero": ratio_le_zero
                    }
                    parsed_list.append(row)

                    if iter_end != -1 and iteration_val == iter_end:
                        break
        
        self.dataframe = pd.DataFrame(parsed_list)
        self.calculate_trends()
        self.status = ProcessStatus.COMPLETED

    def _calculate_adaptive_ema(self, series: pd.Series, iterations: pd.Series) -> pd.Series:
        result: np.ndarray = np.zeros(len(series))
        
        if len(series) == 0:
            return pd.Series(dtype=float)
            
        ema: float = series.iloc[0]
        result[0] = ema
        
        for idx in range(1, len(series)):
            val: float = series.iloc[idx]
            current_iter: float = float(iterations.iloc[idx])
            
            if current_iter < self.transition_iteration:
                progress: float = current_iter / self.transition_iteration
                current_span: float = self.ema_span_early + (self.ema_span_late - self.ema_span_early) * progress
            else:
                current_span: float = float(self.ema_span_late)
                
            alpha: float = 2.0 / (current_span + 1.0)
            ema = alpha * val + (1.0 - alpha) * ema
            result[idx] = ema
            
        return pd.Series(result, index=series.index)

    def calculate_trends(self) -> None:
        if self.dataframe.empty:
            return

        iterations: pd.Series = self.dataframe["iteration"]

        # Adaptive EMA 적용 (오직 EMA만 남김)
        self.dataframe["cos_mean_adaptive_ema"] = self._calculate_adaptive_ema(self.dataframe["cos_mean"], iterations)
        self.dataframe["cos_ratio_adaptive_ema"] = self._calculate_adaptive_ema(self.dataframe["cos_ratio_le_zero"], iterations)

    def visualize(self, save_path : str  = "visualization_cos_result.png", N: int = 10) -> None:
        if self.dataframe.empty:
            return

        fig: mfigure.Figure
        axes_array: Any
        fig, axes_array = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
        
        ax1: maxes.Axes = axes_array[0]
        ax2: maxes.Axes = axes_array[1]

        # chart 1: cos_mean (Raw + Adaptive EMA)
        ax1.plot(self.dataframe["iteration"], self.dataframe["cos_mean"], label="Raw Data", color="green", alpha=0.2)
        ax1.plot(
            self.dataframe["iteration"], 
            self.dataframe["cos_mean_adaptive_ema"], 
            # label=f"Adaptive EMA (span: {self.ema_span_early} -> {self.ema_span_late})", 
            label=f"Adaptive EMA", 
            color="blue", 
            linewidth=2
        )
        ax1.set_title(f"Cosine Similarity Mean Trend (N={N})")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cos Mean")
        ax1.legend()
        ax1.grid(True)

        # chart 2: cos_ratio_le_zero (Raw + Adaptive EMA)
        ax2.plot(self.dataframe["iteration"], self.dataframe["cos_ratio_le_zero"], label="Raw Data", color="red", alpha=0.2)
        ax2.plot(
            self.dataframe["iteration"], 
            self.dataframe["cos_ratio_adaptive_ema"], 
            # label=f"Adaptive EMA (span: {self.ema_span_early} -> {self.ema_span_late})", 
            label=f"Adaptive EMA", 
            color="darkred", 
            linewidth=2
        )
        ax2.set_title(f"Cosine Similarity Ratio (<=0) Trend(N={N})")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Ratio (<=0)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    MODE = "FLTrust_quantized_perp"
    N = 10
    iter_end = 390

    target_file_path: str = f"checkpoints/quantize_perp/{N}/logs/FLTrust_quantized_perp_N{N}_values_log.jsonl"
    save_path: str = f"{MODE}_N{N}_visualization_cos_result.png"

    analyzer: LogAnalyzer = LogAnalyzer(target_file_path)
    analyzer.process_file(iter_end)
    analyzer.visualize(save_path, N)