"""
MantiQ-Auth: Multi-Modality Plot Generator
Generates high-quality publication-ready plots for multi-modality evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
SUMMARY_FILE = OUTPUT_DIR / "statistical_evaluation_summary.json"

def main():
    if not SUMMARY_FILE.exists():
        print(f"Error: {SUMMARY_FILE} not found. Please run run_massive_evaluation.py first.")
        return

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    modalities = list(data.keys())
    
    # Extract JPEG FPRs
    q90_fpr = [data[m]["jpeg_90"]["mean"] * 100 for m in modalities]
    q80_fpr = [data[m]["jpeg_80"]["mean"] * 100 for m in modalities]
    q70_fpr = [data[m]["jpeg_70"]["mean"] * 100 for m in modalities]
    
    # Extract JPEG FNRs
    fnr = [data[m]["fnr"]["mean"] * 100 for m in modalities]
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })
    
    # Create Figure 1: FPR Grouped Bar Chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(modalities))
    width = 0.25
    
    rects1 = ax1.bar(x - width, q90_fpr, width, label='JPEG Q=90', color='#4c72b0')
    rects2 = ax1.bar(x, q80_fpr, width, label='JPEG Q=80', color='#dd8452')
    rects3 = ax1.bar(x + width, q70_fpr, width, label='JPEG Q=70', color='#c44e52')
    
    ax1.set_ylabel('False Positive Rate (FPR %)')
    ax1.set_title('Robustness Comparison under Lossy JPEG Compression')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modalities)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    fig_path1 = OUTPUT_DIR / "multimodality_fpr_comparison.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Saved FPR plot to {fig_path1}")
    
    # Create Figure 2: FNR Bar Chart
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    colors = ['#55a868', '#4c72b0', '#c44e52', '#dd8452']
    rects_fnr = ax2.bar(modalities, fnr, color=colors, width=0.5, edgecolor='grey', alpha=0.9)
    
    ax2.set_ylabel('False Negative Rate (FNR %)')
    ax2.set_title('Security Vulnerability (FNR % under Localized Tampering)')
    ax2.set_ylim(0, 100)
    
    for rect in rects_fnr:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    fig_path2 = OUTPUT_DIR / "multimodality_fnr_comparison.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Saved FNR plot to {fig_path2}")

if __name__ == "__main__":
    main()
