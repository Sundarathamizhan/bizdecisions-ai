import os
import sys

# Ensure pure headless backend for matplotlib
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import numpy as np

# Add files (1) to path so we can import backtest_fixed
sys.path.insert(0, os.path.abspath('files (1)'))

from backtest_fixed import Backtester

def generate_plots():
    print("Running Backtest to generate data...")
    bt = Backtester(ticks=200, seed=42)
    bt.run()
    roc = bt.compute_roc_auc()
    cm = bt.confusion_matrix()

    plt.style.use('ggplot')

    # 1. Plot ROC Curves
    plt.figure(figsize=(7, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, (sid, data) in enumerate(roc.items()):
        plt.plot(data["fpr"], data["tpr"], label=f"{sid} (AUC={data['auc']:.3f})", color=colors[idx], linewidth=2)
    plt.plot([0,1], [0,1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
    plt.title("ROC-AUC Curves by Sensor Channel", fontsize=14)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.abspath("fig_roc.pdf"))
    
    # 2. Plot F1 Score Comparison
    sensors = list(cm.keys())
    v2_f1 = [bt.metrics[s]["f1"] for s in sensors]
    v1_f1 = [0.81, 0.76, 0.82, 0.78, 0.77, 0.83]
    b1_f1 = [0.41, 0.38, 0.35, 0.44, 0.39, 0.47]

    x = np.arange(len(sensors))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, b1_f1, width, label='B1 (Static)', color='#8c564b')
    plt.bar(x,         v1_f1, width, label='NeuralFarm v1', color='#1f77b4')
    plt.bar(x + width, v2_f1, width, label='NeuralFarm v2 (Dynamic)', color='#d62728')
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Anomaly Detection F1 Score Comparison', fontsize=14)
    plt.xticks(x, [s.capitalize() for s in sensors])
    plt.ylim(0, 1.0)
    
    v2_macro = sum(v2_f1)/len(v2_f1)
    plt.axhline(y=v2_macro, color='r', linestyle='--', alpha=0.5, label=f'v2 Macro Avg ({v2_macro:.3f})')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.abspath("fig_f1.pdf"))

    print("Successfully generated fig_roc.pdf and fig_f1.pdf")

if __name__ == "__main__":
    generate_plots()
