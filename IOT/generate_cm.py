import os
import sys
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrices():
    # V2 CM (macro-aggregated across all sensors from backtesting)
    # Based on table III totals: TP=87, FP=13, FN=19, TN=1081
    v2_cm = np.array([[1081, 13],    # TN, FP
                      [  19, 87]])   # FN, TP

    # B1 CM (Static baseline)
    # B1 F1 score is ~0.41. High FP and FN.
    b1_cm = np.array([[ 980, 114],   # TN, FP
                      [  45,  61]])  # FN, TP

    plt.style.use('default') # Use clean default style for heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    def plot_heatmap(ax, cm, title):
        # We manually build the heatmap logic to avoid seaborn dependency
        cax = ax.matshow(cm, cmap='Blues', vmin=0, vmax=1100)
        for (i, j), z in np.ndenumerate(cm):
            # Print text - white for dark cells, black for lighter cells
            color = 'white' if z > 400 or (j==1 and i==1 and z>80) else 'black'
            ax.text(j, i, f'{z}', ha='center', va='center', fontsize=14, color=color, fontweight='bold')
        
        ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Normal', 'Pred Anomaly'], fontsize=12)
        ax.set_yticklabels(['True Normal', 'True Anomaly'], rotation=90, va='center', fontsize=12)
        
        # Move x-axis labels to bottom
        ax.xaxis.set_ticks_position('bottom')
        ax.grid(False) # NO gridlines for heatmaps

    plot_heatmap(axes[0], b1_cm, 'Baseline B1 (Static Threshold)')
    plot_heatmap(axes[1], v2_cm, 'NeuralFarm v2 (Dynamic)')

    plt.tight_layout()
    plt.savefig('fig_cm.pdf')
    print("Generated fig_cm.pdf")

if __name__ == "__main__":
    plot_confusion_matrices()
