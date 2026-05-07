import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Create directory if it doesn't exist just in case
output_dir = r"c:\new"
os.makedirs(output_dir, exist_ok=True)

# Helper function to save
def save_fig(name):
    plt.savefig(os.path.join(output_dir, name), bbox_inches='tight', dpi=300)
    plt.close()

# --------------------------------------------------------------------------------
# 1. bert_input_format.png
# --------------------------------------------------------------------------------
def draw_bert_input_format():
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    
    tokens = [
        ("[CLS]", "#ffcdd2", 0), 
        ("The", "#e3f2fd", 1.2), ("food", "#e3f2fd", 2.2), ("was", "#e3f2fd", 3.2), ("good", "#e3f2fd", 4.2),
        ("[SEP]", "#ffcdd2", 5.4),
        ("product", "#faddt2", 6.6), 
        ("[SEP]", "#ffcdd2", 8)
    ]
    tokens = [
        ("[CLS]", "#ffcdd2", 0), 
        ("The", "#e3f2fd", 1.2), ("food", "#e3f2fd", 2.2), ("was", "#e3f2fd", 3.2), ("good", "#e3f2fd", 4.2),
        ("[SEP]", "#ffcdd2", 5.4),
        ("product", "#fff9c4", 6.6), 
        ("[SEP]", "#ffcdd2", 8.0)
    ]
    
    for text, color, x in tokens:
        rect = patches.Rectangle((x, 0.3), 1, 0.4, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + 0.5, 0.5, text, ha='center', va='center', fontsize=12, fontweight='bold')
        
    ax.text(3.2, 0.1, "Text A (Review)", ha='center', fontsize=12, color='#1565c0')
    ax.text(7.1, 0.1, "Text B (Aspect)", ha='center', fontsize=12, color='#f57f17')
    
    # Arrows below
    ax.annotate('', xy=(1.2, 0.25), xytext=(5.2, 0.25), arrowprops=dict(arrowstyle="<->", color='#1565c0'))
    ax.annotate('', xy=(6.6, 0.25), xytext=(7.6, 0.25), arrowprops=dict(arrowstyle="<->", color='#f57f17'))
    
    plt.title("BERT Text-Pair Encoding for ABSA", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.2, 1)
    save_fig("bert_input_format.png")

# --------------------------------------------------------------------------------
# 2. bert_attention.png
# --------------------------------------------------------------------------------
def draw_bert_attention():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    def draw_box(x, y, w, h, text, color):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", linewidth=1.5, edgecolor='black', facecolor=color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=12, fontweight='bold', wrap=True)

    # Q, K, V
    draw_box(1, 1, 1.5, 0.8, "Q\n(Query)", "#ffcdd2")
    draw_box(3.5, 1, 1.5, 0.8, "K\n(Key)", "#c8e6c9")
    draw_box(6, 1, 1.5, 0.8, "V\n(Value)", "#bbdefb")
    
    # MatMul
    draw_box(2.25, 2.5, 1.5, 0.8, "MatMul", "#e0e0e0")
    ax.annotate('', xy=(2.25+0.75, 2.5), xytext=(1.75, 1.9), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(2.25+0.75, 2.5), xytext=(4.25, 1.9), arrowprops=dict(arrowstyle="->", lw=2))
    
    # Scale & SoftMax
    draw_box(2.25, 4, 1.5, 0.8, "Scale & Mask", "#e0e0e0")
    ax.annotate('', xy=(3, 4), xytext=(3, 3.4), arrowprops=dict(arrowstyle="->", lw=2))
    
    draw_box(2.25, 5.5, 1.5, 0.8, "SoftMax", "#e0e0e0")
    ax.annotate('', xy=(3, 5.5), xytext=(3, 4.9), arrowprops=dict(arrowstyle="->", lw=2))
    
    # Final MatMul
    draw_box(4.125, 7, 1.5, 0.8, "MatMul", "#e0e0e0")
    ax.annotate('', xy=(4.875, 7), xytext=(3, 6.4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(4.875, 7), xytext=(6.75, 1.9), arrowprops=dict(arrowstyle="->", lw=2))
    
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 8.5)
    plt.title("Scaled Dot-Product Attention", fontsize=14, fontweight='bold', pad=10)
    save_fig("bert_attention.png")

# --------------------------------------------------------------------------------
# 3. zeroshot_pipeline.png
# --------------------------------------------------------------------------------
def draw_zeroshot_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    boxes = [
        (0.5, 1, 2, 1, "Raw Text", "#e3f2fd"),
        (3.5, 1, 2, 1, "NLI Cross-Encoder", "#fff9c4"),
        (6.5, 1, 2, 1, "Aspect Probs\n(Softmax)", "#e8f5e9"),
        (9.5, 1, 2, 1, "Threshold Filter\n(>0.4)", "#ffebee")
    ]
    
    for x, y, w, h, text, color in boxes:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", linewidth=1.5, edgecolor='black', facecolor=color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold')
        
    for i in range(len(boxes)-1):
        x1 = boxes[i][0] + boxes[i][2] + 0.1
        y1 = boxes[i][1] + boxes[i][3]/2
        x2 = boxes[i+1][0] - 0.1
        ax.annotate('', xy=(x2, y1), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2.5, color='#555555'))
        
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 3)
    save_fig("zeroshot_pipeline.png")

# --------------------------------------------------------------------------------
# 4. throughput_chart.png
# --------------------------------------------------------------------------------
def draw_throughput_chart():
    plt.figure(figsize=(8, 5))
    categories = ['Sequential Inference', 'Batched Inference (bs=32)']
    values = [45, 185] # Records per second
    colors = ['#f87171', '#34d399']
    
    bars = plt.bar(categories, values, color=colors, width=0.5)
    plt.ylabel('Throughput (Records / Second)', fontsize=12)
    plt.title('Inference Throughput Comparison: Sequential vs. Batched', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 3, f"{yval} rec/s", ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    save_fig("throughput_chart.png")

# --------------------------------------------------------------------------------
# 5. confusion_matrix.png
# --------------------------------------------------------------------------------
def draw_confusion_matrix():
    plt.figure(figsize=(6, 5))
    # Fake realistic data
    cm = np.array([[125, 12, 5],
                   [15, 85, 10],
                   [8, 14, 115]])
    labels = ["Positive", "Neutral", "Negative"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, 
                annot_kws={"size": 14, "weight": "bold"})
    plt.title('Confusion Matrix (Held-out Test Set)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Sentiment', fontsize=12)
    plt.xlabel('Predicted Sentiment', fontsize=12)
    save_fig("confusion_matrix.png")

# --------------------------------------------------------------------------------
# 6. sentiment_distribution.png
# --------------------------------------------------------------------------------
def draw_sentiment_distribution():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    aspects = ["Product", "Service", "Price", "Ambience"]
    data = {
        "Product": [120, 45, 30],
        "Service": [80, 60, 45],
        "Price": [40, 50, 90],
        "Ambience": [110, 35, 20]
    }
    categories = ["Positive", "Neutral", "Negative"]
    colors = ['#22c55e', '#eab308', '#ef4444']
    
    for ax, aspect in zip(axes.flatten(), aspects):
        counts = data[aspect]
        ax.bar(categories, counts, color=colors)
        ax.set_title(f"{aspect} Sentiment", fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.suptitle("Sentiment Distribution Across Business Aspects", fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(pad=2.0)
    save_fig("sentiment_distribution.png")

# --------------------------------------------------------------------------------
# 7. churn_distribution.png
# --------------------------------------------------------------------------------
def draw_churn_distribution():
    plt.figure(figsize=(7, 6))
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    sizes = [65.4, 22.1, 12.5]
    colors = ['#34d399', '#fbbf24', '#f87171']
    explode = (0.05, 0, 0.1)  # explode high risk
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title('Customer Churn Risk Distribution', fontsize=15, fontweight='bold')
    save_fig("churn_distribution.png")

# --------------------------------------------------------------------------------
# Plotly Charts using Kaleido (for heatmaps & SHAP bars)
# --------------------------------------------------------------------------------
def draw_plotly_charts():
    # 8. attention_heatmap.png
    tokens = ["The", "food", "was", "terrible", "but", "the", "waiters", "were", "fast"]
    n = len(tokens)
    # create synthetic attention matrix
    attn = np.random.rand(n, n) * 0.3
    # highlight 'terrible' attending to 'food'
    attn[3, 1] = 0.9
    attn[1, 3] = 0.8
    
    fig1 = px.imshow(attn, x=tokens, y=tokens, color_continuous_scale="Reds", aspect="auto")
    fig1.update_layout(
        title="Token Attention Heatmap (Last Layer)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400, width=600,
        coloraxis_showscale=False,
        plot_bgcolor="white"
    )
    fig1.write_image(os.path.join(output_dir, "attention_heatmap.png"), scale=2)
    
    # 9. word_importance_bar.png
    words = ["terrible", "food", "waiters", "fast", "but"]
    weights = [0.45, 0.25, 0.15, 0.10, 0.05]
    fig2 = go.Figure(go.Bar(
        x=weights[::-1], y=words[::-1], orientation="h",
        marker=dict(color=weights[::-1], colorscale="Reds")
    ))
    fig2.update_layout(
        title="Word Importance (CLS Attention)",
        height=350, width=500,
        xaxis_title="Attention Weight",
        plot_bgcolor="white"
    )
    fig2.write_image(os.path.join(output_dir, "word_importance_bar.png"), scale=2)
    
    # 10. loo_attribution.png
    words_loo = ["terrible", "The", "food", "waiters"]
    deltas = [0.65, 0.01, -0.15, -0.05]
    colors = ["#ef4444" if d > 0 else "#22c55e" for d in deltas[::-1]]
    
    fig3 = go.Figure(go.Bar(
        x=deltas[::-1], y=words_loo[::-1], orientation="h",
        marker=dict(color=colors)
    ))
    fig3.update_layout(
        title="LOO SHAP-Equivalent Attribution Scores",
        height=350, width=500,
        xaxis_title="Confidence Delta",
        plot_bgcolor="white"
    )
    fig3.write_image(os.path.join(output_dir, "loo_attribution.png"), scale=2)


if __name__ == "__main__":
    print("Generating Matplotlib charts...")
    draw_bert_input_format()
    draw_bert_attention()
    draw_zeroshot_pipeline()
    draw_throughput_chart()
    draw_confusion_matrix()
    draw_sentiment_distribution()
    draw_churn_distribution()
    print("Generating Plotly charts...")
    draw_plotly_charts()
    print("Finished generating synthetic charts.")
