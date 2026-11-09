import matplotlib.pyplot as plt

def plot_iou_vs_source_count(df_iou, output_path, title="IoU vs. True Source Count", ylabel="IoU of matched slots"):
    """
    Generates a boxplot showing the IoU of true positive slots 
    grouped by the true number of sources in the sample.
    """
    plt.figure(figsize=(6, 6))
    
    try:
        import seaborn as sns
        sns.boxplot(data=df_iou, x="num_sources", y="iou", hue="num_sources", palette="Blues", legend=False)
    except ImportError:
        groups = [group["iou"].values for _, group in df_iou.groupby("num_sources")]
        labels = [str(k) for k, _ in df_iou.groupby("num_sources")]
        plt.boxplot(groups, labels=labels, patch_artist=True)
    
    #plt.title(title, fontsize=16)
    plt.xlabel("Number of true sources", fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved IoU vs. Source Count plot to {output_path}")