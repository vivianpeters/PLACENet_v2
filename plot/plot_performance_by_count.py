@staticmethod
    def plot_iou_vs_source_count(df_iou, output_path, title="IoU vs. True Source Count"):
        """
        Generates a boxplot showing the IoU of true positive slots 
        grouped by the true number of sources in the sample.
        """
        import seaborn as sns
        plt.figure(figsize=(6, 6))
        
        # We ensure 'True Source Count' is treated as a categorical variable for plotting
        sns.boxplot(data=df_iou, x="True Source Count", y="IoU", palette="Blues")
        #sns.stripplot(data=df_iou, x="True Source Count", y="IoU", color="black", alpha=0.3, jitter=True)
        
        #plt.title(title, fontsize=16)
        plt.xlabel("Number of true sources", fontsize=28)
        plt.ylabel("IoU of matched slots", fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved IoU vs. Source Count plot to {output_path}")