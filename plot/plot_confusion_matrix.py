import numpy as np
import matplotlib.pyplot as plt

def plot_voxel_confusion_matrix(cm, labels, output_path):
    cm = np.array(cm)
    cm_plot = cm #/ 1000.0
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=26)
    #cbar.ax.set_title(r'$\times 10^3$', fontsize=26, pad=15)
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(labels, fontsize=26)
    ax.set_yticklabels(labels, fontsize=26)
    ax.set_ylabel('True Source Count', fontsize=26, labelpad=8)
    ax.set_xlabel('Predicted Source Count', fontsize=26, labelpad=8)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    thresh = cm_plot.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_plot[i, j]
            ax.text(j, i, str(int(val)),
                   ha="center", va="center",
                   rotation=45,
                   color="white" if val > thresh else "black",
                   fontsize=22)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved voxel source count confusion matrix to {output_path}")

def plot_combined_confusion_matrix(cm, labels, title, output_path):
    """Plot and save combined confusion matrix."""
    cm = np.array(cm)
    cm_plot = cm / 1000.0
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.set_title(r'$\times 10^3$', fontsize=24, pad=15)
    
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(labels, fontsize=24)
    ax.set_yticklabels(labels, fontsize=24)
    #ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Source Count', fontsize=24, labelpad=8)
    ax.set_xlabel('Predicted Source Count', fontsize=24, labelpad=8)
    
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    thresh = cm_plot.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_plot[i, j]
            ax.text(j, i, f"{val:.3g}",
                   ha="center", va="center",
                   color="white" if val > thresh else "black",
                   fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved combined confusion matrix to {output_path}")

def plot_slot_confusion_matrix(conf_cm, conf_cm_path):
    conf_cm_plot = conf_cm / 1000.0
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(conf_cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=26)
    cbar.ax.set_title(r'$\times 10^3$', fontsize=24, pad=15)
    
    labels = ['Empty', 'Valid']
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(labels, fontsize=30)
    ax.set_yticklabels(labels, fontsize=30)
    ax.set_ylabel('Ground truth', fontsize=30, labelpad=8)
    ax.set_xlabel('Prediction', fontsize=30, labelpad=8)
    
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    
    thresh = conf_cm_plot.max() / 2.
    for i in range(2):
        for j in range(2):
            val = conf_cm_plot[i, j]
            ax.text(j, i, f"{val:.3g}",
                   ha="center", va="center",
                   color="white" if val > thresh else "black",
                   fontsize=30)
    
    plt.tight_layout()
    conf_cm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(conf_cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved slot confusion matrix to {conf_cm_path}")
