import matplotlib.pyplot as plt
import numpy as np

def plot_affine_evolution(affine_params, params_to_plot, save_path):
    """
    Plot the evolution of affine transformation parameters over iterations.
    
    Args:
        affine_params (dict): Dictionary containing parameter evolution data
        params_to_plot (list): List of parameter names to plot
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, param in enumerate(params_to_plot):
        ax = axes[idx]
        for scan_num, data in affine_params.items():
            if param in data:
                ax.plot(data['iterations'], data[param], 'o-', 
                        label=f'Scan {scan_num}', markersize=4, alpha=0.7)
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel(param.capitalize())
        ax.set_title(f'{param.capitalize()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved affine evolution plot to: {save_path}")

def plot_affine_summary(affine_params, params_to_plot, save_path):
    """
    Plot the final values of affine transformation parameters for each scan.
    
    Args:
        affine_params (dict): Dictionary containing parameter data
        params_to_plot (list): List of parameter names to plot
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, param in enumerate(params_to_plot):
        ax = axes[idx]
        final_affine_values = {scan_num: data[param][-1] 
                              for scan_num, data in affine_params.items() 
                              if param in data}
        
        if final_affine_values:
            ax.plot(list(final_affine_values.keys()), 
                    list(final_affine_values.values()), 
                    'o-', markersize=8)
            
            mean_val = np.mean(list(final_affine_values.values()))
            std_val = np.std(list(final_affine_values.values()))
            
            ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Scan Number')
        ax.set_ylabel(f'Final {param.capitalize()}')
        ax.set_title(f'Final {param.capitalize()} for Each Scan')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved affine summary plot to: {save_path}") 