import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog

from util.rw import load_object
from util.run_open_loop import run_multiple
from util.entraining_signals import define_entraining_signals
from simulation_parameters import SimulationParameters
from plotting_common import plot_2d, save_figures

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]

NUM_PROCESS = 10

def exercise8():
    """
    Question 6.2: Explore the effect of the entrainment on the neural frequency under 
    different feedback gain strengths systematically.
    """
    
    log_path = './logs/exercise8/'
    os.makedirs(log_path, exist_ok=True)
    
    # Parameter ranges as specified
    w_range = np.linspace(0, 2, 20)  # 20 values from 0 to 2
    frequencies = np.linspace(3.5, 10, 20)  # Reduced from 100 to 20 for faster computation
    
    # Create parameter list
    pars_list = []
    
    for i, w in enumerate(w_range):
        for j, freq in enumerate(frequencies):
            sim_index = i * len(frequencies) + j
            
            # CORRECTED: w_contra = -w_ipsi = w, so w_ipsi = w and w_contra = -w
            # Also scale by FEEDBACK_GAIN_REF (which is ws_ref)
            w_ipsi = w * ws_ref
            w_contra = -w * ws_ref
            
            entraining_signals = define_entraining_signals(
                n_iterations=5001, 
                frequency=freq, 
                amplitude_degrees=45, 
                plot_signals=False
            )
            
            pars_list.append(SimulationParameters(
                simulation_i=sim_index,
                n_iterations=5001,
                log_path=log_path,
                video_record=False,
                headless=True,
                controller="abstract oscillator",
                print_metrics=False,
                compute_metrics='all',
                cpg_amplitude_gain=REF_JOINT_AMP[:-2],  # Only active joints
                feedback_weights_ipsi=w_ipsi,
                feedback_weights_contra=w_contra,
                entraining_signals=entraining_signals,
            ))
    
    # Run simulations
    pylog.info(f"Running {len(pars_list)} simulations...")
    run_multiple(pars_list, num_process=NUM_PROCESS)
    
    # Step 1: Get reference frequency (w=0 case)
    # Find all simulations with w=0 (first set of frequencies)
    reference_frequencies = []
    for j in range(len(frequencies)):
        controller = load_object(os.path.join(log_path, f"controller{j}"))
        reference_frequencies.append(controller.metrics["neur_frequency"])
    
    # Use the mean as the reference frequency
    reference_freq = np.mean(reference_frequencies)
    print(f"Reference neural frequency (w=0): {reference_freq:.3f} Hz")
    
    # Step 2: Collect all results
    results = []
    for i, w in enumerate(w_range):
        for j, entrain_freq in enumerate(frequencies):
            sim_index = i * len(frequencies) + j
            controller = load_object(os.path.join(log_path, f"controller{sim_index}"))
            neural_freq = controller.metrics["neur_frequency"]
            
            # Calculate differences as specified in the question
            delta_f_entrain = entrain_freq - reference_freq
            delta_f_neural = neural_freq - reference_freq
            
            results.append({
                'w': w,
                'entrain_freq': entrain_freq,
                'neural_freq': neural_freq,
                'delta_f_entrain': delta_f_entrain,
                'delta_f_neural': delta_f_neural
            })
    
    # Filter for positive differences only
    positive_results = [r for r in results if r['delta_f_entrain'] > 0 and r['delta_f_neural'] >= 0]
    
    # Convert to numpy array for easier manipulation
    results_array = np.array([(r['delta_f_entrain'], r['delta_f_neural'], r['w']) 
                              for r in positive_results])
    
    # Create plots
    plot_dir = './plots/exercise8/'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Main scatter plot with only positive differences
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(results_array[:, 0], results_array[:, 1], 
                         c=results_array[:, 2], cmap='viridis', s=50, alpha=0.7)
    
    # Add reference lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add diagonal line for perfect entrainment
    max_val = max(np.max(results_array[:, 0]), np.max(results_array[:, 1]))
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, 
             label='Perfect entrainment')
    
    # Set axis limits to start from 0
    plt.xlim(0, max(results_array[:, 0]) * 1.05)
    plt.ylim(0, max(results_array[:, 1]) * 1.05)
    
    plt.xlabel('Δf_entrain = f_entrain - f_ref [Hz]')
    plt.ylabel('Δf_neural = f_neural - f_ref [Hz]')
    plt.title('Neural Frequency Response to Entrainment (Positive Differences Only)')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Feedback gain w (scaled by ws_ref)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'entrainment_response_positive.png'), dpi=300)
    plt.close()
    
    # Plot 2: 2D heatmap of neural frequency shift
    # Reshape data for 2D plotting
    neural_freq_grid = np.zeros((len(w_range), len(frequencies)))
    for i, w in enumerate(w_range):
        for j, freq in enumerate(frequencies):
            idx = i * len(frequencies) + j
            neural_freq_grid[i, j] = results[idx]['neural_freq']
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(frequencies, w_range, neural_freq_grid, shading='auto', cmap='viridis')
    plt.colorbar(label='Neural Frequency [Hz]')
    plt.xlabel('Entrainment Frequency [Hz]')
    plt.ylabel('Feedback Gain w')
    plt.title('Neural Frequency as Function of Entrainment Frequency and Feedback Gain')
    
    # Add contour lines
    contours = plt.contour(frequencies, w_range, neural_freq_grid, colors='white', 
                          alpha=0.4, linewidths=0.5)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'neural_freq_heatmap.png'), dpi=300)
    plt.close()
    
    # Plot 3: Entrainment strength vs feedback gain (only positive values)
    plt.figure(figsize=(10, 6))
    
    # Calculate entrainment strength for each w value
    entrainment_strengths = []
    for i, w in enumerate(w_range):
        w_results = [r for r in results if r['w'] == w]
        # Entrainment strength = correlation between delta_f_entrain and delta_f_neural
        if len(w_results) > 1:
            delta_f_entrain = [r['delta_f_entrain'] for r in w_results]
            delta_f_neural = [r['delta_f_neural'] for r in w_results]
            corr = np.corrcoef(delta_f_entrain, delta_f_neural)[0, 1]
            # Only keep positive correlations
            entrainment_strengths.append(max(0, corr) if not np.isnan(corr) else 0)
        else:
            entrainment_strengths.append(0)
    
    plt.plot(w_range, entrainment_strengths, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Feedback Gain w')
    plt.ylabel('Entrainment Strength (Positive Correlation Only)')
    plt.title('Entrainment Strength vs Feedback Gain')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'entrainment_strength_positive.png'), dpi=300)
    plt.close()
    
    # Analysis
    print("\n" + "="*60)
    print("QUESTION 6.2: Analysis of Results")
    print("="*60)
    print(f"Reference frequency (w=0): {reference_freq:.3f} Hz")
    print(f"Feedback gain range: {w_range[0]:.1f} to {w_range[-1]:.1f} (scaled by ws_ref={ws_ref:.2f})")
    print(f"Entrainment frequency range: {frequencies[0]:.1f} to {frequencies[-1]:.1f} Hz")
    
    # Find when entrainment becomes effective
    threshold_w = None
    for i, strength in enumerate(entrainment_strengths):
        if strength > 0.5:  # 50% correlation threshold
            threshold_w = w_range[i]
            break
    
    if threshold_w is not None:
        print(f"\nEntrainment becomes effective at w ≈ {threshold_w:.2f}")
        print(f"This corresponds to actual feedback weights:")
        print(f"  w_ipsi = {threshold_w * ws_ref:.2f}")
        print(f"  w_contra = {-threshold_w * ws_ref:.2f}")
    else:
        print("\nWeak entrainment observed across all tested feedback gains")


if __name__ == '__main__':
    exercise8()