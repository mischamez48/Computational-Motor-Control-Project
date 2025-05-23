
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

    log_path = './logs/exercise8/'

    w_range = np.linspace(0, 2, 20)
    frequencies = np.linspace(3.5, 10, 100)
    
    pars_list = []

    for i, w in enumerate(w_range):
        for j, freq in enumerate(frequencies):
            sim_index = i*len(frequencies) + j
            w_ipsi = -w
            w_contra = w
            entraining_signals = define_entraining_signals(n_iterations=5001, frequency=freq, amplitude_degrees=45, plot_signals=False)
            pars_list.append(SimulationParameters(
                simulation_i=sim_index,
                n_iterations=5001,
                log_path=log_path,
                video_record=False,
                headless=True,
                controller="abstract oscillator",
                print_metrics=False,
                compute_metrics='all',
                cpg_amplitude_gain=REF_JOINT_AMP,
                feedback_weights_ipsi=w_ipsi,
                feedback_weights_contra=w_contra,
                entraining_signals=entraining_signals,
            ))

    run_multiple(pars_list, num_process=NUM_PROCESS)
        # Step 1: Load all results
    results = []
    reference_freq = None

    for i, w in enumerate(w_range):
        for j, entrain_freq in enumerate(frequencies):
            sim_index = i * len(frequencies) + j
            controller = load_object(os.path.join(log_path, f"controller{sim_index}"))
            neural_freq = controller.metrics["neur_frequency"]

            if i == 0:  # w = 0, used as reference
                reference_freq = neural_freq

            results.append([entrain_freq, w, neural_freq])

    results = np.array(results)  # shape: (N, 3)

    # Step 2: Compute frequency differences
    entrain_freqs = results[:, 0]
    feedbacks = results[:, 1]
    neural_freqs = results[:, 2]
    df_entrain = entrain_freqs - reference_freq
    df_neural = neural_freqs - reference_freq

    # Build results array for plot_2d: [entrain_freq, feedback_gain, delta_neural_freq]
    data_2d = np.column_stack([entrain_freqs, feedbacks, df_neural])

    plt.figure("neural_freq_vs_entrainment")
    plot_2d(
        results=data_2d,
        labels=["Entrainment Frequency [Hz]", "Feedback Gain w", "Δ Neural Frequency [Hz]"],
        cmap='viridis'
    )
    plt.title("Δ Neural Frequency vs Entrainment and Feedback Gain")
    plt.tight_layout()
    os.makedirs("./plots/exercise8/", exist_ok=True)
    plt.savefig("./plots/exercise8/entrainment_2d_map.png")
    plt.close()

    print("2D entrainment heatmap saved to ./plots/exercise8/")

    plt.figure("entrainment_tracking_scatter")
    plt.scatter(df_entrain, df_neural, c=feedbacks, cmap="viridis", s=10)
    plt.plot([0, 6], [0, 6], 'k--', label="Perfect entrainment")  # y = x line   # PEFECT ENTREAINMENT ??
    plt.xlabel("Δf_entrain = f_ext - f_ref")
    plt.ylabel("Δf_neural = f_cpg - f_ref")
    cbar = plt.colorbar()
    cbar.set_label("Feedback Gain w")
    plt.title("CPG Entrainment Behavior")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/exercise8/entrainment_tracking_scatter.png")
    plt.close()



if __name__ == '__main__':

    exercise8()
