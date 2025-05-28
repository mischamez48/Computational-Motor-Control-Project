import os
import numpy as np
import farms_pylog as pylog

from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def visualize_swimming_comparison():
    """
    Run simulations with GUI to visualize different swimming modes
    """
    
    # 1. Normal CPG swimming
    print("\n" + "="*60)
    print("VISUALIZATION 1: Normal CPG Swimming")
    print("Close the window to continue to next simulation")
    print("="*60)
    
    pars_normal = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        compute_metrics='all',
        print_metrics=True,
        headless=False,  # Show GUI
        video_record=False,  # Don't save video
        fast=False,  # Real-time speed
        cpg_amplitude_gain=REF_JOINT_AMP[:-2],
        weights_body2body=30,
        weights_body2body_contralateral=10,
        feedback_weights_ipsi=1.0,
        feedback_weights_contra=-1.0,
    )
    
    controller_normal = run_single(pars_normal)
    
    # 2. No ipsilateral connections
    print("\n" + "="*60)
    print("VISUALIZATION 2: Swimming without Ipsilateral Connections")
    print("Close the window to continue to next simulation")
    print("="*60)
    
    pars_no_ipsi = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        compute_metrics='all',
        print_metrics=True,
        headless=False,  # Show GUI
        video_record=False,
        fast=False,
        cpg_amplitude_gain=REF_JOINT_AMP[:-2],
        weights_body2body=0,  # No ipsilateral
        weights_body2body_contralateral=10,
        feedback_weights_ipsi=1.0 * ws_ref,  # w=1.0 (best from results)
        feedback_weights_contra=-1.0 * ws_ref,
        initial_phases=np.random.uniform(-0.1, 0.1, 26),
    )
    
    controller_no_ipsi = run_single(pars_no_ipsi)
    
    # 3. No contralateral connections
    print("\n" + "="*60)
    print("VISUALIZATION 3: Swimming without Contralateral Connections")
    print("Close the window to continue to next simulation")
    print("="*60)
    
    pars_no_contra = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        compute_metrics='all',
        print_metrics=True,
        headless=False,  # Show GUI
        video_record=False,
        fast=False,
        cpg_amplitude_gain=REF_JOINT_AMP[:-2],
        weights_body2body=30,
        weights_body2body_contralateral=0,  # No contralateral
        feedback_weights_ipsi=0.6 * ws_ref,  # w=0.6 (from results)
        feedback_weights_contra=-0.6 * ws_ref,
        initial_phases=np.random.uniform(-0.1, 0.1, 26),
    )
    
    controller_no_contra = run_single(pars_no_contra)
    
    # 4. CPG-free swimming
    print("\n" + "="*60)
    print("VISUALIZATION 4: CPG-Free Swimming (No CPG connections)")
    print("Close the window to continue")
    print("="*60)
    
    pars_cpg_free = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        compute_metrics='all',
        print_metrics=True,
        headless=False,  # Show GUI
        video_record=False,
        fast=False,
        cpg_amplitude_gain=REF_JOINT_AMP[:-2],
        weights_body2body=0,  # No ipsilateral
        weights_body2body_contralateral=0,  # No contralateral
        feedback_weights_ipsi=0.6 * ws_ref,  # w=0.6
        feedback_weights_contra=-0.6 * ws_ref,
        initial_phases=np.random.uniform(-0.1, 0.1, 26),
    )
    
    controller_cpg_free = run_single(pars_cpg_free)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    
    # Print comparison summary
    print("\nSummary of Swimming Modes:")
    print(f"{'Mode':<30} {'Frequency (Hz)':<15} {'Speed (m/s)':<15}")
    print("-"*60)
    print(f"{'Normal CPG':<30} {controller_normal.metrics['neur_frequency']:<15.3f} {controller_normal.metrics['mech_speed_fwd']:<15.4f}")
    print(f"{'No Ipsilateral':<30} {controller_no_ipsi.metrics['neur_frequency']:<15.3f} {controller_no_ipsi.metrics['mech_speed_fwd']:<15.4f}")
    print(f"{'No Contralateral':<30} {controller_no_contra.metrics['neur_frequency']:<15.3f} {controller_no_contra.metrics['mech_speed_fwd']:<15.4f}")
    print(f"{'CPG-Free':<30} {controller_cpg_free.metrics['neur_frequency']:<15.3f} {controller_cpg_free.metrics['mech_speed_fwd']:<15.4f}")


def visualize_specific_case(config_name, w_value):
    """
    Visualize a specific configuration and feedback gain
    """
    configs = {
        'normal': (30, 10),
        'no_ipsi': (0, 10),
        'no_contra': (30, 0),
        'cpg_free': (0, 0)
    }
    
    w_b2b, w_contra = configs[config_name]
    
    print(f"\nVisualizing {config_name} with w={w_value}")
    
    pars = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        compute_metrics='all',
        print_metrics=True,
        headless=False,  # Show GUI
        video_record=False,
        fast=False,
        cpg_amplitude_gain=REF_JOINT_AMP[:-2],
        weights_body2body=w_b2b,
        weights_body2body_contralateral=w_contra,
        feedback_weights_ipsi=w_value * ws_ref,
        feedback_weights_contra=-w_value * ws_ref,
        initial_phases=np.random.uniform(-0.1, 0.1, 26),
    )
    
    controller = run_single(pars)
    return controller


if __name__ == '__main__':
    # Option 1: Run all comparisons
    visualize_swimming_comparison()
    
    # Option 2: Visualize specific cases
    # visualize_specific_case('cpg_free', 0.8)
    # visualize_specific_case('no_ipsi', 1.0)