

"""
File of metrics computations
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, fftconvolve


POINTS_POS = np.array(
    [
        0.000,
        0.003, 0.004, 0.005,
        0.006, 0.007, 0.008,
        0.009, 0.010, 0.011,
        0.012, 0.013, 0.014,
        0.015, 0.016, 0.017,
        0.018,
    ]
)

BODY_LENGTH = POINTS_POS[-1] - POINTS_POS[0]
JOINTS_POS = POINTS_POS[1:-1]
LINKS_COM_POS = (POINTS_POS[1:] + POINTS_POS[:-1]) / 2

LINKS_MASSES = np.array(
    [
        3.27e-06, 4.05e-06, 4.85e-06, 5.15e-06,
        5.49e-06, 5.47e-06, 5.15e-06, 4.24e-06,
        3.40e-06, 2.65e-06, 1.73e-06, 1.17e-06,
        8.25e-07, 9.19e-07, 9.36e-07, 8.46e-07,
    ]
)

TOTAL_MASS = np.sum(LINKS_MASSES)

# ------ SIGNAL PROCESSING TOOLS ------


def get_filtered_signals(
    signals: np.ndarray,
    signal_dt: float,
    fcut_hp: float = None,
    fcut_lp: float = None,
    filt_order: int = 5,
):
    ''' Butterwort, zero-phase filtering '''

    # Nyquist frequency
    fnyq = 0.5 / signal_dt

    # Filters
    if fcut_hp is not None:
        num, den = butter(filt_order, fcut_hp/fnyq,  btype='highpass')
        signals = filtfilt(num, den, signals)

    if fcut_lp is not None:
        num, den = butter(filt_order, fcut_lp/fnyq, btype='lowpass')
        signals = filtfilt(num, den, signals)

    return signals


def get_phase_lag(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sig_dt: float,
    sig_freq: float,
):
    ''' Compute the phase lag between two signals '''
    xcorr = np.correlate(sig2, sig1, "full")
    n_lag = np.argmax(xcorr) - len(xcorr) // 2
    t_lag = n_lag * sig_dt
    phase_lag = t_lag * sig_freq

    # Delay > Period --> No phase lag can be computed
    if abs(phase_lag) > 1.0:
        phase_lag = 0.0

    return phase_lag


def remove_signals_offset(signals: np.ndarray):
    ''' Removed offset from the signals '''
    return (signals.T - np.mean(signals, axis=1)).T


def get_body_linear_fit_step(
    coordinates_xy: np.ndarray,
    n_links_pca: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the PCA of the links positions at a given step '''

    cov_mat = np.cov(
        [
            coordinates_xy[step, :n_links_pca, 0],
            coordinates_xy[step, :n_links_pca, 1],
        ]
    )
    eig_values, eig_vecs = np.linalg.eig(cov_mat)
    largest_index = np.argmax(eig_values)
    direction_fwd = eig_vecs[:, largest_index]

    # Align the direction with the tail-head axis
    p_tail2head = coordinates_xy[step, 0] - coordinates_xy[step, n_links_pca-1]
    direction_sign = np.sign(np.dot(p_tail2head, direction_fwd))
    direction_fwd = direction_sign * direction_fwd

    direction_left = np.cross(
        [0, 0, 1],
        [direction_fwd[0], direction_fwd[1], 0]
    )[:2]

    return direction_fwd, direction_left


def get_com_position(
    links_positions: np.ndarray,
):
    ''' Compute the center of mass position '''
    com_positions = np.sum(
        links_positions * LINKS_MASSES[:, np.newaxis],
        axis=1,
    ) / TOTAL_MASS
    return com_positions


def get_com_velocity(
    links_velocities: np.ndarray,
):
    ''' Compute the center of mass velocity '''
    com_velocities = np.sum(
        links_velocities * LINKS_MASSES[:, np.newaxis],
        axis=1,
    ) / TOTAL_MASS
    return com_velocities


def compute_frequency_fft(
    times: np.ndarray,
    signals: np.ndarray,
):
    ''' Computes the max frequency, index and amplitude of the FFT spectrum '''

    dt_sig = times[1]-times[0]
    n_step = len(times)

    # Filter high-frequency noise
    smooth_signals = signals.copy()
    smooth_signals = remove_signals_offset(smooth_signals)
    smooth_signals = get_filtered_signals(smooth_signals, dt_sig, fcut_lp=50)

    # Compute FFT
    signals_fft = np.fft.fft(smooth_signals, axis=1)
    freqs = np.fft.fftfreq(n_step, d=dt_sig)

    signals_fft_mod = np.abs(signals_fft)
    ind_max_signals = np.argmax(signals_fft_mod[:, :n_step//2], axis=1)
    freq_max_signals = freqs[ind_max_signals]
    amp_max_signals = 2*signals_fft_mod[:,
                                        :n_step//2][np.arange(signals_fft_mod.shape[0]),
                                                    ind_max_signals]/signals_fft_mod.shape[1]

    return freq_max_signals, ind_max_signals, amp_max_signals


def compute_neural_phase_lags(
    times: np.ndarray,
    signals: np.ndarray,
    freqs: np.ndarray,
    inds_couples: list[list[int, int]]
) -> np.ndarray:
    '''
    Computes the IPL evolution based on the cross correlation of signals.
    Returns the IPLs between adjacent signals
    '''
    n_couples = len(inds_couples)

    if not n_couples:
        return np.nan

    dt_sig = times[1] - times[0]
    ipls = np.zeros(n_couples)
    signals_f = remove_signals_offset(signals)

    for ind_couple, (ind1, ind2) in enumerate(inds_couples):
        ipls[ind_couple] = get_phase_lag(
            sig1=signals_f[ind1],
            sig2=signals_f[ind2],
            sig_dt=dt_sig,
            sig_freq=np.mean([freqs[ind1], freqs[ind2]]),
        )

    return ipls

# ------ MECHANICAL METRICS ------


def compute_travel_distance(
    links_positions
):
    """Compute total travel distance, regardless of its curvature"""
    com_pos = get_com_position(links_positions)
    com_dist = np.sum(
        np.sqrt(
            np.diff(com_pos[:, 0])**2 +
            np.diff(com_pos[:, 1])**2
        )
    )
    return com_dist


def compute_torques_sum(
    joints_torques
):
    """Compute sum of torques"""
    return np.sum(np.abs(joints_torques))


def compute_energy_sum(
    joints_torques,
    joints_velocities,
    timestep,
):
    """
    Compute sum of energy consumptions.
    #NOTE: Only take positive values (no energy storing of the active part)
    """
    powers = np.clip(joints_torques * joints_velocities, a_min=0, a_max=None)
    return np.sum(powers) * timestep


def compute_speed(
    links_positions,
    links_vel,
):
    '''
    Computes the axial and lateral speed based on the PCA of the links positions
    '''

    n_steps = links_positions.shape[0]
    n_links = links_positions.shape[1]
    n_links_trunk = n_links // 2
    links_pos_xy = links_positions[:, :, :2]
    links_vel_xy = links_vel[:, :, :2]
    com_vel_xy = get_com_velocity(links_vel_xy)

    speed_forward = []
    speed_lateral = []

    for idx in range(n_steps):

        # Compute the PCA of the links positions
        direction_fwd, direction_left = get_body_linear_fit_step(
            coordinates_xy=links_pos_xy,
            n_links_pca=n_links_trunk,
            step=idx,
        )

        # Compute the forward and lateral speed
        v_com_forward_proj = np.dot(com_vel_xy[idx], direction_fwd)
        v_com_lateral_proj = np.dot(com_vel_xy[idx], direction_left)

        speed_forward.append(v_com_forward_proj)
        speed_lateral.append(v_com_lateral_proj)

    return speed_forward, speed_lateral


def compute_mechanical_phase_lags(
    times: np.ndarray,
    signals: np.ndarray,
    freqs: np.ndarray,
    inds_couples: list[list[int, int]]
) -> np.ndarray:
    '''
    Computes the IPL evolution based on the cross correlation of signals.
    Returns the IPLs between adjacent signals
    '''
    n_couples = len(inds_couples)

    if not n_couples:
        return np.nan

    dt_sig = times[1] - times[0]
    ipls = np.zeros(n_couples)
    signals_f = remove_signals_offset(signals)

    for ind_couple, (ind1, ind2) in enumerate(inds_couples):
        ipls[ind_couple] = get_phase_lag(
            sig1=signals_f[ind1],
            sig2=signals_f[ind2],
            sig_dt=dt_sig,
            sig_freq=np.mean([freqs[ind1], freqs[ind2]]),
        )

    return ipls


# ------ OVERALL METRICS ------
def compute_neural_metrics(network):
    """ Compute all the metrics of the neural controller """

    metrics = {}
    n_iterations = network.pars.n_iterations
    sim_fraction = 0.6
    n_steps_considered = round(n_iterations * sim_fraction)

    # Consider sim_fraction number of steps
    times = network.times[-n_steps_considered:]
    motor_output = network.motor_out[-n_steps_considered:]

    # Get active joints
    motor_output_m = np.mean(motor_output, axis=0)
    inactive_osc = np.all(motor_output == motor_output_m, axis=0)
    inactive_joints = inactive_osc[::2] & inactive_osc[1::2]

    n_oscillators = motor_output.shape[1]
    n_joints = n_oscillators // 2

    first_active_joint = np.argmax(~inactive_joints)
    last_active_joint = n_joints - 1 - np.argmax(inactive_joints[::-1])
    active_body_length = JOINTS_POS[last_active_joint] - \
        JOINTS_POS[first_active_joint]
    active_body_fraction = active_body_length / BODY_LENGTH

    # Get signals for the active joints
    signals = motor_output[:, network.motor_l]-motor_output[:, network.motor_r]
    signals = signals[:, ~inactive_joints]

    n_signals = signals.shape[1]

    # Compute frequency
    neur_frequencies, _, neur_amps = compute_frequency_fft(times, signals.T)
    metrics["neur_frequency"] = np.mean(neur_frequencies)

    # Compute IPLS
    iplss = compute_neural_phase_lags(
        times=times,
        signals=signals.T,
        freqs=neur_frequencies,
        inds_couples=[[i, i+1] for i in range(n_signals-1)]
    )

    metrics["neur_ipls"] = np.mean(iplss)
    metrics["neur_twl"] = np.sum(iplss) / active_body_fraction

    # Compute amplitude
    metrics["neur_amp"] = np.mean(neur_amps)

    return metrics


def compute_mechanical_metrics(network):
    """ Compute all mechanical metrics """

    metrics = {}
    sim_fraction = 0.6
    n_iterations = network.pars.n_iterations
    n_steps_considered = round(n_iterations * sim_fraction)

    times = network.times[-n_steps_considered:-1]
    links_positions = network.links_positions[-n_steps_considered:-1]
    link_velocities = network.links_velocities[-n_steps_considered:-1]
    joints_positions = network.joints_positions[-n_steps_considered:-1]
    joints_velocities = network.joints_velocities[-n_steps_considered:-1]
    joints_active_torques = network.joints_active_torques[-n_steps_considered:-1]

    timestep = times[1] - times[0]
    duration = times[-1] - times[0]

    # Mechanical frequencies
    mech_signals = joints_positions[-n_steps_considered:, :]
    joint_frequencies, _, joint_amplitudes = compute_frequency_fft(
        times, mech_signals.T)

    frequency = np.mean(joint_frequencies)

    metrics["mech_joint_frequencies"] = joint_frequencies
    metrics["mech_mean_frequency"] = frequency

    # Forward and lateral speed
    (speed_fwd_evolution, speed_lat_evolution) = compute_speed(
        links_positions=links_positions,
        links_vel=link_velocities,
    )
    speed_fwd = np.mean(speed_fwd_evolution)
    speed_lat = np.mean(speed_lat_evolution)

    metrics["mech_speed_fwd"] = speed_fwd
    metrics["mech_speed_lat"] = speed_lat

    # Joint torques
    torque = compute_torques_sum(
        joints_torques=joints_active_torques,
    )
    metrics["mech_torque"] = torque

    # Energy consumption
    energy = compute_energy_sum(
        joints_torques=joints_active_torques,
        joints_velocities=joints_velocities,
        timestep=timestep,
    )
    metrics["mech_energy"] = energy

    # Cost of transport
    distance_fwd = np.abs(speed_fwd * duration)
    if distance_fwd != 0:
        cost_of_transport = energy / distance_fwd
    else:
        cost_of_transport = np.inf

    metrics["mech_cot"] = cost_of_transport

    # Joint oscillation amplitudes
    metrics["mech_joint_amplitudes"] = joint_amplitudes
    metrics["mech_mean_amplitude"] = np.mean(joint_amplitudes)

    # Mechanical phase lags
    active_body_length = JOINTS_POS[-1] - JOINTS_POS[0]
    active_body_fraction = active_body_length / BODY_LENGTH

    iplss = compute_mechanical_phase_lags(
        times=times,
        signals=mech_signals.T,
        freqs=joint_frequencies,
        inds_couples=[[i, i+1] for i in range(mech_signals.shape[1]-1)]
    )

    metrics["mech_ipls"] = np.mean(iplss)
    metrics["mech_twl"] = np.sum(iplss) / active_body_fraction

    # Check for physics error
    if network.mujoco_error:
        metrics = dict.fromkeys(metrics, np.nan)

    return metrics


def compute_all_metrics(network):
    """ All neural and mechanical metrics """
    metrics = {
        **compute_neural_metrics(network),
        **compute_mechanical_metrics(network)
    }
    return metrics

