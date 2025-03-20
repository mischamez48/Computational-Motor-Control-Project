
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_JOINTS_AXIS = 15


def _plot_muscle_params(
    muscle_pars: dict,
):
    ''' Plot the muscle parameters '''
    joints = np.arange(N_JOINTS_AXIS)
    alphas = np.array([muscle_pars[joint][1]['alpha'] for joint in joints])
    betas = np.array([muscle_pars[joint][1]['beta'] for joint in joints])
    deltas = np.array([muscle_pars[joint][1]['delta'] for joint in joints])

    def _subplot(value_new, var_name, sub_plot_ind):
        plt.subplot(3, 1, sub_plot_ind)
        plt.plot(joints, value_new, 'o-', label=f'{var_name}')
        plt.ylabel(var_name)
        plt.yscale('log')
        plt.legend()

    plt.figure(figsize=(12, 8))
    _subplot(alphas, 'Alpha', 1)
    _subplot(betas,  'Beta', 2)
    _subplot(deltas, 'Delta', 3)

    plt.xlabel('Joint Index')
    plt.tight_layout()
    plt.show()


def load_muscle_parameters_from_file(
    parameters_file: str,
    folder_name: str = 'muscle_parameters',
):
    ''' Load the muscle parameters '''

    # Load the parameter
    muscle_params_df = pd.read_csv(f'{folder_name}/{parameters_file}')

    # Example
    # muscle_parameters_options = [
    #     [['joint_0'],  {'alpha': 6.9-08,  'beta': 8.3e-08, 'delta': 2.0-09 }],
    #     ...
    #     [['joint_14'], {'alpha': 7.1-09,  'beta': 8.4e-09, 'delta': 2.2-10 }]
    # ]

    muscle_parameters_options = [
        [
            [f'joint_{i}'],
            {
                'alpha': muscle_params_df.loc[i, 'alpha'],
                'beta': muscle_params_df.loc[i, 'beta'],
                'delta': muscle_params_df.loc[i, 'delta'],
                'gamma': 1.0,
                'epsilon': 0,
            }
        ]
        for i in range(N_JOINTS_AXIS)
    ]

    return muscle_parameters_options


def scale_muscle_parameters(
    muscle_parameters: list,
    muscle_factors: np.ndarray = np.ones(N_JOINTS_AXIS),
    head_joints: int = 1,
    tail_joints: int = 3,
):
    ''' Scale the muscle parameters '''

    # Target joints
    head_inds = np.arange(head_joints)
    tail_inds = np.arange(N_JOINTS_AXIS - tail_joints, N_JOINTS_AXIS)

    alphas = np.array([muscle_parameters[joint][1]['alpha']
                      for joint in range(N_JOINTS_AXIS)])
    betas = np.array([muscle_parameters[joint][1]['beta']
                     for joint in range(N_JOINTS_AXIS)])
    deltas = np.array([muscle_parameters[joint][1]['delta']
                      for joint in range(N_JOINTS_AXIS)])

    # Cap head joints
    first_non_head_joint = head_joints
    for head_i in head_inds:
        muscle_parameters[head_i][1]['alpha'] = alphas[first_non_head_joint]
        muscle_parameters[head_i][1]['beta'] = betas[first_non_head_joint]
        muscle_parameters[head_i][1]['delta'] = deltas[first_non_head_joint]

    # Cap tail joints
    last_non_tail_joint = N_JOINTS_AXIS - tail_joints - 1
    for tail_i in tail_inds:
        muscle_parameters[tail_i][1]['alpha'] = alphas[last_non_tail_joint]
        muscle_parameters[tail_i][1]['beta'] = betas[last_non_tail_joint]
        muscle_parameters[tail_i][1]['delta'] = deltas[last_non_tail_joint]

    # Joint stiffness scaling
    for joint_i, joint_factor in enumerate(muscle_factors):
        muscle_parameters[joint_i][1]['alpha'] *= joint_factor
        muscle_parameters[joint_i][1]['beta'] *= joint_factor
        muscle_parameters[joint_i][1]['delta'] *= joint_factor**0.5

    return muscle_parameters


def update_muscle_param(animat_options, pars):

    # Load muscle parameters
    muscle_tag = animat_options['muscle_parameters_tag']
    muscle_file = f'muscle_parameters_{muscle_tag}.csv'

    muscle_parameters = load_muscle_parameters_from_file(
        parameters_file=muscle_file,
        folder_name='muscle_parameters',
    )

    # Scale muscle parameters
    muscle_factors = np.ones(N_JOINTS_AXIS)
    head_joints = 1
    tail_joints = 3
    plot_muscles = False

    muscle_parameters = scale_muscle_parameters(
        muscle_parameters=muscle_parameters,
        muscle_factors=muscle_factors,
        head_joints=head_joints,
        tail_joints=tail_joints,
    )

    if plot_muscles:
        _plot_muscle_params(muscle_parameters)

    # Assign muscle parameters
    for joint_i, muscle_pars in enumerate(muscle_parameters):
        animat_options["control"]["muscles"][joint_i]["alpha"] = muscle_pars[1]['alpha']
        animat_options["control"]["muscles"][joint_i]["beta"] = muscle_pars[1]['beta']
        animat_options["control"]["muscles"][joint_i]["gamma"] = muscle_pars[1]['gamma']
        animat_options["control"]["muscles"][joint_i]["delta"] = muscle_pars[1]['delta']

        animat_options["control"]["muscles"][joint_i]["delta"] *= pars.damping_factor

    return


def update_drag_param(animat_options):
    for link in animat_options["morphology"]["links"]:
        link["drag_coefficients"][0][1] = 0.3*link["drag_coefficients"][0][1]
        link["drag_coefficients"][0][2] = -0.7

