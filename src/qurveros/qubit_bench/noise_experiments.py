"""
This module automates some commonly used experiments to assess
the robustness properties of a pulse.
"""

import numpy as np
import logging

from qurveros.misctools import progbar_range
from qurveros.settings import settings
from qurveros.qubit_bench import simulator, noisetools
from qurveros import drag


def _control_dict_with(control_dict, **field_updates):

    """
    Returns a shallow copy of a control dictionary with some fields replaced.

    Experiments simulate the copy, so a failed run cannot leave the caller's
    pulses permanently modified.
    """

    updated = dict(control_dict)
    updated.update(field_updates)
    return updated


def static_dephasing_experiment(control_dict, u_target, max_points=None,
                                progbar_depth=0, *, tg_alpha=None, lam=None):

    r"""
    Implements a static additive dephasing error experiment.

    The noise Hamiltonian enters additively as
    $H_{\text{n}} = \frac{\delta_z}{2}\sigma_z$.

    The error range is found in qurveros.settings.

    Args:
        control_dict (dict): The control_dict of the quantum evolution.
        u_target (array): The SU(2) gate.
        max_points (int): The maximum number of delta_z points
        for the simulation.
        progbar_depth (int): Indentation level of the progress bar.
        The simultaneous-source experiment passes 1 so its nested bar
        lines up under the outer one.
        tg_alpha (float): When given, every point is simulated as a qutrit
        with this T_g α (see simulator.simulate_qutrit). The noise is added
        to delta, which shifts the qutrit levels as (1/2, -1/2, -3/2): the
        frequency noise of a ladder system up to a global phase.
        lam (float): The |1⟩–|2⟩ coupling ratio for the qutrit.

    Returns:
        A dictionary with keys:
        {
        experiment (str): The description of the experiment.
        tg_delta_z (array): An array for the values of the error in
        dimensionless units.
        infidelity (array): An array for the gate infidelity. For a qutrit
        this is the subspace infidelity, which includes leakage.
        leakage (array): Final leakage, only for a qutrit.
        }
    """

    EXPERIMENT_TITLE = 'static additive dephasing experiment'
    qutrit = tg_alpha is not None

    if max_points is None:
        max_points = settings.options['MAX_POINTS']

    Tg = control_dict['times'][-1]
    delta_z_vec = np.logspace(*settings.options['DELTA_Z_INTERVAL'],
                              max_points)/Tg

    fidelity_results = np.zeros(delta_z_vec.shape)
    leakage_results = np.zeros(delta_z_vec.shape)

    delta_no_error = np.copy(control_dict['delta'])
    const_vector = np.ones(delta_no_error.shape)

    if progbar_depth == 0:
        print(EXPERIMENT_TITLE.capitalize())

    for delta_z_idx in progbar_range(max_points,
                                     title='Static additive dephasing error',
                                     depth=progbar_depth):

        delta_z = delta_z_vec[delta_z_idx]
        noisy_dict = _control_dict_with(
            control_dict,
            delta=delta_no_error + delta_z*const_vector)

        if qutrit:
            sim_noise_dict = simulator.simulate_qutrit(noisy_dict, u_target,
                                                       tg_alpha, lam=lam)
            fidelity_results[delta_z_idx] = \
                sim_noise_dict['subspace_fidelity']
            leakage_results[delta_z_idx] = sim_noise_dict['leakage']
        else:
            sim_noise_dict = simulator.simulate_control_dict(noisy_dict,
                                                             u_target)
            fidelity_results[delta_z_idx] = \
                sim_noise_dict['avg_gate_fidelity']

    sim_dict = {}

    sim_dict['experiment'] = EXPERIMENT_TITLE

    sim_dict['infidelity'] = 1 - fidelity_results
    sim_dict['tg_delta_z'] = Tg*delta_z_vec

    if qutrit:
        sim_dict['leakage'] = leakage_results
        sim_dict['tg_alpha'] = tg_alpha

    return sim_dict


def simult_source_experiment(control_dict, u_target, max_points=None, *,
                             tg_alpha=None, lam=None):

    r"""
    Implements a simultaneous error source experiment with
    static additive dephasing error and multiplicative static driving error.

    The noise Hamiltonian is defined as:
    $H_{\text{n}} = \frac{\varepsilon\, \Omega}{2}[\cos\Phi\sigma_x +
    \sin\Phi\sigma_y] +\frac{\delta_z}{2}\sigma_z$

    The error range is found in qurveros.settings.

    Args:
        control_dict (dict): The control_dict of the quantum evolution.
        u_target (array): The SU(2) gate.
        max_points (int): The maximum number of delta_z points
        for the simulation.
        tg_alpha (float): When given, the grid is simulated as a qutrit.
        See static_dephasing_experiment.
        lam (float): The |1⟩–|2⟩ coupling ratio for the qutrit.

    Returns:
        A dictionary with keys:
        {
        experiment (str): The description of the experiment.
        epsilon (array): An array for the values
        of the driving field error.
        tg_delta_z (array): An array for the values
        of the dephasing error (dimensionless units).
        infidelity_matrix (array): An array for the gate infidelity.
        Each column provides the infidelity of a static dephasing
        experiment under a fixed driving error.
        leakage_matrix (array): Final leakage on the same grid,
        only for a qutrit.
        }
    """

    EXPERIMENT_TITLE = ('static additive dephasing and multiplicative '
                        'driving field experiment')
    qutrit = tg_alpha is not None

    if max_points is None:
        max_points = settings.options['MAX_POINTS']

    pulse_error_vec = np.logspace(*settings.options['EPSILON_INTERVAL'],
                                  max_points)

    infidelity_matrix = np.zeros([max_points, max_points])
    leakage_matrix = np.zeros([max_points, max_points])
    omega_no_error = np.copy(control_dict['omega'])

    print(EXPERIMENT_TITLE.capitalize())

    for pulse_idx in progbar_range(max_points,
                                   title='Multiplicative driving field error'):

        pulse_error = pulse_error_vec[pulse_idx]

        scaled_dict = _control_dict_with(
            control_dict,
            omega=(1 + pulse_error)*omega_no_error)

        dephasing_sim = static_dephasing_experiment(scaled_dict,
                                                    u_target,
                                                    max_points,
                                                    progbar_depth=1,
                                                    tg_alpha=tg_alpha,
                                                    lam=lam)

        infidelity_matrix[:, pulse_idx] = dephasing_sim['infidelity']
        if qutrit:
            leakage_matrix[:, pulse_idx] = dephasing_sim['leakage']

    sim_dict = {}

    sim_dict['experiment'] = EXPERIMENT_TITLE

    sim_dict['epsilon'] = pulse_error_vec
    sim_dict['tg_delta_z'] = dephasing_sim['tg_delta_z']
    sim_dict['infidelity_matrix'] = infidelity_matrix

    if qutrit:
        sim_dict['leakage_matrix'] = leakage_matrix
        sim_dict['tg_alpha'] = tg_alpha

    return sim_dict


def td_dephasing_experiment(control_dict, u_target, alpha, rng,
                            num_realizations=None,
                            max_points=None):

    r"""
    Implements a time-dependent additive dephasing error experiment.

    The noise PSD in the simulation is S = T_g/((omega/omega_B))^alpha
    with omega_B = 2\pi/Tg.

    The error range is found in qurveros.settings.

    Args:
        control_dict (dict): The control_dict of the quantum evolution.
        u_target (array): The SU(2) ideal gate.
        alpha (float): The power law noise exponent 1/f^alpha.
        rng (np.random): The random number generator.
        num_realizations (int): The number of realizations to average over.
        max_points (int): The maximum number of delta_z points
        for the simulation.

    Returns:
        A dictionary with keys:
        {
        experiment (str): The description of the experiment.
        tg_delta_z (array): An array for the values of the error in
        dimensionless units.
        infidelity_matrix (array): An array for the gate infidelity.
        Every column contains the infidelity of the noise realizations at
        a fixed noise strength.
        alpha (float): The value of alpha used for the simulation.
        }
    """

    EXPERIMENT_NAME = f'Time-dependent additive dephasing (alpha = {alpha})'

    logger = logging.getLogger('TDsim')

    if max_points is None:
        max_points = settings.options['MAX_POINTS']

    if num_realizations is None:
        num_realizations = settings.options['NUM_REALIZATIONS']

    Tg = control_dict['times'][-1]
    delta_z_vec = np.logspace(*settings.options['DELTA_Z_INTERVAL'],
                              max_points)/Tg

    infidelity_matrix = np.zeros([num_realizations, max_points])

    delta_no_error = np.copy(control_dict['delta'])
    n_points = len(control_dict['omega'])

    for delta_z_idx in progbar_range(max_points, title=EXPERIMENT_NAME):

        delta_z = delta_z_vec[delta_z_idx]

        logger.info(f'Generating noise for Tg_delta_z={Tg*delta_z}.')

        if alpha == 0:
            noise_in = noisetools.get_white_noise_array(
                                    num_realizations=num_realizations,
                                    n_points=n_points,
                                    rng=rng)

        else:
            noise_in = noisetools.get_colored_noise_array(
                                    num_realizations=num_realizations,
                                    n_points=n_points,
                                    alpha=alpha,
                                    rng=rng)

        logger.info(
         f'Total number of noise realizations generated: {num_realizations}')

        # Removes the N-dependence from the white noise
        noise_in = noise_in*np.sqrt(n_points)

        # Create omega_B for the simulated spectra.
        omega_b_scale = (2*np.pi/(n_points))**(alpha)
        noise_in = noise_in*np.sqrt(omega_b_scale)

        logger.info('Noise generation done.')

        for noise_run in progbar_range(num_realizations,
                                       title='Noise realizations',
                                       depth=1):

            noise_vector = delta_z * noise_in[noise_run, :]
            noisy_dict = _control_dict_with(
                control_dict,
                delta=delta_no_error + noise_vector)

            try:

                sim_noise_dict = simulator.simulate_control_dict(
                    noisy_dict,
                    u_target)

                fidelity = sim_noise_dict['avg_gate_fidelity']

            except Exception:

                logger.info(
                    f'Error in noise realization: {noise_run}\n'
                    f' and delta_z: {delta_z}\n'
                    f' Check index [{noise_run}, {delta_z_idx}] for nan.')

                logger.exception('')
                fidelity = np.nan

            infidelity_matrix[noise_run, delta_z_idx] = 1-fidelity

    sim_dict = {}

    sim_dict['experiment'] = EXPERIMENT_NAME
    sim_dict['tg_delta_z'] = Tg*delta_z_vec

    sim_dict['infidelity_matrix'] = infidelity_matrix
    sim_dict['alpha'] = alpha

    return sim_dict


def _drag_betas(betas):

    if betas is None:
        n_betas = settings.options['DRAG_BETA_POINTS']
        return np.linspace(0.0, 2.0, n_betas)
    return np.asarray(betas, dtype=float)


def drag_order_experiment(control_dict, u_target, tg_alpha, *,
                          orders=(0, 1, 2), betas=None, lam=None):

    """
    Compares DRAG orders on one pulse.

    For every order and every beta, the pulse is corrected and simulated as
    a qutrit. beta scales the theoretical DRAG terms. Order 0 ignores beta
    and is repeated on each column so the arrays stay rectangular.

    Returns:
        orders, betas: the axes of the scan.
        leakage, peak_leakage, subspace_infidelity, subspace_phase:
            arrays with shape (n_orders, n_betas).
        leakage_estimate: the same shape, from the Fourier proxy in drag.py.
        tg_alpha, lam: the values used.
    """

    if lam is None:
        lam = settings.options['DRAG_LAMBDA']

    orders = np.asarray(tuple(orders), dtype=int)
    betas = _drag_betas(betas)
    shape = (len(orders), len(betas))

    leakage = np.zeros(shape)
    peak_leakage = np.zeros(shape)
    subspace_infidelity = np.zeros(shape)
    subspace_phase = np.zeros(shape)
    leakage_estimate = np.zeros(shape)

    print('DRAG order comparison')

    for order_idx, order in enumerate(orders):
        for beta_idx in progbar_range(len(betas), title=f'DRAG order {order}'):

            beta_value = float(betas[beta_idx])
            corrected = drag.apply_drag(control_dict, tg_alpha,
                                        order=int(order), beta=beta_value,
                                        lam=lam)
            simulated = simulator.simulate_qutrit(corrected, u_target,
                                                  tg_alpha, lam=lam)
            leakage[order_idx, beta_idx] = simulated['leakage']
            peak_leakage[order_idx, beta_idx] = simulated['peak_leakage']
            subspace_infidelity[order_idx, beta_idx] = \
                simulated['subspace_infidelity']
            subspace_phase[order_idx, beta_idx] = simulated['subspace_phase']
            leakage_estimate[order_idx, beta_idx] = drag.estimate_leakage(
                corrected, tg_alpha, lam=lam)

    return {
        'experiment': 'DRAG order comparison',
        'orders': orders,
        'betas': betas,
        'leakage': leakage,
        'peak_leakage': peak_leakage,
        'subspace_infidelity': subspace_infidelity,
        'subspace_phase': subspace_phase,
        'leakage_estimate': leakage_estimate,
        'tg_alpha': tg_alpha,
        'lam': lam,
    }


def _qutrit_dephasing_sweep(control_dict, u_target, tg_alpha, lam, max_points):

    """
    Static dephasing of one pulse on the three-level simulator.

    Infidelity is the subspace infidelity of the 2x2 block, so leakage and
    the Stark-shifted qubit rotation both contribute.
    """

    sweep = static_dephasing_experiment(control_dict, u_target, max_points,
                                        tg_alpha=tg_alpha, lam=lam)
    return sweep['tg_delta_z'], sweep['infidelity'], sweep['leakage']


def drag_dephasing_comparison(control_dict, u_target, tg_alpha, *,
                              order=1, beta=1.0, lam=None, max_points=None):

    """
    Static dephasing of a pulse before and after DRAG, on the qutrit.

    Both sweeps use simulate_qutrit. The subspace infidelity therefore
    includes leakage and the detuning correction, which a two-level sweep
    of the same pulses cannot see. A large |tg_alpha| makes the DRAG
    correction small, so the two infidelity curves approach each other.
    Very large |tg_alpha| also makes the qutrit diagonal stiff; keep the
    value inside the range the ODE solver can integrate.
    """

    if lam is None:
        lam = settings.options['DRAG_LAMBDA']

    corrected = drag.apply_drag(control_dict, tg_alpha, order=order,
                                beta=beta, lam=lam)
    tg_delta_z, bare_infidelity, bare_leakage = _qutrit_dephasing_sweep(
        control_dict, u_target, tg_alpha, lam, max_points)
    _, drag_infidelity, drag_leakage = _qutrit_dephasing_sweep(
        corrected, u_target, tg_alpha, lam, max_points)
    return {
        'experiment': 'qutrit static dephasing before and after DRAG',
        'tg_alpha': tg_alpha,
        'order': order,
        'beta': beta,
        'lam': lam,
        'bare_infidelity': bare_infidelity,
        'drag_infidelity': drag_infidelity,
        'bare_leakage': bare_leakage,
        'drag_leakage': drag_leakage,
        'tg_delta_z': tg_delta_z,
    }
