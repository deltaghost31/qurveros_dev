"""
Shared parameters and helpers for the BARQ X-gate leakage notebooks.

Two-level simulations are scale free: every pulse lives on t/T_g in [0, 1].
The three-level simulator needs the product T_g α, so the notebooks fix a
transmon-like anharmonicity and a gate time here and derive tg_alpha from
them. Change ALPHA_MHZ, LAM, or GATE_TIME_NS to model another device.
"""

from pathlib import Path
import shutil

import numpy as np
import qutip

from qurveros.qubit_bench import quantumtools, simulator
from qurveros.spacecurve import BezierCurve

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'data'
CONTROL_POINTS_FILE = DATA_DIR / 'xgate_barq_points.csv'
STORED_POINTS_FILE = (HERE.parents[1] / 'results' / 'control_points'
                      / 'xgate_step50000.csv')

# α/2π in MHz. Negative for a transmon.
ALPHA_MHZ = -200.0
# |1⟩–|2⟩ coupling relative to |0⟩–|1⟩. √2 for a transmon.
LAM = np.sqrt(2)
GATE_TIME_NS = 50.0
GATE_TIMES_NS = (30.0, 50.0, 80.0, 120.0)

N_POINTS = 4096

U_TARGET = qutip.sigmax()
IDENTITY = qutip.qeye(2)


def tg_alpha(gate_time_ns=GATE_TIME_NS, alpha_mhz=ALPHA_MHZ):

    """T_g α in the dimensionless units of the simulator."""

    return 2 * np.pi * alpha_mhz * 1e6 * gate_time_ns * 1e-9


def peak_drive_mhz(control_dict, gate_time_ns=GATE_TIME_NS):

    """Ω_max / 2π in MHz when one pulse of the dictionary lasts gate_time_ns."""

    tg_omega_max = np.max(np.abs(control_dict['omega']))
    return tg_omega_max / (2 * np.pi * gate_time_ns * 1e-9) / 1e6


def store_control_points(barqcurve=None):

    """
    Writes the X-gate control points used by every notebook.

    With a BarqCurve, its optimized points are saved. Without one, the
    points stored with the package (50,000 Adam steps) are copied.
    """

    DATA_DIR.mkdir(exist_ok=True)
    if barqcurve is None:
        shutil.copy(STORED_POINTS_FILE, CONTROL_POINTS_FILE)
    else:
        barqcurve.save_bezier_control_points(CONTROL_POINTS_FILE)
    return CONTROL_POINTS_FILE


def load_xgate_curve():

    """The X-gate Bezier curve with its Frenet dictionary evaluated."""

    return BezierCurve.from_file(CONTROL_POINTS_FILE, is_barq=True)


def xgate_control(n_points=N_POINTS):

    """The X-gate TTC control dictionary sampled on n_points."""

    curve = load_xgate_curve()
    curve.evaluate_control_dict('TTC', n_points=n_points)
    return curve.get_control_dict()


def save_result(name, result):

    """Saves the array entries of a result dictionary to data/name.npz."""

    DATA_DIR.mkdir(exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in result.items()
              if isinstance(value, (np.ndarray, float, int, np.floating))}
    np.savez(DATA_DIR / f'{name}.npz', **arrays)


def load_result(name):

    """Loads a result saved with save_result as a plain dictionary."""

    with np.load(DATA_DIR / f'{name}.npz') as stored:
        return {key: stored[key] for key in stored.files}


def _with_noise(control_dict, kind, value):

    noisy = dict(control_dict)
    if kind == 'dephasing':
        noisy['delta'] = np.asarray(control_dict['delta']) + value
    elif kind == 'amplitude':
        noisy['omega'] = np.asarray(control_dict['omega']) * (1 + value)
    else:
        raise ValueError("kind must be 'dephasing' or 'amplitude'.")
    return noisy


def _final_unitary(control_dict, tg_alpha_value):

    if tg_alpha_value is None:
        return np.asarray(simulator.simulate_control_dict(
            control_dict, IDENTITY)['u_final'])
    return np.asarray(simulator.simulate_qutrit(
        control_dict, IDENTITY, tg_alpha_value, lam=LAM)['u_final'])


def sensitivity_sweep(control_dict, kind, values, *, u_target,
                      tg_alpha_value=None):

    """
    Static noise sweep that separates robustness from the error floor.

    kind is 'dephasing' (values are T_g δ_z, with T_g one pulse of the
    dictionary's time unit) or 'amplitude' (values are ε).

    Returns:
        infidelity: against u_target. For a qutrit this is the subspace
            infidelity, so it contains leakage and the noiseless error floor.
        sensitivity: 1 - F of the error operator U_noiseless† U_noisy,
            restricted to the qubit subspace. It is zero without noise and
            ignores the phase of |2⟩, so its log-log slope shows the order
            at which the noise enters the qubit: 2 for first order, 4 when
            the first order cancels.
        leakage: final leakage, only for a qutrit.
    """

    values = np.asarray(values, dtype=float)
    reference = _final_unitary(control_dict, tg_alpha_value)
    infidelity = np.zeros_like(values)
    sensitivity = np.zeros_like(values)
    leakage = np.zeros_like(values)

    for index, value in enumerate(values):
        noisy = _with_noise(control_dict, kind, value)
        if tg_alpha_value is None:
            result = simulator.simulate_control_dict(noisy, u_target)
            infidelity[index] = 1 - result['avg_gate_fidelity']
        else:
            result = simulator.simulate_qutrit(noisy, u_target,
                                               tg_alpha_value, lam=LAM)
            infidelity[index] = result['subspace_infidelity']
            leakage[index] = result['leakage']
        error = reference.conj().T @ result['u_final']
        sensitivity[index] = 1 - quantumtools.calculate_gate_fidelity(
            error[:2, :2], np.eye(2))

    sweep = {'kind': kind, 'values': values, 'infidelity': infidelity,
             'sensitivity': np.abs(sensitivity)}
    if tg_alpha_value is not None:
        sweep['leakage'] = leakage
    return sweep


def loglog_slope(x, y, floor=1e-15):

    """Least-squares slope of log10(y) against log10(x), ignoring y <= floor."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = y > floor
    if np.count_nonzero(keep) < 2:
        return np.nan
    return float(np.polyfit(np.log10(x[keep]), np.log10(y[keep]), 1)[0])


def error_budget(u_final, u_target):

    """
    Splits the subspace infidelity of a qutrit gate.

    The 2x2 block is written as V P with V unitary (polar decomposition).
    coherent_infidelity is the infidelity of V alone, the error that stays
    inside the qubit. The rest of the subspace infidelity is leakage, so
    subspace_infidelity ≈ leakage + coherent_infidelity.
    """

    block = quantumtools.qubit_subspace(u_final)
    left, _, right = np.linalg.svd(block)
    unitary_part = left @ right
    subspace = 1 - quantumtools.calculate_gate_fidelity(block, u_target)
    coherent = 1 - quantumtools.calculate_gate_fidelity(unitary_part,
                                                        u_target)
    return {
        'subspace_infidelity': float(subspace),
        'leakage': quantumtools.calculate_leakage(u_final),
        'coherent_infidelity': float(coherent),
    }


def per_pulse_axis(sim_dict, n_pulses):

    """
    A copy of a noise-grid result whose tg_delta_z refers to one pulse.

    The noise experiments report T δ_z for the whole duration T of the
    dictionary. A sequence of n_pulses unit pulses lasts n_pulses T_g.
    """

    relabeled = dict(sim_dict)
    relabeled['tg_delta_z'] = np.asarray(sim_dict['tg_delta_z']) / n_pulses
    return relabeled
