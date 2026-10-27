"""
This module provides helper functions to simulate the control fields.
"""

import qutip
import numpy as np
import scipy.integrate
from qurveros.qubit_bench import quantumtools
from qurveros.settings import settings


def simulate_control_dict(control_dict, u_target):

    r"""
    Simulates the control_dict that contains the fields to
    control a single-qubit Hamiltonian of the form:

    H = \frac{\Omega}{2}(\cos\Phi \sigma_x + \sin\Phi \sigma_y)
        + \frac{\Delta}{2}\sigma_z

    Args:
        control_dict (dict): A dictionary with keys 'omega','phi','delta'.
        u_target (array): The SU(2) operator for the target quantum gate.

    Returns:
        A dictionary with entries:
            adj_final: The adjoint representation of the simulated control at
            the end of the evolution.
            adj_target: The adjoint representation of the target gate.
            avg_gate_fidelity: The average gate fidelity.
                See definition from quantumtools.calculate_gate_fidelity.
            u_final: The quantum gate implemented at the end of the evolution.
    """

    sim_dict = {}

    u_final = _single_qubit_sim(control_dict)[-1]
    avg_gate_fidelity = quantumtools.calculate_gate_fidelity(u_final, u_target)

    sim_dict['adj_final'] = quantumtools.calculate_adj_rep(u_final)
    sim_dict['adj_target'] = quantumtools.calculate_adj_rep(u_target)
    sim_dict['avg_gate_fidelity'] = avg_gate_fidelity
    sim_dict['u_final'] = u_final.full()

    return sim_dict


def _time_and_drive(control_dict):

    """
    Time grid and drive coefficients in the normalized units of the propagator.

    times[-1] is the gate duration in the dictionary. The usual control
    dictionary already uses a grid that ends at 1, so the scale factor is 1.
    """

    times = np.asarray(control_dict['times'], dtype=float)
    gate_time = times[-1]
    tlist = times / gate_time
    omega = np.asarray(control_dict['omega'], dtype=float)
    phi = np.asarray(control_dict['phi'], dtype=float)
    delta = np.asarray(control_dict['delta'], dtype=float)

    x_control = gate_time * omega * np.cos(phi)
    y_control = gate_time * omega * np.sin(phi)
    z_control = gate_time * delta
    return tlist, x_control, y_control, z_control


def _propagate(hamiltonian_terms, tlist):

    hamiltonian = qutip.QobjEvo(hamiltonian_terms, tlist=tlist)
    return qutip.propagator(hamiltonian, t=tlist, args=None,
                            options=settings.options['QUTIP_OPTIONS'])


def qutrit_drive_operators(lam):

    """
    Drive operators whose {|0⟩, |1⟩} block is σ/2.

    The |1⟩–|2⟩ coupling is lam times the |0⟩–|1⟩ coupling. lam = √2 for a
    transmon. The detuning operator matches (delta/2) σ_z on the qubit
    block and places |2⟩ at -3 delta/2, so a harmonic ladder stays harmonic
    before the anharmonicity is added.
    """

    hx = 0.5 * qutip.Qobj(np.array([
        [0, 1, 0],
        [1, 0, lam],
        [0, lam, 0],
    ], dtype=float))
    hy = 0.5 * qutip.Qobj(np.array([
        [0, -1j, 0],
        [1j, 0, -1j * lam],
        [0, 1j * lam, 0],
    ], dtype=complex))
    hz_delta = qutip.Qobj(np.diag([0.5, -0.5, -1.5]))
    hz_alpha = qutip.Qobj(np.diag([0.0, 0.0, 1.0]))
    return hx, hy, hz_delta, hz_alpha


def _single_qubit_sim(control_dict):

    """
    Simulates the quantum evolution of a single qubit.
    See controltools.py for the specification of the control.

    The simulation is done in the unitless time: t' = t/Tg where Tg
    is the total gate time.
    """

    tlist, x_control, y_control, z_control = _time_and_drive(control_dict)

    terms = [
        [qutip.sigmax() / 2, x_control],
        [qutip.sigmay() / 2, y_control],
        [qutip.sigmaz() / 2, z_control],
    ]
    return _propagate(terms, tlist)


def _qutrit_propagator(control_dict, tg_alpha, lam):

    """
    Propagator of the three-level Hamiltonian at every sample time.
    """

    tlist, x_control, y_control, z_control = _time_and_drive(control_dict)
    gate_time = np.asarray(control_dict['times'], dtype=float)[-1]
    alpha_control = np.full_like(tlist, gate_time * tg_alpha)
    hx, hy, hz_delta, hz_alpha = qutrit_drive_operators(lam)

    terms = [
        [hx, x_control],
        [hy, y_control],
        [hz_delta, z_control],
        [hz_alpha, alpha_control],
    ]
    return _propagate(terms, tlist)


def simulate_qutrit(control_dict, u_target, tg_alpha, lam=None):

    """
    Simulates a three-level transmon and reports leakage.

    The {|0⟩, |1⟩} drive matches the qubit simulator. tg_alpha is the
    dimensionless anharmonicity T_g α added to |2⟩.

    Returns:
        u_final: the 3x3 propagator at the end of the gate.
        leakage, leakage_from_0, leakage_from_1: final leakage.
        peak_leakage: the largest average leakage at any sample, including
            population that returns to the qubit subspace before the end.
        subspace_fidelity, subspace_infidelity: Pedersen fidelity of the
            2x2 block against u_target.
        subspace_phase: phase of Tr(U_target† U_block), in radians.
    """

    if lam is None:
        lam = settings.options['DRAG_LAMBDA']

    propagators = _qutrit_propagator(control_dict, tg_alpha, lam)
    unitaries = np.array([step.full() for step in propagators])
    population_2 = np.abs(unitaries[:, 2, :2]) ** 2
    leakage_trace = 0.5 * np.sum(population_2, axis=1)
    final = propagators[-1]
    from_0, from_1, leakage = quantumtools.leakage_components(final)
    block = quantumtools.qubit_subspace(final)
    fidelity = quantumtools.calculate_gate_fidelity(block, u_target)

    return {
        'u_final': final.full(),
        'leakage': leakage,
        'leakage_from_0': from_0,
        'leakage_from_1': from_1,
        'peak_leakage': float(np.max(leakage_trace)),
        'leakage_trace': leakage_trace,
        'population_2_from_0': population_2[:, 0],
        'population_2_from_1': population_2[:, 1],
        'times': np.asarray(control_dict['times'], dtype=float),
        'subspace_fidelity': fidelity,
        'subspace_infidelity': 1.0 - fidelity,
        'subspace_phase': quantumtools.subspace_phase(block, u_target),
    }


def qubit_propagators(control_dict):

    """
    The two-level propagator U(s) at every sample, as an (n, 2, 2) array.
    """

    return np.array([step.full() for step in _single_qubit_sim(control_dict)])


def _toggling_components(unitaries, operator):

    """
    Components n_l(s) = Tr(σ_l U(s)† A(s) U(s)) / 2 for a Hermitian A.

    operator has shape (n, 2, 2), or (2, 2) when it is constant.
    """

    paulis = np.array([qutip.sigmax().full(), qutip.sigmay().full(),
                       qutip.sigmaz().full()])
    operator = np.broadcast_to(operator, unitaries.shape)
    rotated = np.einsum('nji,njk,nkl->nil', unitaries.conj(), operator,
                        unitaries)
    return 0.5 * np.einsum('lij,nji->nl', paulis, rotated).real


def first_order_error_vectors(control_dict):

    r"""
    First-order sensitivity of a two-level pulse to static noise.

    In the toggling frame of the ideal evolution, a static dephasing
    $\frac{\delta_z}{2}\sigma_z$ and a multiplicative drive error
    $\varepsilon$ generate, to first order, the rotations
    $(T_g\delta_z)\,\vec r_z\cdot\vec\sigma/2$ and
    $\varepsilon\,\vec r_\varepsilon\cdot\vec\sigma/2$ with

        r_z = ∫ adj(U)† ẑ ds,
        r_ε = ∫ T_g Ω adj(U)† (cos Φ, sin Φ, 0) ds,

    over normalized time s in [0, 1]. T_g is times[-1], so the dephasing
    axis matches tg_delta_z of the static dephasing experiment. The average
    gate infidelity is then approximately (T_g δ_z)² |r_z|² / 6 and
    ε² |r_ε|² / 6. For a curve-derived pulse, |r_z| is the closure
    |closed_test| and |r_ε| is the tantrix area of the space curve.

    Returns:
        A dictionary with keys r_z and r_epsilon (3-vectors), their norms
        r_z_norm and r_epsilon_norm, and the running integrals r_z_trace
        and r_epsilon_trace with shape (n, 3).
    """

    tlist, x_control, y_control, _ = _time_and_drive(control_dict)
    unitaries = qubit_propagators(control_dict)

    sigma_z = qutip.sigmaz().full()
    drive = (x_control[:, None, None] * qutip.sigmax().full()
             + y_control[:, None, None] * qutip.sigmay().full())

    z_integrand = _toggling_components(unitaries, sigma_z)
    eps_integrand = _toggling_components(unitaries, drive)

    r_z_trace = scipy.integrate.cumulative_trapezoid(
        z_integrand, tlist, axis=0, initial=0)
    r_eps_trace = scipy.integrate.cumulative_trapezoid(
        eps_integrand, tlist, axis=0, initial=0)

    return {
        'r_z': r_z_trace[-1],
        'r_epsilon': r_eps_trace[-1],
        'r_z_norm': float(np.linalg.norm(r_z_trace[-1])),
        'r_epsilon_norm': float(np.linalg.norm(r_eps_trace[-1])),
        'r_z_trace': r_z_trace,
        'r_epsilon_trace': r_eps_trace,
    }
