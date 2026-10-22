"""
This module provides some helper functions involving quantum operations.
"""

import qutip
import numpy as np


def calculate_gate_fidelity(u_sim, u_target):

    """
    Calculates the gate fidelity between a simulated gate and a target gate.

    Args:
        u_sim (array): The simulated SU(2) gate.
        u_target (array): The target SU(2) gate.

    Returns:
        fidelity (float): The average gate fidelity between the gates.

    Note:

    There exist multiple definitions of the gate fidelity. Care must be
    taken on the factors involved in each definition.

    For convenience, we define M = u_target^{dagger} u_sim

    In our work, we used the average gate fidelity as defined in:

    Fidelity of quantum operations by Pedersen et al.
    https://doi.org/10.1016/j.physleta.2007.02.069
    """

    u_target = qutip.Qobj(u_target)
    u_sim = qutip.Qobj(u_sim)

    M = (u_target.dag())*(u_sim)

    unitary_m_check = (M * (M.dag())).tr()
    unitary_m_tr_sq = (M.tr())*(M.dag().tr())

    d = M.shape[0]
    normalization = d*(d+1)

    avg_fidelity = (unitary_m_check + unitary_m_tr_sq)/normalization

    return avg_fidelity.real


def qubit_subspace(gate):

    """
    Returns the top-left 2x2 block of a qutrit propagator.

    The block is the qubit-subspace evolution. It is not unitary once
    amplitude has leaked to |2⟩.
    """

    unitary = np.asarray(qutip.Qobj(gate).full())
    return unitary[:2, :2]


def calculate_leakage(gate):

    """
    Average probability of ending in a state outside the qubit subspace.

    Inputs |0⟩ and |1⟩ are weighted equally. For a qutrit this is
    (|U_20|² + |U_21|²) / 2.
    """

    unitary = np.asarray(qutip.Qobj(gate).full())
    leaked_from_0 = np.sum(np.abs(unitary[2:, 0]) ** 2)
    leaked_from_1 = np.sum(np.abs(unitary[2:, 1]) ** 2)
    return float(0.5 * (leaked_from_0 + leaked_from_1))


def leakage_components(gate):

    """
    Leakage from |0⟩ and from |1⟩ separately, then their average.
    """

    unitary = np.asarray(qutip.Qobj(gate).full())
    leaked_from_0 = float(np.sum(np.abs(unitary[2:, 0]) ** 2))
    leaked_from_1 = float(np.sum(np.abs(unitary[2:, 1]) ** 2))
    average = 0.5 * (leaked_from_0 + leaked_from_1)
    return leaked_from_0, leaked_from_1, average


def subspace_phase(u_block, u_target):

    """
    Phase of Tr(U_target† U_block), in radians.

    A DRAG Stark shift shows up here even when the leakage itself is small.
    The global phase of a perfect Pauli gate is included, so compare two
    pulses against the same target rather than expecting zero.
    """

    block = qutip.Qobj(u_block)
    target = qutip.Qobj(u_target)
    overlap = (target.dag() * block).tr()
    return float(np.angle(overlap))


def calculate_adj_rep(gate):

    """
    Calculates the adjoint representation of a given SU(2) gate, from the
    definition of the transfer matrix.
    """

    A = 1j*np.zeros((4, 3))
    for idx, pauli in enumerate([qutip.sigmax(),
                                 qutip.sigmay(),
                                 qutip.sigmaz()]):

        A[:, idx] = qutip.operator_to_vector(pauli).full().flatten()

    gate = qutip.Qobj(gate)

    A_dag = np.conjugate(A.T)

    adj_rep = 0.5*A_dag @ (qutip.to_super(gate).full()) @ A

    return adj_rep.real
