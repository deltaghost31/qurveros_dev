"""
Derivative Removal by Adiabatic Gate (DRAG) as a post-processing step.

The input is a control dictionary in the normalized units used by qurveros:
time runs from 0 to 1, and omega, phi, and delta are already multiplied by
the gate time. tg_alpha is T_g * alpha in those same units. Derivatives are
taken with respect to that normalized time.

Orders follow Motzoi et al., PRL 103, 110501 (2009) and Gambetta et al.,
PRA 83, 012308 (2011):

    order 0: the pulse unchanged
    order 1: ε' = ε - i β (dε/ds) / tg_alpha
    order 2: the order-1 term, plus
             ε' += β ((λ² - 4) / 8) (d²ε/ds²) / tg_alpha²
             delta' = delta - β ((λ² - 4) / 4) |ε|² / tg_alpha

The detuning sign is the Gambetta coefficient with one minus, because this
package writes the qubit detuning as (Δ/2) σ_z. In that convention
E_1 - E_0 = -Δ, so the compensating Stark shift enters with the opposite
sign from a Hamiltonian that assigns the detuning to |1⟩⟨1|.

beta scales the whole correction. beta = 0 reproduces order 0.
"""

import numpy as np
import scipy.interpolate

from qurveros.settings import settings


def gaussian_pi_control(n_points=400, sigma=0.15):

    """
    A resonant Gaussian pi pulse about x, in normalized units.

    The pulse area is pi, so a two-level simulation implements a Pauli X
    up to a global phase convention of the propagator.
    """

    times = np.linspace(0.0, 1.0, n_points)
    envelope = np.exp(-0.5 * ((times - 0.5) / sigma) ** 2)
    area = np.trapezoid(envelope, times)
    omega = envelope * (np.pi / area)

    return {
        'times': times,
        'omega': omega,
        'phi': np.zeros_like(times),
        'delta': np.zeros_like(times),
    }


def complex_envelope(control_dict):

    """
    Returns ε = omega * exp(i phi) and the normalized time grid.
    """

    omega = np.asarray(control_dict['omega'], dtype=float)
    phi = np.asarray(control_dict['phi'], dtype=float)
    times = np.asarray(control_dict['times'], dtype=float)
    return omega * np.exp(1j * phi), times


def _envelope_derivatives(envelope, times):

    """
    First and second derivatives of a complex envelope on a uniform grid.
    """

    real = scipy.interpolate.CubicSpline(times, np.real(envelope))
    imag = scipy.interpolate.CubicSpline(times, np.imag(envelope))
    first = real(times, 1) + 1j * imag(times, 1)
    second = real(times, 2) + 1j * imag(times, 2)
    return first, second


def apply_drag(control_dict, tg_alpha, *, order=1, beta=1.0, lam=None):

    """
    Returns a new control dictionary with a DRAG correction applied.

    Args:
        control_dict (dict): Pulses with keys times, omega, phi, and delta.
        tg_alpha (float): Dimensionless anharmonicity T_g α. Negative for
            a transmon.
        order (int): 0, 1, or 2.
        beta (float): Scale applied to every DRAG term. 0 leaves the pulse
            unchanged. 1 is the theoretical transmon coefficient.
        lam (float): |1⟩–|2⟩ coupling relative to |0⟩–|1⟩. The default is
            settings.options['DRAG_LAMBDA'], which is √2 for a transmon.

    Returns:
        A copy of control_dict. omega, phi, and delta are replaced when the
        correction is nonzero. drag_order, beta, tg_alpha, and lam are
        recorded on the copy.

    Raises:
        ValueError: The order is not 0, 1, or 2, or tg_alpha is 0 when a
            correction is requested.
    """

    if order not in (0, 1, 2):
        raise ValueError('DRAG order must be 0, 1, or 2.')

    if lam is None:
        lam = settings.options['DRAG_LAMBDA']

    corrected = dict(control_dict)
    corrected['drag_order'] = order
    corrected['beta'] = beta
    corrected['tg_alpha'] = tg_alpha
    corrected['lam'] = lam

    if order == 0 or beta == 0.0:
        corrected['omega'] = np.array(control_dict['omega'], dtype=float, copy=True)
        corrected['phi'] = np.array(control_dict['phi'], dtype=float, copy=True)
        corrected['delta'] = np.array(control_dict['delta'], dtype=float, copy=True)
        return corrected

    if tg_alpha == 0:
        raise ValueError('tg_alpha must be nonzero to apply DRAG.')

    envelope, times = complex_envelope(control_dict)
    first, second = _envelope_derivatives(envelope, times)

    updated = envelope - 1j * beta * first / tg_alpha
    delta = np.array(control_dict['delta'], dtype=float, copy=True)

    if order == 2:
        level_factor = (lam ** 2 - 4.0)
        updated = updated + beta * (level_factor / 8.0) * second / tg_alpha ** 2
        # Minus: (delta/2) σ_z defines detuning with the opposite sign from
        # the |1⟩⟨1| convention used to quote the Stark coefficient.
        delta = delta - beta * (level_factor / 4.0) * np.abs(envelope) ** 2 / tg_alpha

    corrected['omega'] = np.abs(updated)
    corrected['phi'] = np.unwrap(np.arctan2(np.imag(updated), np.real(updated)))
    corrected['delta'] = delta
    return corrected


def estimate_leakage(control_dict, tg_alpha, lam=None):

    """
    First-order leakage proxy: the spectral weight of the envelope at T_g α.

    The returned number is (λ/2)² |∫ ε(s) exp(i T_g α s) ds|². It is not the
    simulated leakage. It ranks DRAG strengths by how much drive remains at
    the |1⟩–|2⟩ transition, which is the quantity first-order DRAG removes.
    """

    if lam is None:
        lam = settings.options['DRAG_LAMBDA']

    envelope, times = complex_envelope(control_dict)
    duration = times[-1]
    normalized = times / duration
    # omega is already in per-gate units, so the integral is over s in [0, 1].
    amplitude = np.trapezoid(envelope * np.exp(1j * tg_alpha * normalized),
                             normalized)
    return float((lam ** 2 / 4.0) * np.abs(amplitude) ** 2)
