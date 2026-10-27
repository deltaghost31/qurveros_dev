"""
Derivative Removal by Adiabatic Gate (DRAG) as a post-processing step.

The input is a control dictionary in the normalized units used by qurveros.
``omega``, ``phi``, and ``delta`` are already multiplied by the gate time.
``tg_alpha`` is ``T_g α`` in those units. Derivatives are taken with respect
to normalized time ``s = (t - t0) / (t1 - t0)``, so a grid that does not end
at 1 is still differentiated correctly. For dictionaries produced by
``evaluate_control_dict``, ``s`` is the stored ``times`` array.

The formulas are Motzoi et al., Phys. Rev. Lett. 103, 110501 (2009),
translated into the package convention ``H ⊃ (Δ/2) σ_z``. In that convention
``E_1 - E_0 = -Δ``. Motzoi adds the Stark shift to ``|1⟩⟨1|``, so every
detuning correction below is the negative of the coefficient quoted for that
convention. ``β`` multiplies only the derivative quadrature. The Stark shift
and the higher-order radial terms keep their analytic coefficients.

    order 0: the pulse unchanged
    order 1: ε' = ε - i β (dε/ds) / ν
             Δ' = Δ - ((λ² - 4) / 4) |ε|² / ν
    order 2: the order-1 terms, plus Motzoi's next corrections
             ε' = ε_rad - i β (dε/ds) / ν
                  + i β [33(λ² - 2) / 24] |ε|² (dε/ds) / ν³
             ε_rad = ε { 1 + (λ² - 4) |ε|² / (8 ν²)
                         - (13λ⁴ - 76λ² + 112) |ε|⁴ / (128 ν⁴) }
             Δ' includes + (λ⁴ - 7λ² + 12) |ε|⁴ / (16 ν³)

``ν`` is ``tg_alpha``. For a real positive envelope the radial factor
reproduces Motzoi's corrections to ``E^x``, and the imaginary terms
reproduce ``E^y = -Ė/α + 33(λ²-2) E² Ė / (24 α³)``. ``beta = 0`` leaves the
pulse unchanged.
"""

import numpy as np
import scipy.integrate
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


def _envelope_derivative(envelope, times):

    """
    Derivative of a complex envelope with respect to normalized time.

    Normalized time runs from 0 to 1 between the first and last samples,
    whatever numbers are stored in ``times``.
    """

    times = np.asarray(times, dtype=float)
    span = float(times[-1] - times[0])
    if span == 0.0:
        raise ValueError('DRAG needs a nonzero time interval.')
    normalized = (times - times[0]) / span
    real = scipy.interpolate.CubicSpline(normalized, np.real(envelope))
    imag = scipy.interpolate.CubicSpline(normalized, np.imag(envelope))
    return real(normalized, 1) + 1j * imag(normalized, 1)


def apply_drag(control_dict, tg_alpha, *, order=1, beta=1.0, lam=None):

    """
    Returns a new control dictionary with a DRAG correction applied.

    Args:
        control_dict (dict): Pulses with keys times, omega, phi, and delta.
        tg_alpha (float): Dimensionless anharmonicity T_g α. Negative for
            a transmon.
        order (int): 0, 1, or 2. Order 1 is the derivative quadrature plus
            the AC Stark detuning. Order 2 adds Motzoi's higher-order
            radial, derivative, and detuning terms.
        beta (float): Scale of the derivative quadrature only. 0 leaves the
            pulse unchanged. The Stark shift is not multiplied by beta.
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
    derivative = _envelope_derivative(envelope, times)
    amplitude_sq = np.abs(envelope) ** 2
    # Minus relative to a |1⟩⟨1| detuning: (delta/2) σ_z has E_1 - E_0 = -delta.
    stark = -((lam ** 2 - 4.0) / 4.0) * amplitude_sq / tg_alpha
    delta = np.array(control_dict['delta'], dtype=float, copy=True) + stark

    quadrature = -1j * beta * derivative / tg_alpha
    if order == 1:
        updated = envelope + quadrature
    else:
        lam2 = lam ** 2
        lam4 = lam2 ** 2
        radial = (1.0
                  + (lam2 - 4.0) * amplitude_sq / (8.0 * tg_alpha ** 2)
                  - (13.0 * lam4 - 76.0 * lam2 + 112.0) * amplitude_sq ** 2
                  / (128.0 * tg_alpha ** 4))
        quadrature = quadrature + (
            1j * beta * (33.0 * (lam2 - 2.0) / 24.0)
            * amplitude_sq * derivative / tg_alpha ** 3)
        delta = delta + (
            (lam4 - 7.0 * lam2 + 12.0) * amplitude_sq ** 2
            / (16.0 * tg_alpha ** 3))
        updated = envelope * radial + quadrature

    corrected['omega'] = np.abs(updated)
    corrected['phi'] = np.unwrap(np.arctan2(np.imag(updated), np.real(updated)))
    corrected['delta'] = delta
    return corrected


def estimate_leakage(control_dict, tg_alpha, lam=None):

    """
    Bare spectral weight of the envelope at the leakage transition.

    The returned number is (λ/2)² |∫ ε(s) exp(i ν s) ds|², with ν = T_g α.
    This is the Fourier component removed by the derivative term when the
    envelope vanishes at both endpoints. It is not the gate leakage: that
    amplitude also contains the ideal qubit factor ⟨1|U_q(s)|j⟩, and it is
    not the leakage computed by simulate_qutrit. Use it only to rank pulses
    by residual spectral weight.
    """

    if lam is None:
        lam = settings.options['DRAG_LAMBDA']

    envelope, times = complex_envelope(control_dict)
    span = float(times[-1] - times[0])
    normalized = (times - times[0]) / span
    # omega is already in per-gate units, so the integral is over s in [0, 1].
    amplitude = np.trapezoid(envelope * np.exp(1j * tg_alpha * normalized),
                             normalized)
    return float((lam ** 2 / 4.0) * np.abs(amplitude) ** 2)


def first_order_leakage(control_dict, tg_alpha, lam=None):

    r"""
    Leakage to |2⟩ to first order in the |1⟩–|2⟩ coupling.

    The qubit block evolves exactly under U_q(s) from the two-level
    simulator, and |2⟩ accumulates the phase
    θ_2(s) = ∫ T_g (α - 3Δ/2) ds of the qutrit Hamiltonian. The amplitude
    to leak from |j⟩ is

        A_j(s) = -i (λ/2) ∫_0^s T_g ε(s') e^{i θ_2(s')} ⟨1|U_q(s')|j⟩ ds',

    with T_g = times[-1], so tg_alpha keeps the meaning it has in
    simulate_qutrit. The Stark shift that the same coupling puts on |1⟩ is
    second order and is left out. The transient (peak) leakage is therefore
    reliable, but when the final leakage is a small remainder of a much
    larger transient it can differ from simulate_qutrit by a factor of a few.

    Returns:
        A dictionary with leakage, leakage_from_0, leakage_from_1 at the end
        of the pulse, leakage_trace (the average over |0⟩ and |1⟩ at every
        sample), and times.
    """

    from qurveros.qubit_bench import simulator

    if lam is None:
        lam = settings.options['DRAG_LAMBDA']

    times = np.asarray(control_dict['times'], dtype=float)
    gate_time = times[-1]
    tlist = times / gate_time
    envelope, _ = complex_envelope(control_dict)
    delta = np.asarray(control_dict['delta'], dtype=float)

    level_2_energy = gate_time * (tg_alpha - 1.5 * delta)
    theta_2 = scipy.integrate.cumulative_trapezoid(
        level_2_energy, tlist, initial=0)

    unitaries = simulator.qubit_propagators(control_dict)
    coupling = (gate_time * (lam / 2.0) * envelope
                * np.exp(1j * theta_2))[:, None] * unitaries[:, 1, :]
    amplitudes = -1j * scipy.integrate.cumulative_trapezoid(
        coupling, tlist, axis=0, initial=0)

    population = np.abs(amplitudes) ** 2
    return {
        'leakage': float(0.5 * np.sum(population[-1])),
        'leakage_from_0': float(population[-1, 0]),
        'leakage_from_1': float(population[-1, 1]),
        'leakage_trace': 0.5 * np.sum(population, axis=1),
        'times': times,
    }
