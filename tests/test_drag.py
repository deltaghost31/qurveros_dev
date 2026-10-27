"""
Tests for the qutrit simulator and DRAG post-processing.
"""

import unittest

import matplotlib
matplotlib.use('Agg')

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import qutip

from qurveros import controltools, drag, plottools
from qurveros.qubit_bench import noise_experiments, quantumtools, simulator
from qurveros.spacecurve import SpaceCurve


TG_ALPHA = -20.0
N_POINTS = 300


def bessel_curve(n_curve=2048):

    """
    The robust Bessel curve used by the curve tests.
    """

    def curve(x, params):
        theta = params[0] * jnp.cos(x)
        phi = params[1] * theta
        return jnp.array([
            jnp.cos(phi) * jnp.sin(theta),
            jnp.sin(phi) * jnp.sin(theta),
            jnp.cos(theta),
        ])

    spacecurve = SpaceCurve(curve=curve, order=1, interval=[0, 2 * jnp.pi],
                            params=[5.5201, 0.5660])
    spacecurve.evaluate_frenet_dict(n_points=n_curve)
    return spacecurve


def bessel_control(n_curve=2048, n_sim=400):

    """
    The robust Bessel pulse used by the curve tests, on a shorter grid.
    """

    spacecurve = bessel_curve(n_curve)
    spacecurve.evaluate_control_dict('XY', n_points=n_sim)
    return spacecurve.get_control_dict()


class DragPulseTestCase(unittest.TestCase):

    def setUp(self):
        self.pulse = drag.gaussian_pi_control(n_points=N_POINTS, sigma=0.12)
        self.target = qutip.sigmax()

    def test_order_zero_is_identity(self):
        corrected = drag.apply_drag(self.pulse, TG_ALPHA, order=0, beta=1.0)
        for field in ('omega', 'phi', 'delta'):
            self.assertTrue(np.allclose(corrected[field], self.pulse[field]))
        # The caller's dictionary is not modified.
        self.pulse['omega'][0] = 3.0
        self.assertNotEqual(corrected['omega'][0], 3.0)

    def test_large_anharmonicity_removes_the_correction(self):
        huge = -1.0e6
        for order in (1, 2):
            corrected = drag.apply_drag(self.pulse, huge, order=order, beta=1.0)
            self.assertTrue(np.allclose(corrected['omega'], self.pulse['omega'],
                                        atol=1e-4))
            self.assertTrue(np.allclose(corrected['delta'], self.pulse['delta'],
                                        atol=1e-4))

    def test_unknown_order_raises(self):
        with self.assertRaises(ValueError):
            drag.apply_drag(self.pulse, TG_ALPHA, order=3)

    def test_order_one_detuning(self):
        corrected = drag.apply_drag(self.pulse, TG_ALPHA, order=1, beta=0.4,
                                    lam=np.sqrt(2))
        # (delta/2) σ_z has the opposite sign from a |1⟩⟨1| Stark shift.
        # beta scales the quadrature only, so the detuning is independent of it.
        expected = -((2.0 - 4.0) / 4.0) * self.pulse['omega'] ** 2 / TG_ALPHA
        self.assertTrue(np.allclose(corrected['delta'], expected))
        half = drag.apply_drag(self.pulse, TG_ALPHA, order=1, beta=0.4,
                               lam=np.sqrt(2))
        full = drag.apply_drag(self.pulse, TG_ALPHA, order=1, beta=1.0,
                               lam=np.sqrt(2))
        self.assertTrue(np.allclose(half['delta'], full['delta']))
        self.assertFalse(np.allclose(half['phi'], full['phi']))

    def test_order_two_matches_motzoi_real_pulse(self):
        lam = np.sqrt(2)
        nu = TG_ALPHA
        base = self.pulse['omega']
        first = drag._envelope_derivative(
            base.astype(complex), self.pulse['times'])
        lam2 = lam ** 2
        lam4 = lam2 ** 2
        amplitude_sq = base ** 2
        radial = (1.0
                  + (lam2 - 4.0) * amplitude_sq / (8.0 * nu ** 2)
                  - (13.0 * lam4 - 76.0 * lam2 + 112.0) * amplitude_sq ** 2
                  / (128.0 * nu ** 4))
        expected = base * radial - 1j * first / nu
        corrected = drag.apply_drag(self.pulse, nu, order=2, beta=1.0, lam=lam)
        got = corrected['omega'] * np.exp(1j * corrected['phi'])
        self.assertTrue(np.allclose(got, expected, atol=1e-10))
        quartic = (lam4 - 7.0 * lam2 + 12.0) * amplitude_sq ** 2 / (16.0 * nu ** 3)
        stark = -((lam2 - 4.0) / 4.0) * amplitude_sq / nu
        self.assertTrue(np.allclose(corrected['delta'], stark + quartic))

    def test_qutrit_operators_match_pauli_block(self):
        hx, hy, hz_delta, _ = simulator.qutrit_drive_operators(np.sqrt(2))
        self.assertTrue(np.allclose(hx.full()[:2, :2],
                                    (qutip.sigmax() / 2).full()))
        self.assertTrue(np.allclose(hy.full()[:2, :2],
                                    (qutip.sigmay() / 2).full()))
        self.assertTrue(np.allclose(hz_delta.full()[:2, :2],
                                    (qutip.sigmaz() / 2).full()))

    def test_large_anharmonicity_matches_qubit_simulator(self):
        tg_alpha = -1000.0
        qutrit = simulator.simulate_qutrit(self.pulse, self.target, tg_alpha)
        qubit = simulator.simulate_control_dict(self.pulse, self.target)
        self.assertTrue(np.isclose(
            qutrit['subspace_fidelity'],
            qubit['avg_gate_fidelity'],
            atol=1e-4))

    def test_first_order_reduces_leakage(self):
        bare = simulator.simulate_qutrit(
            drag.apply_drag(self.pulse, TG_ALPHA, order=0),
            self.target, TG_ALPHA)
        first = simulator.simulate_qutrit(
            drag.apply_drag(self.pulse, TG_ALPHA, order=1, beta=1.0),
            self.target, TG_ALPHA)
        self.assertLess(first['leakage'], bare['leakage'])
        self.assertLess(first['peak_leakage'], bare['peak_leakage'])

    def test_first_order_stark_reduces_subspace_error(self):
        derivative_only = drag.apply_drag(self.pulse, TG_ALPHA, order=1,
                                          beta=1.0)
        derivative_only['delta'] = np.zeros_like(self.pulse['delta'])
        without_stark = simulator.simulate_qutrit(
            derivative_only, self.target, TG_ALPHA)
        with_stark = simulator.simulate_qutrit(
            drag.apply_drag(self.pulse, TG_ALPHA, order=1, beta=1.0),
            self.target, TG_ALPHA)
        self.assertLess(with_stark['subspace_infidelity'],
                        without_stark['subspace_infidelity'])

    def test_leakage_estimate_ranks_beta(self):
        betas = np.linspace(0.0, 1.5, 6)
        estimates = []
        leakages = []
        for beta in betas:
            corrected = drag.apply_drag(self.pulse, TG_ALPHA, order=1,
                                        beta=float(beta))
            estimates.append(drag.estimate_leakage(corrected, TG_ALPHA))
            leakages.append(simulator.simulate_qutrit(
                corrected, self.target, TG_ALPHA)['leakage'])
        # The proxy ranks spectral weight, not simulate_qutrit leakage.
        # For this Gaussian the weight is smallest at the analytic beta.
        self.assertEqual(int(np.argmin(estimates)), int(np.argmin(np.abs(betas - 1.0))))
        self.assertLess(estimates[int(np.argmin(estimates))], estimates[0])
        self.assertLess(leakages[-1], leakages[0])


class DragExperimentTestCase(unittest.TestCase):

    def test_scan_shape(self):
        pulse = drag.gaussian_pi_control(n_points=80, sigma=0.15)
        betas = np.linspace(0.0, 1.0, 3)
        result = noise_experiments.drag_order_experiment(
            pulse, qutip.sigmax(), tg_alpha=-30.0,
            orders=(0, 1, 2), betas=betas)
        self.assertEqual(result['leakage'].shape, (3, 3))
        self.assertEqual(result['peak_leakage'].shape, (3, 3))
        self.assertEqual(result['subspace_infidelity'].shape, (3, 3))
        self.assertEqual(result['leakage_estimate'].shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(result['leakage'])))

        ax = plottools.plot_drag_scan(result)
        self.assertEqual(len(ax.lines), 3)
        plt.close('all')

    def test_qutrit_dephasing_comparison(self):
        pulse = drag.gaussian_pi_control(n_points=80, sigma=0.15)
        compared = noise_experiments.drag_dephasing_comparison(
            pulse, qutip.sigmax(), tg_alpha=-80.0, order=1, beta=1.0,
            max_points=3)
        self.assertEqual(compared['bare_infidelity'].shape, (3,))
        self.assertEqual(compared['drag_leakage'].shape, (3,))
        self.assertTrue(np.all(np.isfinite(compared['bare_infidelity'])))
        self.assertTrue(np.all(np.isfinite(compared['drag_infidelity'])))
        self.assertTrue(np.all(np.isfinite(compared['bare_leakage'])))
        # At small added detuning the Stark correction lowers subspace error.
        self.assertLess(compared['drag_infidelity'][0],
                        compared['bare_infidelity'][0])

    def test_robust_pulse_stays_flat_after_modest_drag(self):
        control = bessel_control()
        implemented = simulator.simulate_control_dict(
            control, qutip.qeye(2))['u_final']
        corrected = drag.apply_drag(control, tg_alpha=-2000.0, order=1,
                                    beta=1.0)
        fidelity = simulator.simulate_control_dict(
            corrected, implemented)['avg_gate_fidelity']
        self.assertGreater(fidelity, 0.998)

        corrected_gate = simulator.simulate_control_dict(
            corrected, qutip.qeye(2))['u_final']
        sweep = noise_experiments.static_dephasing_experiment(
            corrected, corrected_gate, max_points=4)
        # A plain pi pulse at the same detuning is orders of magnitude worse.
        self.assertLess(sweep['infidelity'][-1], 1e-5)


class FirstOrderTestCase(unittest.TestCase):

    def setUp(self):
        self.pulse = drag.gaussian_pi_control(n_points=N_POINTS, sigma=0.12)
        self.target = qutip.sigmax()

    def test_error_vectors_predict_small_noise_infidelity(self):
        vectors = simulator.first_order_error_vectors(self.pulse)
        delta_z = 1e-3
        dephased = simulator.simulate_control_dict(
            dict(self.pulse, delta=self.pulse['delta'] + delta_z),
            self.target)
        self.assertTrue(np.isclose(1 - dephased['avg_gate_fidelity'],
                                   delta_z ** 2 * vectors['r_z_norm'] ** 2 / 6,
                                   rtol=1e-2))
        epsilon = 1e-3
        scaled = simulator.simulate_control_dict(
            dict(self.pulse, omega=self.pulse['omega'] * (1 + epsilon)),
            self.target)
        self.assertTrue(np.isclose(
            1 - scaled['avg_gate_fidelity'],
            epsilon ** 2 * vectors['r_epsilon_norm'] ** 2 / 6,
            rtol=1e-2))
        self.assertEqual(vectors['r_z_trace'].shape, (N_POINTS, 3))

    def test_error_vectors_match_curve_geometry(self):
        spacecurve = bessel_curve()
        spacecurve.evaluate_robustness_properties()
        robustness = \
            spacecurve.get_robustness_properties().get_robustness_dict()
        spacecurve.evaluate_control_dict('XY', n_points=400)
        vectors = simulator.first_order_error_vectors(
            spacecurve.get_control_dict())
        self.assertTrue(np.isclose(
            vectors['r_z_norm'],
            np.linalg.norm(robustness['closed_test']), rtol=1e-2))
        self.assertLess(vectors['r_z_norm'], 2e-3)

    def test_first_order_leakage_matches_qutrit(self):
        tg_alpha = -100.0
        predicted = drag.first_order_leakage(self.pulse, tg_alpha)
        simulated = simulator.simulate_qutrit(self.pulse, self.target,
                                              tg_alpha)
        self.assertTrue(np.isclose(predicted['leakage'],
                                   simulated['leakage'], rtol=5e-2))
        self.assertTrue(np.isclose(predicted['leakage_trace'].max(),
                                   simulated['peak_leakage'], rtol=5e-2))
        self.assertEqual(predicted['leakage_trace'].shape,
                         simulated['leakage_trace'].shape)

    def test_two_pi_pulses_give_identity(self):
        sequence = controltools.concatenate_control_dicts(self.pulse,
                                                          self.pulse)
        self.assertEqual(len(sequence['times']), 2 * N_POINTS - 1)
        self.assertTrue(np.allclose(sequence['pulse_boundaries'], [1.0, 2.0]))
        fidelity = simulator.simulate_control_dict(
            sequence, qutip.qeye(2))['avg_gate_fidelity']
        self.assertGreater(fidelity, 1 - 1e-8)
        qutrit = simulator.simulate_qutrit(sequence, qutip.qeye(2), -1000.0)
        self.assertGreater(qutrit['subspace_fidelity'], 1 - 1e-4)

    def test_qutrit_noise_grid_matches_qubit_at_large_alpha(self):
        pulse = drag.gaussian_pi_control(n_points=80, sigma=0.15)
        qubit = noise_experiments.simult_source_experiment(
            pulse, self.target, max_points=2)
        qutrit = noise_experiments.simult_source_experiment(
            pulse, self.target, max_points=2, tg_alpha=-1000.0)
        self.assertNotIn('leakage_matrix', qubit)
        self.assertEqual(qutrit['leakage_matrix'].shape, (2, 2))
        # What remains is the 1/tg_alpha Stark shift of |1⟩.
        self.assertTrue(np.allclose(qutrit['infidelity_matrix'],
                                    qubit['infidelity_matrix'],
                                    rtol=5e-2, atol=1e-5))
        plottools.plot_noise_contour(qutrit, key='leakage_matrix',
                                     levels=(-12, -9, -6))
        plt.close('all')


class LeakageHelperTestCase(unittest.TestCase):

    def test_leakage_of_embedded_qubit_gate(self):
        embedded = np.eye(3, dtype=complex)
        embedded[:2, :2] = qutip.sigmax().full()
        self.assertTrue(np.isclose(quantumtools.calculate_leakage(embedded), 0))
        leaked = np.eye(3, dtype=complex)
        leaked[:, 0] = [0, 0, 1]
        from_0, from_1, average = quantumtools.leakage_components(leaked)
        self.assertTrue(np.isclose(from_0, 1.0))
        self.assertTrue(np.isclose(from_1, 0.0))
        self.assertTrue(np.isclose(average, 0.5))
