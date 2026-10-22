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

from qurveros import drag, plottools
from qurveros.qubit_bench import noise_experiments, quantumtools, simulator
from qurveros.spacecurve import SpaceCurve


TG_ALPHA = -20.0
N_POINTS = 300


def bessel_control(n_curve=2048, n_sim=400):

    """
    The robust Bessel pulse used by the curve tests, on a shorter grid.
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

    def test_order_two_detuning(self):
        corrected = drag.apply_drag(self.pulse, TG_ALPHA, order=2, beta=1.0,
                                    lam=np.sqrt(2))
        # Opposite sign from the |1><1| convention: see drag.apply_drag.
        expected = -((2.0 - 4.0) / 4.0) * self.pulse['omega'] ** 2 / TG_ALPHA
        self.assertTrue(np.allclose(corrected['delta'], expected))

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

    def test_second_order_reduces_subspace_error(self):
        first = simulator.simulate_qutrit(
            drag.apply_drag(self.pulse, TG_ALPHA, order=1, beta=1.0),
            self.target, TG_ALPHA)
        second = simulator.simulate_qutrit(
            drag.apply_drag(self.pulse, TG_ALPHA, order=2, beta=1.0),
            self.target, TG_ALPHA)
        bare = simulator.simulate_qutrit(self.pulse, self.target, TG_ALPHA)
        self.assertLess(second['subspace_infidelity'],
                        first['subspace_infidelity'])
        self.assertLess(second['leakage'], bare['leakage'])

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
        self.assertEqual(int(np.argmin(estimates)), int(np.argmin(leakages)))
        self.assertLess(estimates[int(np.argmin(estimates))], estimates[0])


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

    def test_large_anharmonicity_dephasing_matches(self):
        pulse = drag.gaussian_pi_control(n_points=200, sigma=0.15)
        compared = noise_experiments.drag_dephasing_comparison(
            pulse, qutip.sigmax(), tg_alpha=-1.0e8, order=2, beta=1.0,
            max_points=4)
        self.assertTrue(np.allclose(
            compared['bare_infidelity'],
            compared['drag_infidelity'],
            atol=1e-6))

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
