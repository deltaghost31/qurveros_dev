# Qurveros code reference

Qurveros turns a space curve into a single-qubit pulse. The same three curve classes do that work for every ansatz: `SpaceCurve`, then `OptimizableSpaceCurve`, then `BarqCurve`. A Bezier curve is a `SpaceCurve` with a fixed set of control points. BARQ is an optimizable curve whose parameters are forced to encode a chosen gate.

Two dictionaries carry data between stages.

| Dictionary | Produced by | What it holds |
| --- | --- | --- |
| `frenet_dict` | `SpaceCurve.evaluate_frenet_dict` | Frame, curvature, torsion, speed, position, arclength, and the curve parameters, sampled along the curve parameter |
| `control_dict` | `SpaceCurve.evaluate_control_dict` | Time, envelope `omega`, phase `phi`, detuning `delta`, and the final gate as a 3×3 adjoint matrix |

`qubit_bench` never sees a curve. It only simulates a `control_dict`.

## How a calculation runs

```
curve(x, params)                         curve ansatz
        │
        ▼
frametools / beziertools                 derivatives
        │
        ▼
frenet_dict                              frame, curvature, torsion, length
        │
        ├── RobustnessProperties         closed curve, areas, curve filtering index
        │
        ▼
controltools                             control mode XY, XZ, TTC, or resTTC
        │
        ▼
control_dict                             Ω(t), Φ(t), Δ(t) on a uniform time grid
        │
        ▼
qubit_bench                              QuTiP evolution and noise sweeps
```

Optimization does not leave this path. `prepare_optimization_loss` rebuilds the Frenet dictionary inside a JAX loss, and Optax updates `params`. After optimization, the same `evaluate_*` methods produce pulses.

BARQ only changes how `params` becomes a curve. `barqtools` maps free points to Bezier control points that are closed, start and end with zero curvature, and match the target gate. Those points then go through the Bezier derivative code and the rest of the pipeline.

## Class hierarchy

```
SpaceCurve
├── BezierCurve
└── OptimizableSpaceCurve
    └── BarqCurve
```

`SpaceCurve` owns the pipeline: build a vectorized Frenet function, evaluate it, compute robustness, build pulses, plot, and save. Give it either the position (`order=0`) or the tangent (`order=1`). With `order=1` the position is recovered by integrating the tangent.

`OptimizableSpaceCurve` adds `prepare_optimization_loss`, `optimize`, and `params_history`. It does not change the geometry or the pulses. Losses are functions of a Frenet dictionary; `losses.py` supplies the usual ones, and a caller can pass any function with that shape.

`BarqCurve` adds the BARQ parameterization. Its parameters are a dictionary with `free_points`, `pgf_params`, and `prs_params`. `evaluate_control_dict()` defaults to total-torsion-compensation (`TTC`) and passes the binormal angle in explicitly. Calling it with another mode, such as `'resTTC'`, still works.

`BezierCurve` is how a finished BARQ result is reloaded. `BezierCurve.from_file` reads the csv written by `BarqCurve.save_bezier_control_points`. For a BARQ file the first row is the binormal angle, and it is stored on the Frenet dictionary as `barq_angle` so TTC still works. `misctools.prepare_bezier_from_file` calls this method and exists so older notebooks keep working.

Indexing a curve with an integer returns geometry at that sample. Indexing an optimizable curve with a string such as `'100'` or `'0:50'` returns the parameters, loss, and gradient at those optimization steps.

## Modules

### `settings.py`

Holds the numeric options: sample counts, QuTiP tolerances, noise-sweep ranges, derivative order, and plot colors. `qurveros_settings.json` in the working directory, or up to two parents, overrides them. Importing the package also turns on JAX 64-bit mode and NaN checks (`qurveros/__init__.py`).

Some options are read when a function is built or first compiled (`NUM_DERIVS`, `ANGLE_K_MAX` via `K_RANGE`). Set those before creating curves.

### `frametools.py`

The geometry and the SCQC map, all differentiable except the cumulative Simpson helpers at the bottom.

- `make_deriv_array_fun` builds curvature-ready derivatives with `jax.jacrev`.
- `calculate_frenet_dict` returns the tangent, normal, binormal, speed, curvature, and torsion, including at inflection points.
- `calculate_curve_area`, `calculate_tantrix_area`, and `calculate_cfi_value` are the robustness integrals.
- `calculate_frenet_adj` and `calculate_adj_fidelity` turn the endpoint frames into a gate and its fidelity.
- `calculate_singularity_indices`, `calculate_total_twist`, and `calculate_ttc_detuning` support the TTC control mode.

`losses.py` is a thin wrapper that turns those integrals into scalars for Optax.

### `beziertools.py`

Bezier curves in the Bernstein basis. Derivatives are computed by finite differences of the control points, not by differentiating the polynomials. `bezier_adj` builds the endpoint frame used when BARQ fixes the gate. `bernstein_poly` has a custom derivative so the endpoints 0 and 1 stay well behaved.

### `barqtools.py`

`make_barq_fun` is the BARQ map.

1. `pgf_mod`, if provided, adjusts the gate-fixing parameters.
2. `point_gate_fixing` places the six control points around the two ends so the curve closes, the envelope vanishes there, and the adjoint of the target gate is matched. The angle in that construction is `pgf_params['barq_angle']`.
3. `prs_fun`, if provided, rearranges the interior points. The default passes them through.

`get_default_pgf_params_dict` supplies the starting gate-fixing scales.

### `controltools.py`

`calculate_control_dict` converts a Frenet dictionary into pulses. Time is the normalized arclength. The envelope is the signed curvature, the phase is the integrated torsion, and the detuning depends on the mode:

| Mode | Detuning | Phase |
| --- | --- | --- |
| `XY` | 0 | integrated torsion |
| `XZ` | minus torsion | 0 |
| `TTC` | constant total-torsion compensation | torsion plus detuning |
| `resTTC` | forced to 0 | same phase as TTC |

For TTC, pass `ttc_angle` or let the function read it from BARQ parameters or from `frenet_dict['barq_angle']`. Fields are then resampled onto a uniform time grid. `save_control_dict` and `load_control_dict` use csv. `make_dummy_dict` is a zero-pulse dictionary used by the tests.

### `spacecurve.py` and `optspacecurve.py`

The classes described above. Plotting imports `plottools` only when a plot method runs, so importing a curve does not import Matplotlib.

### `plottools.py` and `plot_configs.py`

`plottools` draws the curve, the control fields (`full` or `compact`), the static-noise contour, and the DRAG leakage scan. `plot_configs` applies a Matplotlib style when it is imported. The notebooks do that with `import qurveros.plot_configs`.

### `misctools.py`

`progbar_range` is the text progress bar used by optimization and by the noise sweeps. `prepare_bezier_from_file` is the compatibility wrapper around `BezierCurve.from_file`. The curve classes are imported inside that function, so using the progress bar does not load the curve stack.

### `qubit_bench/`

| File | Role |
| --- | --- |
| `simulator.py` | Builds the single-qubit Hamiltonian and integrates it with QuTiP. `simulate_qutrit` adds a third level |
| `quantumtools.py` | Average gate fidelity, leakage, and the SU(2) to adjoint (3×3) conversion used as BARQ's target |
| `noise_experiments.py` | Static dephasing, simultaneous drive and dephasing error, 1/f^α dephasing, and the DRAG order scan |
| `noisetools.py` | Kasdin FIR colored noise, plus `estimate_psd` for checking spectra |

Noise experiments copy the control dictionary before changing a field, simulate the copy, and leave the caller's pulses untouched. Sweep ranges and the number of realizations come from `settings`.

### DRAG and leakage

`drag.py` corrects a finished `control_dict`. It does not change the curve or the optimizer. `apply_drag(control_dict, tg_alpha, order=1, beta=1)` takes `tg_alpha = T_g α` in the same normalized units as `omega`. Derivatives use normalized time, not the raw sample values. Order 0 leaves the pulse alone. Order 1 adds the derivative quadrature, scaled by `beta`, and the AC Stark detuning at its analytic coefficient. Order 2 adds Motzoi's higher-order radial and detuning corrections. The stored detuning is the coefficient of `σ_z/2`, so the Stark term has the opposite sign from a formula that adds the shift to `|1⟩⟨1|`. `estimate_leakage` is only the bare Fourier weight of the envelope at `T_g α`. It omits the qubit-evolution factor in the leakage amplitude and is not a substitute for `simulate_qutrit`.

`simulate_qutrit` uses the same `|0⟩–|1⟩` drive as the qubit simulator. It reports final leakage, the largest leakage at any time during the gate, and the fidelity and phase of the 2×2 block. `drag_order_experiment` scans orders and beta values. `drag_dephasing_comparison` repeats that sweep on the qutrit, before and after DRAG, and returns subspace infidelity and leakage. `plot_drag_scan` draws final leakage against beta. `examples/drag_leakage.ipynb` runs the scan on a Gaussian π pulse.

## Where to change things

| Goal | Edit |
| --- | --- |
| New curve ansatz | A `curve(x, params)` function and a `SpaceCurve` |
| New robustness or shape penalty | A function `(frenet_dict) -> scalar`, then pass it to `prepare_optimization_loss` |
| New control mode | `controltools.calculate_control_dict` |
| Gate-fixing layout for BARQ | `barqtools.point_gate_fixing` |
| Noise model or fidelity check | `qubit_bench`, without touching the curve classes |
| Leakage or DRAG | `drag.py` after the control dictionary exists, then `simulate_qutrit` |
| Sample counts, tolerances, colors | `settings.py` or `qurveros_settings.json` |

## Tests

`tests/test_curves.py` checks the constant-curvature pulse, the Bessel robustness conditions, and that Bezier finite differences match automatic differentiation. `tests/test_optimization.py` closes a circle by optimization and checks a short BARQ X-gate against stored control points. `tests/test_qubit_bench.py` checks fidelity and the static noise sweeps against an analytic dephasing formula. `tests/test_drag.py` checks the Motzoi DRAG orders, the Stark detuning sign, qutrit leakage, and qutrit dephasing before and after the correction.
