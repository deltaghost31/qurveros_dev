"""
QuTiP simulation and noise experiments for control dictionaries.

This subpackage does not import the curve classes. Give it the pulses
produced by SpaceCurve.evaluate_control_dict, or a dictionary loaded
from a csv with controltools.load_control_dict. DRAG corrections are
applied to that dictionary by qurveros.drag before simulate_qutrit.
"""
