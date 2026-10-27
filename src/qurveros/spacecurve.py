"""
This module contains the class definitions of Spacecurve, BezierCurve and
RobustnessProperties.
"""

import jax
import jax.numpy as jnp
import functools

from qurveros.settings import settings
from qurveros import controltools, frametools, beziertools


class SpaceCurve:

    """
    Implements the SCQC to quantum evolution mapping.

    Each curve contains a strictly one traversal parameter (if it's unit-speed,
    that parameter corresponds to the time variable of the quantum evolution)
    and a set of auxiliary parameters that control its shape.

    Attributes:

        frenet_dict (dict): The frenet dictionary (see frametools.py).
        The entries of the frenet_dict are augmented with global curve
        elements.
        interval (list): The endpoints of the traversal parameter interval.
        params: The auxiliary parameters of the curve.
        control_dict (dict): The control dictionary (see controltools.py)
        robustness_prop (RobustnessProperties): Contains the robustness
        properties of the curve.

    Methods:

        evaluate_frenet_dict: Calculates the frenet dictionary
        at a given interval.
        evaluate_robustness_properties: Calculates the robustness properties
        of the curve.
        get_gate_fidelity: Computes the average gate fidelity based on the
        adjoint representation of SCQC.
        calculate_control_dict: Sets the control dictionary
        depending on the control mode.
        plot_position: Plots the position vector of the curve.
        plot_tantrix: Plots the normalized tangent vector of the curve.
        plot_fields: Plots the control fields depending on the plot mode.
        save_control_dict: Saves the control dictionary in a csv file.

    Note:
        When an instance is indexed, a dictionary is returned with entries:
        'curve', 'frame', 'curvature', 'torsion', 'deriv_array', 'length'.
    """

    def __init__(self, *, curve, order, interval, params=None,
                 deriv_array_fun=None):

        """
        Initializes a spacecurve instance with the desired curve.
        In principle, either a curve or the tangent of a curve can be used
        to describe the mapping. The 'order' argument indicates that choice.

        Args:
            curve (list or array): A three-element list arranged as
                [x,y,z] which correspond to the components of the curve.
            order (int): The order of the derivative that the curve argument
                corresponds to.
                order == 0 -> Position vector
                order == 1 -> Tangent
            interval (list): The closed interval of the curve parameter.
            params: The auxiliary parameters of the curve.
            deriv_array_fun (function): In case where the analytic derivatives
            are pre-calculated and are given in a function that returns an
            array of the form number_of_derivatives x 3, the automatic
            differentiation is sidestepped in favor of the function.
            }

        Raises:
            RuntimeError: Raised when not sufficient information is provided
            for the frenet dictionary to be calculated.
            AttributeError: Raised when an attribute is referenced before
            the corresponding quantity is calculated.
        """

        self.frenet_dict = None
        self.interval = interval
        self.params = params
        self.control_dict = None
        self.robustness_props = None
        # order > 0 means the user supplied a tangent, so the position
        # vector is recovered by integrating it after the frame is known.
        self._reconstruct_position = order > 0

        wrapped_curve = None
        if curve is not None:
            def wrapped_curve(x, params):
                return 1.0*jnp.array(curve(x, params)).flatten()

        if curve is None and deriv_array_fun is None:
            raise RuntimeError('Either a curve or a deriv_array function '
                               'must be provided.')

        if curve is None and order == 0:
            raise RuntimeError('The position vector is not defined.')

        if deriv_array_fun is None:
            deriv_array_fun = frametools.make_deriv_array_fun(wrapped_curve,
                                                              order)

        if order == 0:
            position_fun = wrapped_curve
        else:
            def position_fun(x, params):
                return 0.

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def frenet_dict_fun(x, params):

            deriv_array = deriv_array_fun(x, params)

            frenet_dict = frametools.calculate_frenet_dict(deriv_array)

            frenet_dict['x_values'] = x
            frenet_dict['params'] = params

            frenet_dict['curve'] = position_fun(x, params)

            return frenet_dict

        self._frenet_dict_fun = frenet_dict_fun

    def set_params(self, params):

        """
        Sets the default parameters of the curve.
        The associated elements are set to None so that the geometric
        quantities, the robustness properties and the control fields are
        evaluated for the new set of auxiliary parameters.
        """

        self.params = params

        self.frenet_dict = None
        self.control_dict = None
        self.robustness_props = None

    def get_params(self):
        return self.params

    def evaluate_frenet_dict(self, n_points=None):

        """
        Evaluates the frenet dictionary in a specified number of points.

        The frenet dictionary is augmented with:
            'x_values' : The values of x.
            'params' : The parameters for each value of x.
            'curve': The position vector of the curve.
            'length': The (cumulative) length of the curve.

        Args:
            n_points (int): The number of points where the frenet dictionary
            is evaluated at the given interval. The default value is drawn
            from the settings.options['CURVE_POINTS'].
        """

        if n_points is None:
            n_points = settings.options['CURVE_POINTS']

        x_values = jnp.linspace(*self.interval,  n_points)

        frenet_dict = self._frenet_dict_fun(x_values, self.params)

        if self._reconstruct_position:

            print('The curve will be constructed using the tangent vector.')
            curve = frametools.calculate_curve_from_tantrix(frenet_dict)
            frenet_dict['curve'] = jnp.array(curve)

        speed_int = frametools.calculate_cumulative_length(frenet_dict)
        frenet_dict['length'] = speed_int

        self.frenet_dict = frenet_dict

    def get_frenet_dict(self):
        return self.frenet_dict

    def __getitem__(self, index):

        dict_for_getitem = {}

        if self.frenet_dict is None:
            return dict_for_getitem

        for key in ['curve', 'frame', 'curvature', 'torsion',
                    'deriv_array', 'length']:
            dict_for_getitem[key] = self.frenet_dict[key][index]

        return dict_for_getitem

    def __len__(self):
        return len(self.frenet_dict['x_values'])

    def evaluate_robustness_properties(self):

        """
        Calculates and sets the robustness properties of the curve.
        See the RobustnessProperties class definition for more details.

        Raises:
             AttributeError: An exception is raised when the frenet dictionary
             is not initialized.
        """

        self._require('frenet_dict',
                      'Have you initialized the frenet dict?')

        robustness_props = RobustnessProperties(self.frenet_dict)

        self.robustness_props = robustness_props

    def get_robustness_properties(self):

        self._require('robustness_props',
                      'Have you evaluated the robustness properties?')

        return self.robustness_props

    def evaluate_control_dict(self, control_mode, n_points=None, *,
                              ttc_angle=None):

        """
        Sets the control_dict attribute. The control dictionary contains
        the control fields that implement the curve to quantum evolution
        mapping.

        Args:
            control_mode (str): The control type for the Hamiltonian
            (see controltools.py for more information).

            n_points (int): The number of points where the control dictionary
            is evaluated at the given interval. The default value is drawn
            from the settings.options['SIM_POINTS'].
            ttc_angle (float): Binormal angle forwarded to the TTC control
            modes. BarqCurve supplies this from its gate-fixing parameters.
            Other curves leave it empty and the angle is read from the
            frenet dictionary.

        Raises:
             AttributeError: An exception is raised when the frenet dictionary
             is not initialized.

        Note:
            The number of points used to calculate the curve can be different
            from the number of points used for the simulation. It is highly
            advised that the curve is sampled at a higher rate to capture all
            the fields' details.
        """

        self._require('frenet_dict',
                      'Have you initialized the frenet dict?')

        if n_points is None:
            n_points = settings.options['SIM_POINTS']

        self.control_dict = controltools.calculate_control_dict(
            self.frenet_dict,
            control_mode,
            n_points=n_points,
            ttc_angle=ttc_angle)

    def get_control_dict(self):

        return self.control_dict

    def get_gate_fidelity(self, adj_target):

        """
        Calculates the average gate fidelity defined in
        "A simple formula for the average gate fidelity of a
        quantum dynamical operation" by Michael A. Nielsen, equation (18),
        based on the adjoint representation.
        """

        return frametools.calculate_adj_fidelity(
            self.control_dict['adj_curve'],
            adj_target)

    def plot_position(self):

        """
        Plots the position vector of the curve.
        """
        self._require('frenet_dict',
                      'Have you initialized the frenet dict?')

        self._plot_vector(self.frenet_dict['curve'])

    def plot_tantrix(self):

        """
        Plots the (normalized) tangent vector of the curve.
        """
        self._require('frenet_dict',
                      'Have you initialized the frenet dict?')

        self._plot_vector(self.frenet_dict['frame'][:, 0, :])

    def _plot_vector(self, vector):

        from qurveros import plottools

        Tg = self.frenet_dict['length'][-1]

        plottools.plot_curve(vector, self.frenet_dict['length']/Tg)

    def plot_fields(self, plot_mode='full'):

        """
        Plots the control fields depending on the plot mode.

        Raises:
            AttributeError: An exception is raised when a control dictionary
            is not set.
        """
        self._require('control_dict',
                      'Have you chosen a control mode first?')

        from qurveros import plottools

        plottools.plot_fields(self.control_dict, plot_mode)

    def save_control_dict(self, filename):

        """
        Saves the control fields in a csv file with entries described in
        the controltools.py module.
        """

        self._require('control_dict',
                      'Have you chosen a control mode first?')

        controltools.save_control_dict(self.control_dict, filename)

    def _require(self, attr_name, message):

        """
        Raises AttributeError with the historical message when a pipeline
        step is used before the data it needs has been calculated.
        """

        if getattr(self, attr_name) is None:
            raise AttributeError(message)


class BezierCurve(SpaceCurve):

    """
    This class implements SCQC based on a Bezier curve, hence it
    requires only the control points.

    The deriv_array_fun is passed as argument to the constructor since
    the Bezier curves transfer the parameter differentiation to finite
    differences in the control points.
    """

    def __init__(self, points):

        """
        Receives the control points for the construction of the Bezier curve.

        Args:
            points (array): An M x 3 array of control points.

        Raises:
            ValueError: If the number of control points provided is not
            sufficient.
        """

        if points.shape[0] <= settings.options['NUM_DERIVS']:
            raise ValueError('More control points are required.'
                             ' Check NUM_DERIVS value')

        # Finite differences of the control points replace automatic
        # differentiation. That needs enough points that the differences
        # do not become an empty array, which is the check above.
        bezier_deriv_array_fun = beziertools.make_bezier_deriv_array_fun()

        super().__init__(curve=beziertools.bezier_curve_vec,
                         order=0,
                         interval=[0, 1],
                         params=points,
                         deriv_array_fun=bezier_deriv_array_fun)

    @classmethod
    def from_file(cls, filename, *, is_barq, n_points=None):

        """
        Creates a BezierCurve from a csv file of control points.

        Args:
            filename (str): The file where the data is stored.
            is_barq (bool): If the control points come from a BarqCurve,
                the first row stores the binormal angle used by TTC.
            n_points (int): The number of points used to sample the curve.
                The default is settings.options['CURVE_POINTS'].

        Returns:
            A BezierCurve with the frenet dictionary already evaluated.
            When is_barq is true, that dictionary also contains barq_angle.
        """

        import pandas as pd

        if n_points is None:
            n_points = settings.options['CURVE_POINTS']

        control_points = pd.read_csv(filename).to_numpy()

        barq_angle = None
        if is_barq:
            barq_angle = control_points[0, 0]
            control_points = control_points[1:, :]

        beziercurve = cls(control_points)
        beziercurve.evaluate_frenet_dict(n_points)

        if is_barq:
            beziercurve.get_frenet_dict().update(barq_angle=barq_angle)

        return beziercurve


class RobustnessProperties:

    """
    Calculates the robustness properties associated with a given curve.

    When printed, robustness properties that appear as vectors are expressed
    as the squared norm of that vector.
    """

    def __init__(self, frenet_dict):

        self.frenet_dict = frenet_dict
        self.calculate_properties()

    def __repr__(self):

        lines = []

        for rob_test, value in self.robustness_dict.items():

            print_val = value**2

            if rob_test == 'CFI':
                print_val = value

            lines.append(f"|{rob_test:^25}: \t {jnp.sum(print_val):.4e}")

        return '\n'.join(lines)

    def calculate_properties(self):

        """
        Calculates the robustness dictionary which contains all the robustness
        properties associated with the curve.
        """

        robustness_dict = {}

        Tg = self.frenet_dict['length'][-1]

        robustness_dict['closed_test'] = (self.frenet_dict['curve'][-1] -
                                          self.frenet_dict['curve'][0])/Tg

        robustness_dict['curve_area_test'] = \
            frametools.calculate_curve_area(self.frenet_dict)

        robustness_dict['tantrix_area_test'] = \
            frametools.calculate_tantrix_area(self.frenet_dict)

        robustness_dict['CFI'] = \
            frametools.calculate_cfi_value(self.frenet_dict)

        self.robustness_dict = robustness_dict

    def get_robustness_dict(self):

        return self.robustness_dict
