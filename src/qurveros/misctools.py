"""
Generic helpers that do not depend on the curve classes.

Curve loading lives on BezierCurve.from_file. prepare_bezier_from_file
remains here so existing notebooks and scripts keep working, and it
imports the curve classes only when it is called.
"""

import sys


def progbar_range(num_iters, title='', depth=0):

    """
    Creates a simple progress bar depending on the depth of the nested
    iterations.

    Args:
        num_iters (int): The number of iterations.
        title (str): The title for the progress bar.
        depth (int): Specifies the position in the tree of nested
        iterations.

    Returns:
        A generator that prints the percentage of the current step with
        respect to the number of iterations.
    """

    if depth > 0:
        title = '\t'*depth + ' \\' + '-'*4 + ' ' + title

    print('\n', end='', file=sys.stderr)

    for step in range(num_iters):
        percentage = 100*step/(num_iters)

        print(f'\r{title}:\t{percentage:^6.2f}'+'%', end='', file=sys.stderr)
        yield step

    print(f'\r{title}:\t{100.00:^6.2f}'+'%', end='', file=sys.stderr)
    if depth > 0:
        print('\r'+' '*2*len(title), end='', file=sys.stderr)
    else:
        print('\n', end='', file=sys.stderr)


def prepare_bezier_from_file(filename, *, is_barq, n_points=None):

    """
    Creates a BezierCurve instance from a csv file.

    Args:
        filename (str): The name of the file where the data is stored.
        is_barq (bool): If the control points correspond to a BarqCurve,
        the first element is the barq angle for the TTC method.
        n_points (int): The number of points to sample the curve.

    Returns:
        A BezierCurve instance with the frenet_dict initialized and the
        barq_angle appropriately set (if applicable).
    """

    # Imported here so that users of progbar_range do not load the
    # curve, JAX differentiation, or plotting stack.
    from qurveros.spacecurve import BezierCurve

    return BezierCurve.from_file(filename, is_barq=is_barq, n_points=n_points)
