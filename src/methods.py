"""
Bernstein Polynomials and De Casteljau Algorithm for Bézier Curves

Bernstein Polynomials:
- These are the fundamental basis functions used to define Bézier curves.
- They provide a direct way to compute the Bézier curve using control points and their weights.
- The formula for a Bernstein polynomial is:
  B_{i,n}(t) = binom(n, i) * t^i * (1 - t)^(n - i)
- Bézier curve using Bernstein polynomials:
  P(t) = sum(B_{i,n}(t) * P_i)

De Casteljau Algorithm:
- This is an iterative method for computing Bézier curves.
- It works by repeatedly calculating weighted averages of control points.
- This method is numerically stable and suitable for calculating specific points on the curve.
- The recursive formula is:
  P_{i}^{(k)}(t) = (1 - t) * P_{i}^{(k-1)}(t) + t * P_{i+1}^{(k-1)}(t)
"""

import ctypes
import numpy as np
import os

lib_path = os.path.join(os.path.dirname(__file__), 'libbezier_curve.so')
bezier_lib = ctypes.cdll.LoadLibrary(lib_path)

create_bezier_curve_cpp = bezier_lib.create_bezier_curve
create_bezier_curve_cpp.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.float64),
                                    ctypes.c_int]
create_bezier_curve_cpp.restype = ctypes.c_void_p

compute_bezier_curve_bernstein_cpp = bezier_lib.compute_bezier_curve_bernstein
compute_bezier_curve_bernstein_cpp.argtypes = [ctypes.c_void_p, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64),
                                               np.ctypeslib.ndpointer(dtype=np.float64)]

compute_bezier_curve_de_casteljau_cpp = bezier_lib.compute_bezier_curve_de_casteljau
compute_bezier_curve_de_casteljau_cpp.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  np.ctypeslib.ndpointer(dtype=np.float64),
                                                  np.ctypeslib.ndpointer(dtype=np.float64)]


class BezierCurve:
    def __init__(self, control_points_x, control_points_y):
        self.num_control_points = len(control_points_x)
        self.control_points_x = np.array(control_points_x, dtype=np.float64)
        self.control_points_y = np.array(control_points_y, dtype=np.float64)
        self.bezier_curve = create_bezier_curve_cpp(self.control_points_x, self.control_points_y,
                                                    self.num_control_points)

    def compute_curve_bernstein(self, num_samples):
        curve_x = np.zeros(num_samples, dtype=np.float64)
        curve_y = np.zeros(num_samples, dtype=np.float64)
        compute_bezier_curve_bernstein_cpp(self.bezier_curve, num_samples, curve_x, curve_y)
        return curve_x, curve_y

    def compute_curve_de_casteljau(self, num_samples):
        curve_x = np.zeros(num_samples, dtype=np.float64)
        curve_y = np.zeros(num_samples, dtype=np.float64)
        compute_bezier_curve_de_casteljau_cpp(self.bezier_curve, num_samples, curve_x, curve_y)
        return curve_x, curve_y
