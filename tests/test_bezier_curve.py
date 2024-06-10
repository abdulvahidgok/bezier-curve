import math
import unittest
import time

import numpy as np
from bezier_curve import BezierCurve


def bernstein_poly(i, n, t):
    binomial_coeff = math.comb(n, i)
    return binomial_coeff * (t ** i) * ((1 - t) ** (n - i))


def bezier_curve_bernstein(control_points_x, control_points_y, num_samples):
    n = len(control_points_x) - 1
    t = np.linspace(0, 1, num_samples)
    curve_x = np.zeros(num_samples)
    curve_y = np.zeros(num_samples)

    for i in range(num_samples):
        for j in range(n + 1):
            bernstein = bernstein_poly(j, n, t[i])
            curve_x[i] += control_points_x[j] * bernstein
            curve_y[i] += control_points_y[j] * bernstein

    return curve_x, curve_y


def bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples):
    n = len(control_points_x)
    t = np.linspace(0, 1, num_samples)
    curve_x = np.zeros(num_samples)
    curve_y = np.zeros(num_samples)

    for i, ti in enumerate(t):
        temp_x = np.array(control_points_x, dtype=float)
        temp_y = np.array(control_points_y, dtype=float)

        for k in range(1, n):
            for j in range(n - k):
                temp_x[j] = (1 - ti) * temp_x[j] + ti * temp_x[j + 1]
                temp_y[j] = (1 - ti) * temp_y[j] + ti * temp_y[j + 1]

        curve_x[i] = temp_x[0]
        curve_y[i] = temp_y[0]

    return curve_x, curve_y


class TestBezierCurve(unittest.TestCase):

    def test_bezier_curve_cpp_bernstein(self):
        control_points_x = [0, 1, 2, 3]
        control_points_y = [0, 3, 1, 2]
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_cpp_de_casteljau(self):
        control_points_x = [0, 1, 2, 3]
        control_points_y = [0, 3, 1, 2]
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_different_points_bernstein(self):
        control_points_x = [0, 2, 4, 6, 8]
        control_points_y = [0, 5, 2, 5, 0]
        num_samples = 200
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_different_points_de_casteljau(self):
        control_points_x = [0, 2, 4, 6, 8]
        control_points_y = [0, 5, 2, 5, 0]
        num_samples = 200
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_less_samples_bernstein(self):
        control_points_x = [0, 1, 3, 4]
        control_points_y = [0, 4, 2, 3]
        num_samples = 50
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_less_samples_de_casteljau(self):
        control_points_x = [0, 1, 3, 4]
        control_points_y = [0, 4, 2, 3]
        num_samples = 50
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_more_samples_bernstein(self):
        control_points_x = [0, 1, 3, 4]
        control_points_y = [0, 4, 2, 3]
        num_samples = 500
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_more_samples_de_casteljau(self):
        control_points_x = [0, 1, 3, 4]
        control_points_y = [0, 4, 2, 3]
        num_samples = 500
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_two_points_bernstein(self):
        control_points_x = [0, 1]
        control_points_y = [0, 1]
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_two_points_de_casteljau(self):
        control_points_x = [0, 1]
        control_points_y = [0, 1]
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_complex_bernstein(self):
        control_points_x = [0, 1, 2, 3, 4, 5, 6, 7]
        control_points_y = [0, 3, 1, 2, 5, 4, 2, 0]
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_complex_de_casteljau(self):
        control_points_x = [0, 1, 2, 3, 4, 5, 6, 7]
        control_points_y = [0, 3, 1, 2, 5, 4, 2, 0]
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_nonuniform_bernstein(self):
        control_points_x = [0, 2, 5, 10]
        control_points_y = [0, 2, 1, 5]
        num_samples = 75
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_nonuniform_de_casteljau(self):
        control_points_x = [0, 2, 5, 10]
        control_points_y = [0, 2, 1, 5]
        num_samples = 75
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_increasing_control_points_bernstein(self):
        control_points_x = list(range(10))
        control_points_y = list(range(10))
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_increasing_control_points_de_casteljau(self):
        control_points_x = list(range(10))
        control_points_y = list(range(10))
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_decreasing_control_points_bernstein(self):
        control_points_x = list(range(10, 0, -1))
        control_points_y = list(range(10, 0, -1))
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_decreasing_control_points_de_casteljau(self):
        control_points_x = list(range(10, 0, -1))
        control_points_y = list(range(10, 0, -1))
        num_samples = 100
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_random_points_bernstein(self):
        np.random.seed(0)
        control_points_x = np.random.rand(10).tolist()
        control_points_y = np.random.rand(10).tolist()
        num_samples = 150
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_bernstein(num_samples)
        curve_x_py, curve_y_py = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bezier_curve_random_points_de_casteljau(self):
        np.random.seed(0)
        control_points_x = np.random.rand(10).tolist()
        control_points_y = np.random.rand(10).tolist()
        num_samples = 150
        bezier = BezierCurve(control_points_x, control_points_y)
        curve_x_cpp, curve_y_cpp = bezier.compute_curve_de_casteljau(num_samples)
        curve_x_py, curve_y_py = bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        np.testing.assert_allclose(curve_x_cpp, curve_x_py, rtol=1e-5)
        np.testing.assert_allclose(curve_y_cpp, curve_y_py, rtol=1e-5)

    def test_bernstein_vs_de_casteljau(self):
        control_points_x = [0, 1, 2, 3]
        control_points_y = [0, 3, 1, 2]
        num_samples = 100
        curve_x_bernstein, curve_y_bernstein = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        curve_x_de_casteljau, curve_y_de_casteljau = bezier_curve_de_casteljau(control_points_x, control_points_y,
                                                                               num_samples)
        np.testing.assert_allclose(curve_x_bernstein, curve_x_de_casteljau, rtol=1e-5)
        np.testing.assert_allclose(curve_y_bernstein, curve_y_de_casteljau, rtol=1e-5)

    def test_complex_bernstein_vs_de_casteljau(self):
        control_points_x = [0, 1, 2, 3, 4, 5, 6, 7]
        control_points_y = [0, 3, 1, 2, 5, 4, 2, 0]
        num_samples = 100
        curve_x_bernstein, curve_y_bernstein = bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        curve_x_de_casteljau, curve_y_de_casteljau = bezier_curve_de_casteljau(control_points_x, control_points_y,
                                                                               num_samples)
        np.testing.assert_allclose(curve_x_bernstein, curve_x_de_casteljau, rtol=1e-5)
        np.testing.assert_allclose(curve_y_bernstein, curve_y_de_casteljau, rtol=1e-5)

    def test_performance_cpp_bernstein_vs_python_bernstein(self):
        control_points_x = np.random.rand(10).tolist()
        control_points_y = np.random.rand(10).tolist()
        num_samples = 1000

        bezier = BezierCurve(control_points_x, control_points_y)

        start_time = time.time()
        bezier.compute_curve_bernstein(num_samples)
        cpp_time = time.time() - start_time

        start_time = time.time()
        bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        python_time = time.time() - start_time

        print(f"CPP Bernstein time: {cpp_time:.6f} seconds")
        print(f"Python Bernstein time: {python_time:.6f} seconds")
        self.assertTrue(cpp_time < python_time, "C++ Bernstein should be faster than Python Bernstein")

    def test_performance_cpp_de_casteljau_vs_python_de_casteljau(self):
        control_points_x = np.random.rand(10).tolist()
        control_points_y = np.random.rand(10).tolist()
        num_samples = 1000

        bezier = BezierCurve(control_points_x, control_points_y)

        start_time = time.time()
        bezier.compute_curve_de_casteljau(num_samples)
        cpp_time = time.time() - start_time

        start_time = time.time()
        bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        python_time = time.time() - start_time

        print(f"CPP De Casteljau time: {cpp_time:.6f} seconds")
        print(f"Python De Casteljau time: {python_time:.6f} seconds")
        self.assertTrue(cpp_time < python_time,
                        "C++ De Casteljau should be faster than Python De Casteljau")

    def test_performance_python_bernstein_vs_python_de_casteljau(self):
        control_points_x = np.random.rand(10).tolist()
        control_points_y = np.random.rand(10).tolist()
        num_samples = 1000

        start_time = time.time()
        bezier_curve_bernstein(control_points_x, control_points_y, num_samples)
        bernstein_time = time.time() - start_time

        start_time = time.time()
        bezier_curve_de_casteljau(control_points_x, control_points_y, num_samples)
        de_casteljau_time = time.time() - start_time

        print(f"Python Bernstein time: {bernstein_time:.6f} seconds")
        print(f"Python De Casteljau time: {de_casteljau_time:.6f} seconds")
        self.assertTrue(bernstein_time < de_casteljau_time,
                        "Python Bernstein should be faster than Python De Casteljau")

    def test_performance_cpp_bernstein_vs_cpp_de_casteljau(self):
        control_points_x = np.random.rand(10).tolist()
        control_points_y = np.random.rand(10).tolist()
        num_samples = 1000

        bezier = BezierCurve(control_points_x, control_points_y)

        start_time = time.time()
        bezier.compute_curve_bernstein(num_samples)
        bernstein_time = time.time() - start_time

        start_time = time.time()
        bezier.compute_curve_de_casteljau(num_samples)
        de_casteljau_time = time.time() - start_time

        print(f"CPP Bernstein time: {bernstein_time:.6f} seconds")
        print(f"CPP De Casteljau time: {de_casteljau_time:.6f} seconds")

    def test_performance_cpp_bernstein_vs_cpp_de_casteljau_large_data(self):
        control_points_x = np.random.rand(100).tolist()
        control_points_y = np.random.rand(100).tolist()
        num_samples = 100000

        bezier = BezierCurve(control_points_x, control_points_y)

        start_time = time.time()
        bezier.compute_curve_bernstein(num_samples)
        bernstein_time = time.time() - start_time

        start_time = time.time()
        bezier.compute_curve_de_casteljau(num_samples)
        de_casteljau_time = time.time() - start_time

        print(f"CPP Bernstein with large data time: {bernstein_time:.6f} seconds")
        print(f"CPP De Casteljau with large data time: {de_casteljau_time:.6f} seconds")


if __name__ == '__main__':
    unittest.main()
