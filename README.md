# Bézier Curve Library

This repository contains a C++ and Python implementation of Bézier curves, including both Bernstein polynomial and De Casteljau algorithm methods. The library provides functionality to compute Bézier curves and compare the performance between different implementations.

## Features

- **Bernstein Polynomial Method**: Compute Bézier curves using Bernstein polynomials.
- **De Casteljau Algorithm**: Compute Bézier curves using the iterative De Casteljau algorithm.
- **C++ and Python Implementations**: Both methods are implemented in C++ and Python for performance comparison.
- **Precomputed Binomial Coefficients**: Optimization for Bernstein polynomial calculations.
- **Performance Tests**: Comprehensive performance tests for different methods and implementations.

## Installation
To install the package, run:
```sh
pip install .
```

## Usage

### Python Interface

The library provides a `BezierCurve` class that can be used to compute Bézier curves using both the Bernstein polynomial method and the De Casteljau algorithm.

```python
from bezier_curve import BezierCurve

# Define control points
control_points_x = [0, 1, 2, 3]
control_points_y = [0, 3, 1, 2]
num_samples = 100

# Create a Bézier curve instance
bezier = BezierCurve(control_points_x, control_points_y)

# Compute the curve using Bernstein polynomial method
curve_x_bernstein, curve_y_bernstein = bezier.compute_curve_bernstein(num_samples)

# Compute the curve using De Casteljau algorithm
curve_x_de_casteljau, curve_y_de_casteljau = bezier.compute_curve_de_casteljau(num_samples)
```

## Running Tests
The repository includes performance tests to compare the different methods and implementations. To run the tests, use the following command:
```sh
python -m unittest discover -s tests
```
