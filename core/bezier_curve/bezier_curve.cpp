#include "bezier_curve.h"
#include <cmath>
#include <vector>

static std::vector<std::vector<double>> precompute_binomial_coeff(int n) {
    std::vector<std::vector<double>> binom_coeff(n + 1, std::vector<double>(n + 1, 0));
    for (int i = 0; i <= n; ++i) {
        binom_coeff[i][0] = binom_coeff[i][i] = 1.0;
        for (int j = 1; j < i; ++j) {
            binom_coeff[i][j] = binom_coeff[i-1][j-1] + binom_coeff[i-1][j];
        }
    }
    return binom_coeff;
}

static double bernstein_poly(int i, int n, double t, const std::vector<std::vector<double>>& binom_coeff) {
    return binom_coeff[n][i] * pow(t, i) * pow(1 - t, n - i);
}

BezierCurve::BezierCurve(std::vector<double> ctrl_points_x, std::vector<double> ctrl_points_y)
    : control_points_x(ctrl_points_x), control_points_y(ctrl_points_y) {}

std::pair<std::vector<double>, std::vector<double>> BezierCurve::computeCurveBernstein(int num_samples) {
    std::vector<double> output_x(num_samples);
    std::vector<double> output_y(num_samples);

    double delta_t = 1.0 / (num_samples - 1);
    int n = control_points_x.size() - 1;
    auto binom_coeff = precompute_binomial_coeff(n);

    for (int i = 0; i < num_samples; ++i) {
        double t = i * delta_t;
        double x = 0.0;
        double y = 0.0;

        for (int j = 0; j <= n; ++j) {
            double bernstein = bernstein_poly(j, n, t, binom_coeff);
            x += control_points_x[j] * bernstein;
            y += control_points_y[j] * bernstein;
        }

        output_x[i] = x;
        output_y[i] = y;
    }

    return std::make_pair(output_x, output_y);
}

std::pair<std::vector<double>, std::vector<double>> BezierCurve::computeCurveDeCasteljau(int num_samples) {
    std::vector<double> output_x(num_samples);
    std::vector<double> output_y(num_samples);

    double delta_t = 1.0 / (num_samples - 1);
    int n = control_points_x.size();

    for (int i = 0; i < num_samples; ++i) {
        double t = i * delta_t;
        std::vector<double> temp_x = control_points_x;
        std::vector<double> temp_y = control_points_y;

        for (int k = 1; k < n; ++k) {
            for (int j = 0; j < n - k; ++j) {
                temp_x[j] = (1 - t) * temp_x[j] + t * temp_x[j + 1];
                temp_y[j] = (1 - t) * temp_y[j] + t * temp_y[j + 1];
            }
        }

        output_x[i] = temp_x[0];
        output_y[i] = temp_y[0];
    }

    return std::make_pair(output_x, output_y);
}
