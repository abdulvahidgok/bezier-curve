#include "bezier_curve.h"
#include <vector>

extern "C" {
    BezierCurve* create_bezier_curve(double* ctrl_points_x,
                                     double* ctrl_points_y,
                                     int num_control_points) {
        std::vector<double> ctrl_x(ctrl_points_x, ctrl_points_x + num_control_points);
        std::vector<double> ctrl_y(ctrl_points_y, ctrl_points_y + num_control_points);
        return new BezierCurve(ctrl_x, ctrl_y);
    }

    void compute_bezier_curve_bernstein(BezierCurve* bezier_curve,
                                        int num_samples,
                                        double* curve_x,
                                        double* curve_y) {
        auto curve = bezier_curve->computeCurveBernstein(num_samples);
        std::copy(curve.first.begin(), curve.first.end(), curve_x);
        std::copy(curve.second.begin(), curve.second.end(), curve_y);
    }

    void compute_bezier_curve_de_casteljau(BezierCurve* bezier_curve,
                                           int num_samples,
                                           double* curve_x,
                                           double* curve_y) {
        auto curve = bezier_curve->computeCurveDeCasteljau(num_samples);
        std::copy(curve.first.begin(), curve.first.end(), curve_x);
        std::copy(curve.second.begin(), curve.second.end(), curve_y);
    }
}
