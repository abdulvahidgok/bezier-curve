#ifndef BEZIER_CURVE_H
#define BEZIER_CURVE_H

#include <vector>
#include <utility>

class BezierCurve {
private:
    std::vector<double> control_points_x;
    std::vector<double> control_points_y;

public:
    BezierCurve(std::vector<double> ctrl_points_x, std::vector<double> ctrl_points_y);
    std::pair<std::vector<double>, std::vector<double>> computeCurveBernstein(int num_samples);
    std::pair<std::vector<double>, std::vector<double>> computeCurveDeCasteljau(int num_samples);
};

#endif
