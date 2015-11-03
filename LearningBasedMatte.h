#pragma once

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>

using namespace cv;
using namespace std;
using namespace Eigen;

inline bool areSame(double a, double b)
{
    return fabs(a - b) < 1e-3;
}


enum Trimap {
    TRIMAP_BACKGROUND = -1,
    TRIMAP_UKNOWN = 0,
    TRIMAP_FOREGROUND = 1,
};

enum Matte {
    MATTE_BACKGROUND = 0,
    MATTE_FOREGROUND = 1,    
};

class LearningBasedMatte
{
protected: 
    double regularization_factor;
    double lambda;
    int window_pixels;
    int half_window_size;
    Size window_shape;
    MatrixXd lap_identity;
    MatrixXd lap_identity2;
public:
    LearningBasedMatte();
    void prepare_trimap_mask(Mat& trimap_mask, const Mat& trimap);
    void fill_alpha(Mat& alpha, const Mat& image, const Mat& trimap);
        
    void fill_laplacian(SparseMatrix<double>& laplacian, const Mat& mat, const Mat& mask);
    void fill_laplaciant_coeffs(MatrixXd& result, const MatrixXd& matrix, const double lambda);
    void fill_alpha_star(VectorXd& sparse_vector, const Mat& mat);    
};

