#include "LearningBasedMatte.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>
#include <iomanip>

LearningBasedMatte::LearningBasedMatte()
{
    this->regularization_factor = 800.0;
    this->lambda = 1e-7;
    this->window_shape = Size(3, 3);
    this->window_pixels = window_shape.area();    
    this->half_window_size = (this->window_shape.width - 1) / 2;

    MatrixXd identity(window_pixels, window_pixels);
    identity.setIdentity();
    identity(identity.rows() - 1, identity.cols() - 1) = 0;
    this->lap_identity = identity * this->lambda;

    MatrixXd identity2(window_pixels, window_pixels);
    identity2.setIdentity();

    this->lap_identity2 = identity2;
}


void LearningBasedMatte::prepare_trimap_mask(Mat& trimap_mask, const Mat& trimap)
{
    // Note: Colum major does not matter.
    for (int y = 0; y < trimap_mask.rows; y++)
    {
        for (int x = 0; x < trimap_mask.cols; x++)
        {
            if (trimap.at<uchar>(y, x) == 255 || trimap.at<uchar>(y, x) == 0)
                trimap_mask.at<short>(y, x) = 1;
            else
                trimap_mask.at<short>(y, x) = 0;
        }
    }

    // Apply the erosion operation
    Mat eroded_mask;
    erode(trimap_mask, eroded_mask, getStructuringElement(MORPH_RECT, this->window_shape), Point(-1, -1), 1);        
}

void LearningBasedMatte::fill_alpha(Mat& alpha, const Mat& image, const Mat& trimap)
{
    const int image_pixels = image.size().area();

    // Normalize image. Values will be in [0, 1] interval.
    Mat normalized_image;
    image.convertTo(normalized_image, CV_64FC4);
    normalized_image /= 255.0;
    
    // Prepare trimap mask. Trimap mask contains 1 for foreground and background, for unknown region is set to 0.
    Mat trimap_mask(trimap.size(), CV_16SC1);
    prepare_trimap_mask(trimap_mask, trimap);
       
    // Calculate laplacian. (m x n) x (m x n) matrix.
    SparseMatrix<double> laplacian(image_pixels, image_pixels);
    this->fill_laplacian(laplacian, normalized_image, trimap_mask);
    
    // Set alpha star. Vector contains know alpha values -1 for backgroud, 1 for foreground, 0 for uknown.
    // Note: vector is in converted from column major matrix.
    VectorXd alpha_star(image_pixels);
    this->fill_alpha_star(alpha_star, trimap);

    // Create regularization matrix (m x n) x (m x n) matrix where diagonal is set to alpha star in absolute value.
    SparseMatrix<double> regularization_matrix(image_pixels, image_pixels);
    for (int i = 0; i < image_pixels; i++)
    {
        regularization_matrix.insert(i, i) = this->regularization_factor * abs(alpha_star(i));
    }

    // Get identity matrix
    SparseMatrix<double> identity(image_pixels, image_pixels);
    identity.setIdentity();

    // Solve qudratic minimization problem as Ax = b
    SparseLU<SparseMatrix<double>> solver;
    solver.compute(laplacian + regularization_matrix + identity * 1e-6);
    VectorXd computed_alpha = solver.solve(regularization_matrix * alpha_star);

    // Put result to the image
    // Note: !Column major order matter.
    for (int x = 0, index = 0; x < trimap.cols; x++) {
        for (int y = 0; y < trimap.rows; y++, index++) {
            double value = computed_alpha(index)*0.5 + 0.5;
            if (value > 1) value = 1;
            if (value < 0) value = 0;
            alpha.at<double>(y, x) = value;
        }
    }
}


void LearningBasedMatte::fill_laplacian(SparseMatrix<double>& laplacian, const Mat& image, const Mat& trimap_mask)
{
    const int feature_dimension = 3;
    const Size image_shape = image.size();
    const int image_pixels = image_shape.area();
    const int pixels_in_window = this->window_shape.area();

    // Create indices matrix
    Mat image_indices(image.size(), CV_32S);
    // Note: !Column major order matter.
    for (int x = 0, index = 0; x < image_indices.cols; x++) {
        for (int y = 0; y < image_indices.rows; y++, index++) {
            image_indices.at<int>(y, x) = index;
        }
    }

    MatrixXd channels = MatrixXd::Ones(pixels_in_window, feature_dimension + 1);
    Rect window(0, 0, this->window_shape.width, this->window_shape.height);
    MatrixXd coeffs;
    
    // Note: Colum major does not matter.
    for (int y = this->half_window_size, index = this->half_window_size; y < image_shape.height - this->half_window_size; y++) {
        cout << y << "(" << image_shape.height << ") / " << index / static_cast<double>(image_pixels) << endl;
        
        for (int x = this->half_window_size; x < image_shape.width - this->half_window_size; x++, index++) {
            // Process only unknown regions
            if (trimap_mask.at<short>(y, x))
                continue;
            
            window.x = x - this->half_window_size;
            window.y = y - this->half_window_size;
                        
            // Mat is row major, but we need to acces matrix in column major order. Since it is accessed by index transposition will make this trick.
            Mat image_values_in_window = image(window).t();           
            
            // Note: !Column major order matter.
            for (int i = 0; i < pixels_in_window; i++) {
                Vec3d color = image_values_in_window.at<Vec3d>(i);
                channels(i, 0) = color[0];
                channels(i, 1) = color[1];
                channels(i, 2) = color[2];
                channels(i, 3) = 1;
            }
            
            this->fill_laplaciant_coeffs(coeffs, channels, this->lambda);

            // Mat is row major, but we need to acces matrix in column major order. Since it is accessed by index transposition will make this trick.
            Mat image_indices_in_window = image_indices(window).t();

            // Note: !Column major order matter.
            for (int i = 0; i < coeffs.cols(); i++) {
                for (int j = 0; j < coeffs.rows(); j++) {
                    laplacian.coeffRef(image_indices_in_window.at<int>(j), image_indices_in_window.at<int>(i)) += coeffs(j, i);
                }
            }
        }
    }
}


void LearningBasedMatte::fill_laplaciant_coeffs(MatrixXd& result, const MatrixXd& channels, const double lambda)
{  
    MatrixXd xi_times_xi_t = channels * channels.transpose();

    // Solve Ax = B    
    MatrixXd identity_minus_f = lap_identity2 - (xi_times_xi_t + lap_identity).transpose()
                                                         .fullPivLu()
                                                         .solve(xi_times_xi_t.transpose())
                                                         .transpose();

    result = identity_minus_f.transpose() * identity_minus_f;
}

void LearningBasedMatte::fill_alpha_star(VectorXd& alpha_star, const Mat& trimap)
{
    // Note: !Column major order matter.
    //cout << trimap << endl;
    for (int x = 0, index = 0; x < trimap.cols; x++) {
        for (int y = 0; y < trimap.rows; y++, index++) {        
            uchar value = trimap.at<uchar>(y, x);
            if (value == 255)
                alpha_star(index) = 1;
            else if (value == 0)
                alpha_star(index) = -1;
            else
                alpha_star(index) = 0;
        }
    }
}