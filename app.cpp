#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <assert.h> 

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/Dense>

#include "LearningBasedMatte.h"
#include "globalmatting.h"
#include "guidedfilter.h"

using namespace cv;
using namespace std;

int app()
{
    string datasetPath = "C:\\Users\\Marek\\Desktop\\DIP\\viralcam-console-application\\dataset\\";

    //Mat image = imread(datasetPath + "test-rect2.png", cv::IMREAD_COLOR);
    //Mat trimap = imread(datasetPath + "trimap-test-rect2.png", cv::IMREAD_GRAYSCALE);

    Mat image = imread(datasetPath + "tower.jpg", cv::IMREAD_COLOR);
    Mat trimap = imread(datasetPath + "trimap-tower.png", cv::IMREAD_GRAYSCALE);

    double postscale = 2;
    
    //double prescale = 0.5;
    //resize(image, image, Size(0, 0), prescale, prescale);
    //resize(trimap, trimap, Size(0, 0), prescale, prescale);
    //cvtColor(image, image, COLOR_BGR2RGB);

    //Mat alpha_matte(image.size(), CV_64FC1);

    if (!image.data || !trimap.data)
    {
        printf("No image data \n");
        return -1;
    }

    //LearningBasedMatte matte;
    //matte.fill_alpha(alpha_matte, image, trimap);

    /**global**/
    // (optional) exploit the affinity of neighboring pixels to reduce the 
    // size of the unknown region. please refer to the paper
    // 'Shared Sampling for Real-Time Alpha Matting'.
    expansionOfKnownRegions(image, trimap, 9);

    cv::Mat foreground, alpha;
    globalMatting(image, trimap, foreground, alpha);

    // filter the result with fast guided filter
    // alpha = guidedFilter(image, alpha, 10, 1e-5);

    /**global end**/

    
    const char* alpha_matte_window = "Alpha matte";
    const char* iamge_window = "image";
    const char* trimap_window = "trimap";

    //resize(image, image, Size(0, 0), postscale, postscale);
    //resize(trimap, trimap, Size(0, 0), postscale, postscale);
    //resize(alpha_matte, alpha_matte, Size(0, 0), postscale, postscale);

    namedWindow(alpha_matte_window, WINDOW_AUTOSIZE);
    namedWindow(iamge_window, WINDOW_AUTOSIZE);
    namedWindow(trimap_window, WINDOW_AUTOSIZE);
    imshow(alpha_matte_window, alpha);
    imshow(iamge_window, foreground);
    imshow(trimap_window, trimap);

    waitKey(0);

    return 0;
}

void test_matlab_A_slash_B()
{
    MatrixXd A(2, 2);
    A(0, 0) = 1; A(0, 1) = 3;
    A(1, 0) = 5; A(1, 1) = 6;

    MatrixXd B(2, 2);
    B(0, 0) = 2; B(0, 1) = 6;
    B(1, 0) = 8; B(1, 1) = 9;
 
    // A / B in matlab.
    MatrixXd x = B.transpose().colPivHouseholderQr().solve(A.transpose()).transpose();

    assert(areSame(x(0, 0), 0.5)); assert(areSame(x(0, 1), 0));
    assert(areSame(x(1, 0), 0.1)); assert(areSame(x(1, 1), 0.6));
}

void test_matlab_A_backslash_b()
{
    // A \ B in matlab
    MatrixXd A(2, 2);
    A(0, 0) = 1; A(0, 1) = 3;
    A(1, 0) = 5; A(1, 1) = 6;

    VectorXd b(2);
    b(0) = 2;
    b(1) = 8;

    // A \ b in matlab. Thus solve Ax = b
    MatrixXd x = A.colPivHouseholderQr().solve(b);
    // cout << A.inverse() * b << endl << endl;

    assert(areSame(x(0, 0), 1.3333));
    assert(areSame(x(1, 0), 0.2222));
}

void test_sparse_matlab_A_backslash_b()
{
    // A \ B in matlab
    SparseMatrix<double> A(2, 2);
    A.insert(0, 0) = 1; A.insert(0, 1) = 3;
    A.insert(1, 0) = 5; A.insert(1, 1) = 6;
    A.makeCompressed();

    SparseMatrix<double> i(2, 2);
    i.setIdentity();
    i.makeCompressed();

    VectorXd b(2);
    b(0) = 2;
    b(1) = 8;

    // A \ b in matlab. Thus solve Ax = b
    SparseLU<SparseMatrix<double>> solver;
    solver.compute(A);
    VectorXd x = solver.solve(i * b);
    cout << x << endl << endl;

    assert(areSame(x(0, 0), 1.3333));
    assert(areSame(x(1, 0), 0.2222));
}

int main(int argc, char** argv)
{
    //test_matlab_A_slash_B();
    //test_matlab_A_backslash_b();
    //test_sparse_matlab_A_backslash_b();
    return app();
}