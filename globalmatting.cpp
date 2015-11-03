#include "globalmatting.h"

using namespace std;

template <typename T> static inline T sqr(T a)
{
    return a * a;
}

static vector<cv::Point> findBoundaryPixels(const cv::Mat_<uchar> &trimap, int a, int b)
{
    vector<cv::Point> result;

    for (int x = 1; x < trimap.cols - 1; ++x)
        for (int y = 1; y < trimap.rows - 1; ++y)
        {
            if (trimap(y, x) == a)
            {
                if (trimap(y - 1, x) == b ||
                    trimap(y + 1, x) == b ||
                    trimap(y, x - 1) == b ||
                    trimap(y, x + 1) == b)
                {
                    result.push_back(cv::Point(x, y));
                }
            }
        }

    return result;
}

// Eq. 2
static float calculateAlpha(const cv::Vec3b &F, const cv::Vec3b &B, const cv::Vec3b &I)
{
    float result = 0;
    float div = 1e-6f;
    for (int c = 0; c < 3; ++c)
    {
        float f = F[c];
        float b = B[c];
        float i = I[c];

        result += (i - b) * (f - b);
        div += (f - b) * (f - b);
    }

    return min(max(result / div, 0.f), 1.f);
}

// Eq. 3
static float colorCost(const cv::Vec3b &F, const cv::Vec3b &B, const cv::Vec3b &I, float alpha)
{
    float result = 0;
    for (int c = 0; c < 3; ++c)
    {
        float f = F[c];
        float b = B[c];
        float i = I[c];

        result += sqr(i - (alpha * f + (1 - alpha) * b));
    }

    return sqrt(result);
}

// Eq. 4
static float distCost(const cv::Point &p0, const cv::Point &p1, float minDist)
{
    int dist = sqr(p0.x - p1.x) + sqr(p0.y - p1.y);
    return sqrt(static_cast<float>(dist)) / minDist;
}

static float colorDist(const cv::Vec3b &I0, const cv::Vec3b &I1)
{
    int result = 0;

    for (int c = 0; c < 3; ++c)
        result += sqr(static_cast<int>(I0[c]) - static_cast<int>(I1[c]));

    return sqrt(static_cast<float>(result));
}

static float nearestDistance(const std::vector<cv::Point> &boundary, const cv::Point &p)
{
    int minDist2 = INT_MAX;
    for (size_t i = 0; i < boundary.size(); ++i)
    {
        int dist2 = sqr(boundary[i].x - p.x)  + sqr(boundary[i].y - p.y);
        minDist2 = min(minDist2, dist2);
    }

    return sqrt(static_cast<float>(minDist2));
}


// for sorting the boundary pixels according to intensity
struct IntensityComp
{
    IntensityComp(const cv::Mat_<cv::Vec3b> &image) : image(image)
    {

    }

    bool operator()(const cv::Point &p0, const cv::Point &p1) const
    {
        const cv::Vec3b &c0 = image(p0.y, p0.x);
        const cv::Vec3b &c1 = image(p1.y, p1.x);

        return c0[0] + c0[1] + c0[2] < c1[0] + c1[1] + c1[2];
    }

    const cv::Mat_<cv::Vec3b> &image;
};

static void expansionOfKnownRegions(const cv::Mat_<cv::Vec3b> &image,
                                    cv::Mat_<uchar> &trimap,
                                    int r, float c)
{
    const int image_width = image.cols;
    const int image_heigth = image.rows;

    for (int x = 0; x < image_width; ++x) {
        for (int y = 0; y < image_heigth; ++y)
        {
            if (trimap(y, x) != 128)
                continue;

            const cv::Vec3b &image_color = image(y, x);

            for (int j = y - r; j <= y + r; ++j) {
                for (int i = x - r; i <= x + r; ++i)
                {
                    if (i < 0 || i >= image_width || j < 0 || j >= image_heigth)
                        continue;

                    if (trimap(j, i) != 0 && trimap(j, i) != 255)
                        continue;

                    const cv::Vec3b &I2 = image(j, i);

                    float pd = sqrt(static_cast<float>(sqr(x - i) + sqr(y - j)));
                    float cd = colorDist(image_color, I2);

                    if (pd <= r && cd <= c)
                    {
                        if (trimap(j, i) == 0)
                            trimap(y, x) = 1;
                        else if (trimap(j, i) == 255)
                            trimap(y, x) = 254;
                    }
                }
            }
        }
    }
    
    for (int x = 0; x < trimap.cols; ++x) {
        for (int y = 0; y < trimap.rows; ++y)
        {
            if (trimap(y, x) == 1)
                trimap(y, x) = 0;
            else if (trimap(y, x) == 254)
                trimap(y, x) = 255;

        }
    }        
}

// erode foreground and background regions to increase the size of unknown region
// todo: refactor me: make own erode
static void erodeFB(cv::Mat &_trimap, int r)
{
    cv::Mat_<uchar> &trimap = (cv::Mat_<uchar>&)_trimap;

    int w = trimap.cols;
    int h = trimap.rows;

    cv::Mat_<uchar> foreground(trimap.size(), (uchar)0);
    cv::Mat_<uchar> background(trimap.size(), (uchar)0);

    for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
    {
        if (trimap(y, x) == 0)
            background(y, x) = 1;
        else if (trimap(y, x) == 255)
            foreground(y, x) = 1;
    }


    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(r, r));

    cv::erode(background, background, kernel);
    cv::erode(foreground, foreground, kernel);

    for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
    {
        if (background(y, x) == 0 && foreground(y, x) == 0)
            trimap(y, x) = 128;
    }
}


struct Sample
{
    int foreground_index, background_index;
    float distance2foreground, distance2background;
    float cost, alpha;
};

static void calculateAlphaPatchMatch(const cv::Mat_<cv::Vec3b> &image,
        const cv::Mat_<uchar> &trimap,
        const std::vector<cv::Point> &foregroundBoundary,
        const std::vector<cv::Point> &backgroundBoundary,
        std::vector<std::vector<Sample> > &samples)
{
    int image_width = image.cols;
    int image_heigth = image.rows;

    samples.resize(image_heigth, std::vector<Sample>(image_width));

    for (int y = 0; y < image_heigth; ++y) {
        for (int x = 0; x < image_width; ++x) {
            if (trimap(y, x) == 128) {
                cv::Point p(x, y);

                samples[y][x].foreground_index = rand() % foregroundBoundary.size();
                samples[y][x].background_index = rand() % backgroundBoundary.size();
                samples[y][x].distance2foreground = nearestDistance(foregroundBoundary, p);
                samples[y][x].distance2background = nearestDistance(backgroundBoundary, p);
                samples[y][x].cost = FLT_MAX;
            }
        }
    }
    
    for (int iteration = 0; iteration < 10; ++iteration) {
        for (int y = 0; y < image_heigth; ++y) {
            for (int x = 0; x < image_width; ++x) {
                if (trimap(y, x) != 128)
                    continue;

                const cv::Point current_point(x, y);
                Sample &sample = samples[y][x];
                const cv::Vec3b &image_color = image(y, x);               

                // propagation
                for (int y2 = y - 1; y2 <= y + 1; ++y2) {
                    for (int x2 = x - 1; x2 <= x + 1; ++x2) {
                        if (x2 < 0 || x2 >= image_width || y2 < 0 || y2 >= image_heigth)
                            continue;

                        if (trimap(y2, x2) != 128)
                            continue;

                        Sample &second_sample = samples[y2][x2];

                        const cv::Point &foreground_point = foregroundBoundary[second_sample.foreground_index];
                        const cv::Point &background_point = backgroundBoundary[second_sample.background_index];

                        const cv::Vec3b foreground_color = image(foreground_point.y, foreground_point.x);
                        const cv::Vec3b background_color = image(background_point.y, background_point.x);

                        float alpha = calculateAlpha(foreground_color, background_color, image_color);

                        float cost = colorCost(foreground_color, background_color, image_color, alpha) + 
                            distCost(current_point, foreground_point, sample.distance2foreground) + 
                            distCost(current_point, background_point, sample.distance2background);

                        if (cost < sample.cost)
                        {
                            sample.foreground_index = second_sample.foreground_index;
                            sample.background_index = second_sample.background_index;
                            sample.cost = cost;
                            sample.alpha = alpha;
                        }
                    }
                }
                // random walk
                int max_boundary_width = static_cast<int>(max(foregroundBoundary.size(), backgroundBoundary.size()));

                for (int k = 0;; k++)
                {
                    float r = max_boundary_width * pow(0.5f, k);

                    if (r < 1)
                        break;

                    int foreground_distance = r * (rand() / (RAND_MAX + 1.f));
                    int background_distance = r * (rand() / (RAND_MAX + 1.f));

                    int foreground_index = sample.foreground_index + foreground_distance;
                    int background_index = sample.background_index + background_distance;

                    if (foreground_index < 0 || foreground_index >= foregroundBoundary.size() || background_index < 0 || background_index >= backgroundBoundary.size())
                        continue;

                    const cv::Point &foreground_point = foregroundBoundary[foreground_index];
                    const cv::Point &background_point = backgroundBoundary[background_index];

                    const cv::Vec3b foreground_color = image(foreground_point.y, foreground_point.x);
                    const cv::Vec3b background_color = image(background_point.y, background_point.x);

                    float alpha = calculateAlpha(foreground_color, background_color, image_color);

                    float cost = colorCost(foreground_color, background_color, image_color, alpha) + 
                        distCost(current_point, foreground_point, sample.distance2foreground) + 
                        distCost(current_point, background_point, sample.distance2background);

                    if (cost < sample.cost)
                    {
                        sample.foreground_index = foreground_index;
                        sample.background_index = background_index;
                        sample.cost = cost;
                        sample.alpha = alpha;
                    }
                }
            }
        }
    }
}

static void expansionOfKnownRegionsHelper(const cv::Mat &_image,
                                          cv::Mat &_trimap,
                                          int r, float c)
{
    const cv::Mat_<cv::Vec3b> &image = (const cv::Mat_<cv::Vec3b> &)_image;
    cv::Mat_<uchar> &trimap = (cv::Mat_<uchar>&)_trimap;

    int w = image.cols;
    int h = image.rows;

    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            if (trimap(y, x) != 128)
                continue;

            const cv::Vec3b &I = image(y, x);

            for (int j = y - r; j <= y + r; ++j)
            for (int i = x - r; i <= x + r; ++i) {
                if (i < 0 || i >= w || j < 0 || j >= h)
                    continue;

                if (trimap(j, i) != 0 && trimap(j, i) != 255)
                    continue;

                const cv::Vec3b &I2 = image(j, i);

                float pd = sqrt((float)(sqr(x - i) + sqr(y - j)));
                float cd = colorDist(I, I2);

                if (pd <= r && cd <= c)
                {
                    if (trimap(j, i) == 0)
                        trimap(y, x) = 1;
                    else if (trimap(j, i) == 255)
                        trimap(y, x) = 254;
                }
            }
        }
    }        

    for (int x = 0; x < trimap.cols; ++x) {
        for (int y = 0; y < trimap.rows; ++y) {
            if (trimap(y, x) == 1)
                trimap(y, x) = 0;
            else if (trimap(y, x) == 254)
                trimap(y, x) = 255;

        }
    }
        
}

void expansionOfKnownRegions(cv::InputArray _img, cv::InputOutputArray _trimap, int niter)
{
    cv::Mat img = _img.getMat();
    cv::Mat &trimap = _trimap.getMatRef();

    if (img.empty())
        CV_Error(CV_StsBadArg, "image is empty");
    if (img.type() != CV_8UC3)
        CV_Error(CV_StsBadArg, "image mush have CV_8UC3 type");

    if (trimap.empty())
        CV_Error(CV_StsBadArg, "trimap is empty");
    if (trimap.type() != CV_8UC1)
        CV_Error(CV_StsBadArg, "trimap mush have CV_8UC1 type");

    if (img.size() != trimap.size())
        CV_Error(CV_StsBadArg, "image and trimap mush have same size");

    for (int i = 0; i < niter; ++i)
        expansionOfKnownRegionsHelper(img, trimap, i + 1, niter - i);
    //erodeFB(trimap, 2);
}


static void globalMattingHelper(cv::Mat _image, cv::Mat _trimap, cv::Mat &_foreground, cv::Mat &_alpha, cv::Mat &_conf)
{
    const cv::Mat_<cv::Vec3b> &image = (const cv::Mat_<cv::Vec3b>&)_image;
    const cv::Mat_<uchar> &trimap = (const cv::Mat_<uchar>&)_trimap;

    std::vector<cv::Point> foregroundBoundary = findBoundaryPixels(trimap, 255, 128);
    std::vector<cv::Point> backgroundBoundary = findBoundaryPixels(trimap, 0, 128);

    int n = (int)(foregroundBoundary.size() + backgroundBoundary.size());
    for (int i = 0; i < n; ++i)
    {
        int x = rand() % trimap.cols;
        int y = rand() % trimap.rows;

        if (trimap(y, x) == 0)
            backgroundBoundary.push_back(cv::Point(x, y));
        else if (trimap(y, x) == 255)
            foregroundBoundary.push_back(cv::Point(x, y));
    }

    std::sort(foregroundBoundary.begin(), foregroundBoundary.end(), IntensityComp(image));
    std::sort(backgroundBoundary.begin(), backgroundBoundary.end(), IntensityComp(image));

    std::vector<std::vector<Sample> > samples;
    calculateAlphaPatchMatch(image, trimap, foregroundBoundary, backgroundBoundary, samples);

    _foreground.create(image.size(), CV_8UC3);
    _alpha.create(image.size(), CV_8UC1);
    _conf.create(image.size(), CV_8UC1);

    cv::Mat_<cv::Vec3b> &foreground = (cv::Mat_<cv::Vec3b>&)_foreground;
    cv::Mat_<uchar> &alpha = (cv::Mat_<uchar>&)_alpha;
    cv::Mat_<uchar> &conf = (cv::Mat_<uchar>&)_conf;

    for (int y = 0; y < alpha.rows; ++y)
        for (int x = 0; x < alpha.cols; ++x)
        {
            switch (trimap(y, x))
            {
                case 0:
                    alpha(y, x) = 0;
                    conf(y, x) = 255;
                    foreground(y, x) = 0;
                    break;
                case 128:
                {
                    alpha(y, x) = 255 * samples[y][x].alpha;
                    conf(y, x) = 255 * exp(-samples[y][x].cost / 6);
                    //cv::Point p = foregroundBoundary[samples[y][x].foreground_index];
                    //foreground(y, x) = image(p.y, p.x);
                    foreground(y, x) = image(y, x) * samples[y][x].alpha;
                    break;
                }
                case 255:
                    alpha(y, x) = 255;
                    conf(y, x) = 255;
                    foreground(y, x) = image(y, x);
                    break;
            }
        }
}

void globalMatting(cv::InputArray _image, cv::InputArray _trimap, cv::OutputArray _foreground, cv::OutputArray _alpha, cv::OutputArray _conf)
{
    cv::Mat image = _image.getMat();
    cv::Mat trimap = _trimap.getMat();

    if (image.empty())
        CV_Error(CV_StsBadArg, "image is empty");
    if (image.type() != CV_8UC3)
        CV_Error(CV_StsBadArg, "image mush have CV_8UC3 type");

    if (trimap.empty())
        CV_Error(CV_StsBadArg, "trimap is empty");
    if (trimap.type() != CV_8UC1)
        CV_Error(CV_StsBadArg, "trimap mush have CV_8UC1 type");

    if (image.size() != trimap.size())
        CV_Error(CV_StsBadArg, "image and trimap mush have same size");

    cv::Mat &foreground = _foreground.getMatRef();
    cv::Mat &alpha = _alpha.getMatRef();
    cv::Mat tempConf;

    globalMattingHelper(image, trimap, foreground, alpha, tempConf);

    if(_conf.needed())
        tempConf.copyTo(_conf);
}
