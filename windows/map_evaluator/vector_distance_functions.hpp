#ifndef CAFFE_MAP_EVALUATOR_VECTOR_DISTANCE_FUNCTIONS_HPP
#define CAFFE_MAP_EVALUATOR_VECTOR_DISTANCE_FUNCTIONS_HPP

#include <opencv2/core/core.hpp>

/**
* @brief Currently implemented distance functions.
*/
enum DistanceFunction
{
    L2, L2Squared, L1, Infinity,
    Cosine, Hamming, MaximalDimensionDifference
};


/**
* @brief Cosine distance (angle) between two vectors.
* @param a The first vector
* @param b The second vector
*/
inline double cosineDistance(const cv::Mat a, const cv::Mat b)
{
    cv::Mat perElementMultiplication = a.mul(b);
    double dot = cv::sum(perElementMultiplication)[0];
    double cosAngle = dot / (norm(a) * norm(b));
    return 1 - cosAngle;
}

/**
* @brief Computes hamming distance between two vectors.
*       Nonzero values of the input vectors are set to 1 and zero values are preserved.
* @param a The first vector
* @param b The second vector
*/
inline double hamming(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat x, y, r;
    x = a != 0;
    y = b != 0;
    r = x != y;

    cv::Scalar d;
    d = cv::sum(r) / 255;
    return d[0];
}

/**
* @brief Finds the index with maximal value in vector A and then computes difference
*       between the values of both vectors on this index.
*       Used on features generated from a probability layer.
*       Index of the maximal value in A is ID of the most likely class of the image A.
*       We then compute difference in probability for this class ID with vector B.
*/
inline double maximalDimensionDifference(const cv::Mat& a, const cv::Mat& b)
{
    double min, max;
    cv::Point minIdx, maxIdx;
    cv::minMaxLoc(a, &min, &max, &minIdx, &maxIdx);

    return std::abs(a.at<float>(maxIdx.x) - b.at<float>(maxIdx.x));
}


#endif
