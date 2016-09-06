#ifndef CAFFE_MAP_EVALUATOR_HPP
#define CAFFE_MAP_EVALUATOR_HPP

#define XML_MAT_IDENTIFIER "caffe_features"

#include <string>
#include <fstream>

#include <glog/logging.h>
#include <opencv2/core/core.hpp>

#include "vector_distance_functions.hpp"
#include "image_info.hpp"

/**
* @brief Performs multiple queries using caffe features and evaluates their mean average precision.
*/
class MapEvaluator
{
public:

    /**
    * @brief Constructor loading earlier extracted caffe features and an annotation file where
    *       the image classes and query images are described.
    * @param featuresFile cv::Mat of extracted image features (one per line) serialized into XML file
    *       using cv::FileStorage. The cv::Mat is identified by XML_MAT_IDENTIFIER
    * @param annotationFile A CSV file with class anq query annotation of each input feature/image.
    *       Syntax: image ID;class ID;is query;class count[;other columns are skipped]
    *       File has be sorted by the first column and no line can be missing
    *       (simply, line number has to be the same as image ID).
    */
    MapEvaluator(
        const std::string& featuresFile,
        const std::string& annotationFile)
        : mExcludeQueryFromDB(true),
        mTopK(0),
        mDistanceFunction(L2),
        mDistanceFunctionParameter(0)
    {
        LoadFeatures(featuresFile);
        ParseAnnotationFile(annotationFile);

        CHECK_EQ(mFeatures.rows, mImageInfos.size())
            << "Number of loaded features and number of images in the annotation file are not equal!";
    }

    /**
    * @brief Launches the evaluation on preloaded data.
    */
    double Evaluate();

    /**
    * @brief Distance function for distance measurement between two features.
    * @param distanceFunction Distance function selector.
    */
    void SetDistanceFunction(DistanceFunction distanceFunction)
    {
        mDistanceFunction = distanceFunction;
    }

    /**
    * @brief Set optional parameter of distance measurement.
    * @param parameter Parameter of the distance function.
    */
    void SetDistanceFunctionParameter(double parameter)
    {
        mDistanceFunctionParameter = parameter;
    }

    /**
    * @brief Query only for top K results.
    * @param topK Number of results to use in MAP evaluation.
    */
    void SetTopK(int topK)
    {
        mTopK = topK;
    }

    /**
    * @brief Whether is the query feature excluded from results.
    * @param topK Number of results to use in MAP evaluation.
    */
    void SetExcludeQueryFromResults(bool excludeQueryFromResults)
    {
        mExcludeQueryFromDB = excludeQueryFromResults;
    }

    // TODO: get MAP graph

private:
    cv::Mat mFeatures;
    std::vector<ImageInfo> mImageInfos;
    std::vector<ImageInfo> mQueryInfos;

    bool mExcludeQueryFromDB;
    int mTopK;
    DistanceFunction mDistanceFunction;
    double mDistanceFunctionParameter;

    /**
    * @brief Load features stored as cv::Mat in a XML file, under identifier XML_MAT_IDENTIFIER
    * @param featuresFile Path to the input XML file.
    */
    void LoadFeatures(const std::string& featuresFile);

    /**
    * @brief Load annotation for each image feature.
    * @param annotationFile Path to the input annotation file.
    */
    void ParseAnnotationFile(const std::string& annotationFile);

    /**
    * @brief Average precision evaluation for one query.
    * @param query Query image info.
    * @param sortedImages Vector of sorted results.
    * @param precisionRecallValues Output vector, where P-R values will be stored (used for drawing a P-R graph).
    */
    double EvaluateAveragePrecision(
        const ImageInfo& query,
        const std::vector<ImageInfo>& sortedImages,
        std::vector<std::pair<double, double>>& precisionRecallValues);

    /**
    * @brief Compute distance between two features.
    * @param vectorA First feature vector.
    * @param vectorB Second feature vector.
    */
    double ComputeFeatureDistance(const cv::Mat& vectorA, const cv::Mat& vectorB);
};






#endif
