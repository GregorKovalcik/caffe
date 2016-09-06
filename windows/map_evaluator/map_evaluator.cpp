#include "map_evaluator.hpp"

using namespace std;
using namespace cv;

void MapEvaluator::LoadFeatures(const string& featuresFile)
{
    double timeStart, timeElapsed;
    timeStart = (double)getTickCount();
    LOG(INFO)
        << "Loading features from file: " << featuresFile;

    FileStorage storage(featuresFile, FileStorage::READ);
    storage[XML_MAT_IDENTIFIER] >> mFeatures;
    storage.release();

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO)
        << "Features loaded in " << timeElapsed << " seconds.";
}


void MapEvaluator::ParseAnnotationFile(const string& annotationFile)
{
    double timeStart, timeElapsed;
    timeStart = (double)getTickCount();
    LOG(INFO)
        << "Loading annotation file...";

    ifstream input(annotationFile);
    CHECK(input.is_open())
        << "Error opening annotation file: " << annotationFile;

    string line;
    char delimiter = ';';
    int id = 0;
    while (getline(input, line))
    {
        istringstream lineStream(line);
        string token;
        ImageInfo imageInfo;

        getline(lineStream, token, delimiter);			// ID (skip, we assume line number is the same as ID)
        imageInfo.id = id++;

        getline(lineStream, token, delimiter);			// class id
        int classId;
        try
        {
            imageInfo.classId = stoi(token.c_str());
        }
        catch (exception const& ex)
        {
            LOG(FATAL)
                << "Error parsing column \"class_id\" on line number " << id + 1;
        }


        getline(lineStream, token, delimiter);			// is query? 0/1
        int isQuery = 0;
        try
        {
            imageInfo.isQuery = stoi(token.c_str());
        }
        catch (exception const& ex)
        {
            LOG(FATAL)
                << "Error parsing column \"is_query\" on line number " << id + 1;
        }
        CHECK(isQuery == 0 || isQuery == 1)
            << "Invalid value in column \"is_query\", on line number " << id + 1 << ": 0 or 1 expected but " << isQuery << " received!";


        std::getline(lineStream, token, delimiter);		// class count
        try
        {
            imageInfo.classCount = stoi(token.c_str());
        }
        catch (exception const& ex)
        {
            LOG(FATAL)
                << "Error parsing column \"class_count\" on line number " << id + 1;
        }

        // other columns are ignored

        //std::getline(lineStream, token, delimiter);
        //imageInfo.fileName = token;

        //std::getline(lineStream, token, delimiter);
        //imageInfo.className = token;

        mImageInfos.push_back(imageInfo);
        if (imageInfo.isQuery)
        {
            mQueryInfos.push_back(imageInfo);
        }
    }
    input.close();

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO)
        << "Annotation file loaded in " << timeElapsed << " seconds.";
}


double MapEvaluator::Evaluate()
{
    double meanAveragePrecision = 0.0;
    vector<pair<double, double>> precisionRecallValues;

    for (int iQuery = 0; iQuery < mQueryInfos.size(); iQuery++)
    {
        int queryId = mQueryInfos[iQuery].id;
        Mat queryFeature = mFeatures(cv::Range(queryId, queryId + 1), cv::Range::all());

        // compute distances
        for (int iImage = 0; iImage < mImageInfos.size(); iImage++)
        {
            cv::Mat imageFeature = mFeatures(cv::Range(iImage, iImage + 1), cv::Range::all());
            mImageInfos[iImage].distance = ComputeFeatureDistance(queryFeature, imageFeature);
        }

        // sort results
        vector<ImageInfo> sortedResult(mImageInfos);
        sort(sortedResult.begin(), sortedResult.end());

        // evaluate average precision
        double averagePrecision = EvaluateAveragePrecision(mQueryInfos[iQuery], sortedResult, precisionRecallValues);
        LOG(INFO)
            << "Query " << iQuery + 1 << " of " << mQueryInfos.size() << ", average precision: " << averagePrecision;

        // TODO: use precision recall values to plot graph

        meanAveragePrecision += averagePrecision;
    }
    meanAveragePrecision /= mQueryInfos.size();

    return meanAveragePrecision;
}


double MapEvaluator::EvaluateAveragePrecision(
    const ImageInfo& query,
    const vector<ImageInfo>& sortedResult,
    vector<pair<double, double>>& precisionRecallValues)
{
    double oldRecall = 0.0;
    double oldPrecision = 1.0;
    double recall = 0.0;
    double precision = 0.0;

    double averagePrecision = 0.0;
    double matchCount = 0.0;
    double matches = query.classCount;
    int nResults = (int)sortedResult.size();

    bool wasQueryExcluded = false;

    if (mExcludeQueryFromDB)
    {
        matches--;
        nResults--;
    }

    if (mTopK > matches)
    {
        LOG(WARNING)
            << "Top K (" << mTopK << ") is higher than match count: " << matches;
    }

    if (mTopK > 0)
    {
        nResults = std::min(mTopK, nResults);
        matches = std::min((double)nResults, matches);
    }

    precisionRecallValues.clear();
    precisionRecallValues.reserve(nResults);

    for (int i = 0; i < sortedResult.size(); i++)
    {
        if (precisionRecallValues.size() == nResults)
        {
            break;
        }
        if (mExcludeQueryFromDB && sortedResult[i].id == query.id)
        {
            wasQueryExcluded = true;
            continue;
        }

        if (sortedResult[i].classId == query.classId)
        {
            matchCount++;
        }

        recall = matchCount / matches;
        if (wasQueryExcluded)
        {
            precision = matchCount / i;
        }
        else
        {
            precision = matchCount / (i + 1);
        }

        averagePrecision += (recall - oldRecall) * ((oldPrecision + precision) / 2);
        precisionRecallValues.push_back(make_pair(precision, recall));

        oldRecall = recall;
        oldPrecision = precision;
    }

    return averagePrecision;
}

double MapEvaluator::ComputeFeatureDistance(const cv::Mat& a, const cv::Mat& b)
{
    switch (mDistanceFunction)
    {
    case L2:
        return cv::norm(a, b, NORM_L2);
    case L2Squared:
        return cv::norm(a, b, NORM_L2SQR);
    case L1:
        return cv::norm(a, b, NORM_L1);
    case Infinity:
        return cv::norm(a, b, NORM_INF);
    case Cosine:
        return cosineDistance(a, b);
    case DistanceFunction::Hamming:
        return hamming(a, b);
    case MaximalDimensionDifference:
        return maximalDimensionDifference(a, b);
        // TODO: other norms

    default:
        LOG(FATAL)
            << "Distance function not implemented: " << mDistanceFunction;
        return -1;
    }
}
