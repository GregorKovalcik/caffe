#include <iostream>
#include <boost/algorithm/string.hpp>
#include "map_evaluator.hpp"

using namespace std;


string featuresFile;
string annotationFile;

DistanceFunction distanceFunction = L2Squared;
double distanceFunctionParameter = 0;
int topK = 0;
bool excludeQueryFromResults = false;


string ParseArgumentValueForOption(const string& option, char * const * const& iterator, char * const * const& end)
{
    CHECK(iterator != end) << "Error parsing option \"" << option << "\"";
    return *iterator;
}


void PrintHelpMessage()
{
    cerr <<
        "Caffe feature MAP evaluator.\nAuthor: Gregor Kovalcik\n\n"
        "Usage:\n"
        "    map_evaluator [options] features.xml annotation.csv\n\n"

        "features.xml\n"
        "    - features generated using caffe_feature_extractor, saved in XML format\n"
        "    using cv::FileStorage with identifier "XML_MAT_IDENTIFIER".\n"
        "annotation.csv\n"
        "    - annotation file, storing info about input features in following format:\n"
        "    <image ID (has to be same as line ID)>;<class ID>;<is query>;<class count>\n"
        "    One feature per line.\n"
        "    Image ID is in fact ignored and a line number is used as ID instead.\n\n"

        "Options:\n"
        "-h, --help\n"
        "    Shows the help screen.\n\n"

        "-d, --distance-function <L2|L2Sqr|L1|Linfinity|Cosine|Hamming|MaxDim>\n"
        "    Distance function selector. Implemented functions are:\n"
        "    L1, L2, Linfinity  - standard Lp distances.\n"
        "    L2Sqr              - L2 squared.\n"
        "    Cosine             - Cosine similarity, or (1 - <cosine distance>).\n"
        "    Hamming            - Hamming distance of vectors with nonzero values\n"
        "                         set to 1.\n"
        "    MaxDim             - Distance of two values at the index of the\n"
        "                         highest value in the query vector.\n\n"

        "-k, --top-k <value>\n"
        "    Evaluate top K query results only.\n\n"

        "-e, --exclude-query-from-results\n"
        "    Exclude query feature from the retrieved result set.\n\n"

        << endl;
}


pair<DistanceFunction, double> ParseDistanceFunction(string& distance)
{
    pair<DistanceFunction, double> result;
    result.second = 0;

    boost::algorithm::to_lower(distance);

    if (distance == "l2" || distance == "lp2")
    {
        result.first = DistanceFunction::L2;
    }
    else if (distance == "l2sqr")
    {
        result.first = DistanceFunction::L2Squared;
    }
    else if (distance == "l1" || distance == "lp1")
    {
        result.first = DistanceFunction::L1;
    }
    else if (distance == "linfinity")
    {
        result.first = DistanceFunction::Infinity;
    }
    else if (distance == "cosine")
    {
        result.first = DistanceFunction::Cosine;
    }
    else if (distance == "hamming")
    {
        result.first = DistanceFunction::Hamming;
    }
    else if (distance == "maxdim" || distance == "maximal_dimension_difference")
    {
        result.first = DistanceFunction::MaximalDimensionDifference;
    }
    else
    {
        LOG(FATAL) << "Unknown distance function: " << distance;
    }

    return result;
}


bool ParseProgramArguments(int argc, char * const * const argv)
{
    char * const * const begin = argv;
    char * const * const end = argv + argc;
    char * const * iterator = begin + 1;	// skip the first argument

    // no arguments
    if (iterator == end)
    {
        PrintHelpMessage();
        return false;
    }

    // initialize logging
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = 0;


    // parse optional arguments
    while (iterator != end && **iterator == '-')
    {
        string option = *iterator;
        string value = "";

        if (!option.compare("-h") || !option.compare("--help"))						// help
        {
            PrintHelpMessage();
            return false;
        }
        else if (!option.compare("-d") || !option.compare("--distance-function"))	// distance function
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            pair<DistanceFunction, double> result = ParseDistanceFunction(value);
            distanceFunction = result.first;
            distanceFunctionParameter = result.second;
        }
        else if (!option.compare("-k") || !option.compare("--top-k"))				// top K
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            try
            {
                topK = std::stoi(value);
            }
            catch (const invalid_argument& exception)
            {
                cerr << "Error parsing top K:" << value << endl;
                return false;
            }
        }
        else if (!option.compare("-e") || !option.compare("--exclude-query-from-results"))	// distance function
        {
            excludeQueryFromResults = true;
        }
        else
        {
            LOG(WARNING) << "Unknown option: " << option;
        }

        iterator++;
    }

    // parse mandatory arguments
    featuresFile = ParseArgumentValueForOption("featuresFile", iterator++, end);
    annotationFile = ParseArgumentValueForOption("annotationFile", iterator++, end);

    return true;
}


int main(int argc, char** argv)
{
    // check argument count
    if (ParseProgramArguments(argc, argv))
    {
        MapEvaluator mapEvaluator(featuresFile, annotationFile);
        mapEvaluator.SetDistanceFunction(distanceFunction);
        mapEvaluator.SetDistanceFunctionParameter(distanceFunctionParameter);
        mapEvaluator.SetTopK(topK);
        mapEvaluator.SetExcludeQueryFromResults(excludeQueryFromResults);

        double map = mapEvaluator.Evaluate();
        cout << "Mean average precision: " << map << endl;
    }
    else
    {
        return 1;
    }

    return 0;
}
