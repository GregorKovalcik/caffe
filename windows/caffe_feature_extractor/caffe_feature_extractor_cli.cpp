#include "../caffe_feature_extractor_lib/caffe_feature_extractor_lib.hpp"

using namespace std;


void PrintHelpMessage()
{
    std::cerr <<
        "Caffe feature extractor\nAuthor: Gregor Kovalcik\n\n"
        "This program takes in a trained network and an input data layer name, and then\n"
        "extracts features of the input data produced by the net.\n\n"
        "Usage:\n"
        "    caffe_feature_extractor_cli [options] deploy.prototxt network.caffemodel\n"
        "    mean.binaryproto blob_name1[,name2,...] input_folder file_list\n"
        "    output_file_or_folder\n"
        "Note:\n"
        "    you can extract multiple features in one pass by specifying multiple\n"
        "    feature blob names separated by ','. The names cannot contain white space\n"
        "    characters.\n\n"

        "Options:\n"

        "-h, --help\n"
        "    Shows the help screen.\n\n"

#ifndef CPU_ONLY
        "-m <GPU|CPU>, --mode <GPU|CPU> (default: GPU)\n"
        "    Choose whether to compute features using GPU or CPU.\n\n"
#endif

        "-d, --disable-text-output\n"
        "    Disables the text file output (useful to generate image file output only).\n\n"

        "-i, --image-output\n"
        "    Enables the image output. Each row is the feature of one input image.\n"
        "    Number of columns is equal to the extracted blob size.\n"
        "    This generates four PNG image files per extracted blob. The original\n"
        "    filename is preserved and the extension is replaced:\n"
        "    - (output_filename).png:\n"
        "        Grayscale image, where the feature values are normalized to range 0..1\n"
        "    - (output_filename)_hc.png:\n"
        "        High contrast version of the normalized image. Zero values are copied,\n"
        "        positive values are set to 1 (255 actually, because we are saving it\n"
        "        using 8bits per pixel).\n"
        "    - (output_filename)_br.png:\n"
        "        RGB image, where the feature values are normalized to range -1..+1.\n"
        "        Negative values are printed in blue color while positive values are\n"
        "        printed in red color.\n"
        "    - (output_filename)_brhc.png:\n"
        "        High contrast version of the previous image. Zero values are copied,\n"
        "        positive values are set to red color RGB(255, 0, 0), negative values\n"
        "        are set to blue color RGB(0, 0, 255).\n\n"

        "-r <(int) height>, --image-height <(int) height> (default: 0 - do not split)\n"
        "    Splits the image files if they are higher than the <(int) height>. Useful\n"
        "    when the generated images are too big to fit in the memory.\n\n"

        "-x, --xml-output\n"
        "    Enables XML output. Stores features as OpenCV CV_32FC1 Mat. Each row is\n"
        "    the feature of one input image. This Mat is then serialized using\n"
        "    cv::FileStorage with identifier \"caffe_features\".\n\n"

        "-l <(int) log_level>, --log-level <(int) log_level> (default: 0)\n"
        "    Log suppression level: messages logged at a lower level than this are.\n"
        "    suppressed. The numbers of severity levels INFO, WARNING, ERROR, and FATAL\n"
        "    are 0, 1, 2, and 3, respectively.\n\n"

        "-n <(int) log_level>, --log-every-nth <(int) log_level> \n"
        "    (default in GPU mode: 100, default in CPU mode: 10)\n"
        "    Logs every nth file processed."

        << std::endl;
}


string ParseArgumentValueForOption(const string& option, char * const * const& iterator, char * const * const& end)
{
    CHECK(iterator != end) << "Error parsing option \"" << option << "\"";
    return *iterator;
}


string modelFile;
string trainedFile;
string meanFile;
string blobNames;
string inputFolder;
string inputFileList;
string outputPath;

bool isTextOutputEnabled = true;
bool isImageOutputEnabled = false;
bool isXmlOutputEnabled = false;
int imageMaxHeight = 0;

#ifdef CPU_ONLY
int logEveryNth = 10;
#else
int logEveryNth = 100;
#endif


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

#ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif


    // parse optional arguments
    while (iterator != end && **iterator == '-')
    {
        string option = *iterator;
        string value = "";

        if (!option.compare("-h") || !option.compare("--help"))				// help
        {
            PrintHelpMessage();
            return false;
        }
#ifndef CPU_ONLY
        else if (!option.compare("-m") || !option.compare("--mode"))			// CPU/GPU mode
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            boost::algorithm::to_upper(value);
            if (value == "GPU")
            {
                caffe::Caffe::set_mode(caffe::Caffe::GPU);
            }
            else if (value == "CPU")
            {
                caffe::Caffe::set_mode(caffe::Caffe::CPU);
                logEveryNth = 10;
            }
            else
            {
                cerr << "Unknown mode: " << value << endl;
                return false;
            }
        }
#endif
        else if (!option.compare("-d") || !option.compare("--disable-text-output"))	// text output
        {
            isTextOutputEnabled = false;
        }
        else if (!option.compare("-i") || !option.compare("--image-output"))			// image output
        {
            isImageOutputEnabled = true;
        }
        else if (!option.compare("-r") || !option.compare("--image-height"))			// image height
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            try
            {
                imageMaxHeight = std::stoi(value);
            }
            catch (const invalid_argument&)
            {
                cerr << "Error parsing image height:" << value << endl;
                return false;
            }
        }
        else if (!option.compare("-x") || !option.compare("--xml-output"))			// XML output
        {
            isXmlOutputEnabled = true;
        }
        else if (!option.compare("-l") || !option.compare("--log-level"))			// log level
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            int logLevel;
            try
            {
                logLevel = std::stoi(value);
            }
            catch (const invalid_argument&)
            {
                cerr << "Error parsing log level!" << endl;
                return false;
            }
            FLAGS_minloglevel = logLevel;
        }
        else if (!option.compare("-n") || !option.compare("--log-every-nth"))		// log every nth
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            try
            {
                logEveryNth = std::stoi(value);
            }
            catch (const invalid_argument&)
            {
                cerr << "Error parsing \"log every n-th\":" << value << endl;
                return false;
            }
        }
        else
        {
            cerr << "Unknown option: " << option;
            return false;
        }

        iterator++;
    }

    // parse mandatory arguments
    modelFile = ParseArgumentValueForOption("modelFile", iterator++, end);
    trainedFile = ParseArgumentValueForOption("trainedFile", iterator++, end);
    meanFile = ParseArgumentValueForOption("meanFile", iterator++, end);
    blobNames = ParseArgumentValueForOption("blobNames", iterator++, end);
    inputFolder = ParseArgumentValueForOption("inputFolder", iterator++, end);
    inputFileList = ParseArgumentValueForOption("inputFileList", iterator++, end);
    outputPath = ParseArgumentValueForOption("outputPath", iterator++, end);

    return true;
}


int main(int argc, char** argv)
{
    if (ParseProgramArguments(argc, argv))
    {
        FeatureExtractor featureExtractor(modelFile, trainedFile, meanFile);

        featureExtractor.ExtractFromFileList(inputFolder, inputFileList, outputPath, blobNames,
            isTextOutputEnabled, isImageOutputEnabled, isXmlOutputEnabled, imageMaxHeight, logEveryNth);
    }
    else
    {
        return 1;
    }

    return 0;
}
