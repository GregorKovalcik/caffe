#include "caffe/caffe_feature_extractor_lib/caffe_feature_extractor_lib.hpp"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

namespace caffe {

FeatureExtractor::FeatureExtractor(
    const string& modelFile,
    const string& trainedFile,
    const string& meanFile)
{
    LoadNetwork(modelFile, trainedFile);
    LoadMean(meanFile);
}

FeatureExtractor::~FeatureExtractor()
{
}


Mat FeatureExtractor::ExtractFromImage(const Mat& image, const string& blobName)
{
    if (image.empty())
    {
        LOG(ERROR) << "Unable to decode image.";
        return Mat();
    }
    else
    {
        vector<Mat> inputChannels;
        WrapInputLayer(&inputChannels);
        Preprocess(image, &inputChannels);
        mNet->Forward();

        CHECK(mNet->has_blob(blobName))
            << "Unknown feature blob name: " << blobName;
        boost::shared_ptr<Blob<float> > blob = mNet->blob_by_name(blobName);

        Mat featureImage(1, blob->num() * blob->channels() * blob->width() * blob->height(),
            CV_32FC1, blob->data()->mutable_cpu_data());

        // when returning as a vector:
        //const float* begin = (float*)blob->data()->cpu_data();
        //const float* end = begin + (blob->num() * blob->channels() * blob->width() * blob->height());
        //vector<float> feature(begin, end);

        return featureImage;
    }
}


void FeatureExtractor::ExtractFromStream(
    const string& inputFolder,
    istream& inputStream,
    const string& outputPath,
    const string& blobNames,
    bool enableTextOutput,
    bool enableImageOutput,
    bool enableXmlOutput,
    int imageMaxHeight,
    int logEveryNth)
{
    mIsTextOutputEnabled = enableTextOutput;
    mIsImageOutputEnabled = enableImageOutput;
    mIsXmlOutputEnabled = enableXmlOutput;
    mImageMaxHeight = imageMaxHeight;
    mLogEveryNth = logEveryNth;

    LoadOutputModules(blobNames, outputPath);

    fs::path folder(inputFolder);
    fs::directory_iterator endIterator;

    CHECK(fs::exists(folder))
        << "Path not found: " << folder;

    CHECK(fs::is_directory(folder))
        << "Path is not directory: " << folder;

    double timeStart, timeElapsed;
    timeStart = (double)getTickCount();

    Mat image;
    int processedCount = 0;
    for (string file; getline(inputStream, file);)
    {
        string path = (folder / file).string();
        image = imread(path);
        if (image.empty())
        {
            LOG(ERROR) << "Unable to decode image " << path;
        }
        else
        {
            Process(image);

            for (int iModule = 0; iModule < ((int)mOutputModules.size()); iModule++)
            {
                mOutputModules[iModule]->WriteFeatureFor(path);
            }

            LOG_EVERY_N(INFO, mLogEveryNth) << google::COUNTER << " processed.";
            processedCount++;
        }
    }

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Feature extraction finished in " << timeElapsed << " seconds.\n"
        << processedCount << " files processed.\n"
        << "Average processing speed is " << processedCount / timeElapsed << " features per second.";

    CloseOutputModules();
}


void FeatureExtractor::ExtractFromFileList(
    const string& inputFolder,
    const string& inputFile,
    const string& outputPath,
    const string& blobNames,
    bool enableTextOutput,
    bool enableImageOutput,
    bool enableXmlOutput,
    int imageMaxHeight,
    int logEveryNth)
{
    ifstream inputStream;
    inputStream.open(inputFile.c_str());
    if (inputStream.is_open())
    {
        ExtractFromStream(
            inputFolder,
            inputStream,
            outputPath,
            blobNames,
            enableTextOutput,
            enableImageOutput,
            enableXmlOutput,
            imageMaxHeight,
            logEveryNth);
    }
    else
    {
        LOG(FATAL) <<
            "Error opening file: " << inputFile;
    }
}



void FeatureExtractor::ExtractFromFileOrFolder(
    const string& inputFileOrFolder,
    const string& outputPath,
    const string& blobNames,
    bool enableTextOutput,
    bool enableImageOutput,
    bool enableXmlOutput,
    int imageMaxHeight,
    int logEveryNth)
{
    mIsTextOutputEnabled = enableTextOutput;
    mIsImageOutputEnabled = enableImageOutput;
    mIsXmlOutputEnabled = enableXmlOutput;
    mImageMaxHeight = imageMaxHeight;
    mLogEveryNth = logEveryNth;

    LoadOutputModules(blobNames, outputPath);

    double timeStart, timeElapsed;
    LOG(INFO) << "Loading input directory...";
    timeStart = (double)getTickCount();

    fs::path inputPath(inputFileOrFolder);
    fs::directory_iterator endIterator;

    CHECK(fs::exists(inputPath))
        << "Path not found: " << inputPath;

    CHECK(fs::is_directory(inputPath) || fs::is_regular_file(inputPath))
        << "Path is not directory, nor a regular file: " << inputPath;

    Mat image;
    int processedCount = 0;
    if (fs::is_directory(inputPath))
    {
        for (fs::directory_iterator directoryIterator(inputPath); directoryIterator != endIterator; directoryIterator++)
        {
            if (fs::is_regular_file(directoryIterator->status()))
            {
                const string& file = directoryIterator->path().string();
                image = imread(file);
                if (image.empty())
                {
                    LOG(ERROR) << "Unable to decode image " << file;
                }
                else
                {
                    Process(image);

                    for (int iModule = 0; iModule < ((int)mOutputModules.size()); iModule++)
                    {
                        mOutputModules[iModule]->WriteFeatureFor(file);
                    }

                    LOG_EVERY_N(INFO, mLogEveryNth) << google::COUNTER << " processed.";
                    processedCount++;
                }
            }
        }
    }
    else if (fs::is_regular_file(inputPath))
    {
        const string& file = inputPath.string();
        image = imread(file);
        if (image.empty())
        {
            LOG(ERROR) << "Unable to decode image " << file;
        }
        else
        {
            Process(image);

            for (int iModule = 0; iModule < ((int)mOutputModules.size()); iModule++)
            {
                mOutputModules[iModule]->WriteFeatureFor(file);
            }

            LOG_EVERY_N(INFO, mLogEveryNth) << google::COUNTER << " processed.";
            processedCount++;
        }
    }

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Feature extraction finished in " << timeElapsed << " seconds.\n"
        << processedCount << " files processed.\n"
        << "Average processing speed is " << processedCount / timeElapsed << " features per second.";

    CloseOutputModules();
}


void FeatureExtractor::LoadNetwork(const string& modelFile, const string& trainedFile)
{
    double timeStart, timeElapsed;

    LOG(INFO) << "Loading network file...";
    timeStart = (double)getTickCount();

    mNet.reset(new Net<float>(modelFile, TEST));
    mNet->CopyTrainedLayersFrom(trainedFile);

    CHECK_EQ(mNet->num_inputs(), 1)
        << "Network should have exactly one input.";
    CHECK_EQ(mNet->num_outputs(), 1)
        << "Network should have exactly one output.";

    Blob<float>* inputLayer = mNet->input_blobs()[0];
    mNumberOfChannels = inputLayer->channels();
    CHECK(mNumberOfChannels == 3 || mNumberOfChannels == 1)
        << "Input layer should have 1 or 3 channels.";
    mInputGeometry = Size(inputLayer->width(), inputLayer->height());

    inputLayer->Reshape(1, mNumberOfChannels, mInputGeometry.height, mInputGeometry.width);
    /* Forward dimension change to all layers. */
    mNet->Reshape();

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Network file loaded in " << timeElapsed << " seconds.";
}


void FeatureExtractor::WrapInputLayer(vector<Mat>* inputChannels)
{
    Blob<float>* inputLayer = mNet->input_blobs()[0];

    int width = inputLayer->width();
    int height = inputLayer->height();
    float* inputData = inputLayer->mutable_cpu_data();
    for (int i = 0; i < inputLayer->channels(); ++i)
    {
        Mat channel(height, width, CV_32FC1, inputData);
        inputChannels->push_back(channel);
        inputData += width * height;
    }
}


void FeatureExtractor::LoadMean(const string& meanFile)
{
    double timeStart, timeElapsed;

    LOG(INFO) << "Loading mean file...";
    timeStart = (double)getTickCount();

    BlobProto blobProto;
    ReadProtoFromBinaryFileOrDie(meanFile.c_str(), &blobProto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> meanBlob;
    meanBlob.FromProto(blobProto);
    CHECK_EQ(meanBlob.channels(), mNumberOfChannels)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<Mat> channels;
    float* data = meanBlob.mutable_cpu_data();
    for (int i = 0; i < mNumberOfChannels; i++)
    {
        /* Extract an individual channel. */
        Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += meanBlob.height() * meanBlob.width();
    }

    /* Merge the separate channels into a single image. */
    merge(channels, mMean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    Scalar channelMean = cv::mean(mMean);
    mMean = Mat(mInputGeometry, mMean.type(), channelMean);

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Mean file loaded in " << timeElapsed << " seconds.";
}


void FeatureExtractor::LoadOutputModules(const string& blobNames, const string& outputPath)
{
    double timeStart, timeElapsed;

    LOG(INFO) << "Loading blob names...";
    timeStart = (double)getTickCount();

    vector<std::string> blobNamesSeparated;
    split(blobNamesSeparated, blobNames, boost::is_any_of(","));

    size_t numberOfFeatures = blobNamesSeparated.size();
    for (size_t i = 0; i < numberOfFeatures; i++)
    {
        mOutputModules.push_back(boost::make_shared<OutputModule>(mNet, fs::path(outputPath), blobNamesSeparated[i],
            mIsTextOutputEnabled, mIsImageOutputEnabled, mIsXmlOutputEnabled, mImageMaxHeight));
    }

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Blob names loaded in " << timeElapsed << " seconds. Loaded names are: " << blobNames;
}


void FeatureExtractor::Preprocess(const Mat& image, vector<Mat>* inputChannels)
{
    /* Convert the input image to the input image format of the network. */
    Mat sample;
    if (image.channels() == 3 && mNumberOfChannels == 1)
    {
        cvtColor(image, sample, CV_BGR2GRAY);
    }
    else if (image.channels() == 4 && mNumberOfChannels == 1)
    {
        cvtColor(image, sample, CV_BGRA2GRAY);
    }
    else if (image.channels() == 4 && mNumberOfChannels == 3)
    {
        cvtColor(image, sample, CV_BGRA2BGR);
    }
    else if (image.channels() == 1 && mNumberOfChannels == 3)
    {
        cvtColor(image, sample, CV_GRAY2BGR);
    }
    else
    {
        sample = image;
    }

    Mat sampleResized;
    if (sample.size() != mInputGeometry)
    {
        resize(sample, sampleResized, mInputGeometry);
    }
    else
    {
        sampleResized = sample;
    }

    Mat sampleFloat;
    if (mNumberOfChannels == 3)
    {
        sampleResized.convertTo(sampleFloat, CV_32FC3);
    }
    else
    {
        sampleResized.convertTo(sampleFloat, CV_32FC1);
    }

    Mat sampleNormalized;
    subtract(sampleFloat, mMean, sampleNormalized);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the Mat
    * objects in input_channels. */
    split(sampleNormalized, *inputChannels);

    CHECK(reinterpret_cast<float*>(inputChannels->at(0).data) == mNet->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}


void FeatureExtractor::Process(const Mat& image)
{
    vector<Mat> inputChannels;
    WrapInputLayer(&inputChannels);
    Preprocess(image, &inputChannels);
    mNet->Forward();
}


void FeatureExtractor::CloseOutputModules()
{
    double timeStart, timeElapsed;
    // close output modules
    LOG(INFO) << "Closing output modules...";
    timeStart = (double)getTickCount();
    for (int iModule = 0; iModule < ((int)mOutputModules.size()); iModule++)
    {
        mOutputModules[iModule]->Close();
    }
    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Output modules closed in " << timeElapsed << " seconds.\n";
}

}  // namespace caffe
