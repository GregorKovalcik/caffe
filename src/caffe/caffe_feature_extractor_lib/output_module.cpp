#include "caffe/caffe_feature_extractor_lib/output_module.hpp"

using namespace std;
using namespace cv;
using namespace caffe;
namespace fs = boost::filesystem;

namespace caffe {

OutputModule::OutputModule(
    const boost::shared_ptr<Net<float> >& net,
    const fs::path& outputPath,
    const string& blobName,
    bool enableTextOutput,
    bool enableImageOutput,
    bool enableXmlOutput,
    int numberOfImageRows)
    : mIsTextOutputEnabled(enableTextOutput),
    mIsImageOutputEnabled(enableImageOutput),
    mIsXMLOutputEnabled(enableXmlOutput),
    mImageMaxHeight(numberOfImageRows),
	mFileCounter(0),
	mIsClosed(false)
{
    // check and prepare blob
    CHECK(net->has_blob(blobName))
        << "Unknown feature blob name: " << blobName;
    this->mBlob = net->blob_by_name(blobName);
    string blobNameSafe = ReplaceIllegalCharacters(blobName);

    // output path is a directory
    if (fs::is_directory(outputPath))
    {
        CHECK(fs::exists(outputPath))
            << "Output directory does not exist!";

        mOutputPath = (outputPath / fs::path(blobNameSafe + ".txt")).string();
        mOutputPathStripped = (outputPath / fs::path(blobNameSafe)).string();
    }
    else // output path is a regular file
    {
        fs::path directory = outputPath.parent_path();
        fs::path filename = outputPath.filename().replace_extension("");
        fs::path extension = outputPath.extension();

        mOutputPath = (directory / fs::path(filename.string() + "_" + blobNameSafe + ".txt")).string();
        mOutputPathStripped = (directory / fs::path(filename.string() + "_" + blobNameSafe)).string();
    }

    // set image height
    mImageMaxHeight = numberOfImageRows;

    // open output stream
    if (mIsTextOutputEnabled)
    {
        mOutputStream.open(mOutputPath.c_str());
        CHECK(mOutputStream.is_open())
            << "Error opening output stream for file \"" << mOutputPath << "\"";
    }

    LOG(INFO)
        << "Output module for blob " << blobName << " created. Output will have "
        << mBlob->num() * mBlob->channels() * mBlob->width() * mBlob->height() << " columns.";
}


void OutputModule::Close()
{
    if (!mIsClosed)
    {
        if (mIsTextOutputEnabled)
        {
            mOutputStream.close();
            LOG(INFO) << "Text output file closed: " << mOutputPath;
        }

        if (mFileCounter == 0)	// do not add file number if only one file is created
        {
            if (mIsImageOutputEnabled)
            {
                string outputPath = mOutputPathStripped + ".png";
                string outputPathHC = mOutputPathStripped + "_hc" + ".png";
                string outputPathBR = mOutputPathStripped + "_br" + ".png";
                string outputPathBRHC = mOutputPathStripped + "_brhc" + ".png";

                SaveAndClearImage(mOutputImage, outputPath);
                SaveAndClearImage(mOutputImageContrast, outputPathHC);
                SaveAndClearImage(mOutputImageBlueRed, outputPathBR);
                SaveAndClearImage(mOutputImageBlueRedContrast, outputPathBRHC);
            }

            if (mIsXMLOutputEnabled)
            {
                string outputPath = mOutputPathStripped + ".xml";
                SaveAndClearXML(mOutputXML, outputPath);
            }
        }
        else  // add file counter when the images are split into multiple smaller files
        {
        	std::stringstream fileCounterSStream;
        	fileCounterSStream << mFileCounter;

        	if (mIsImageOutputEnabled)
            {
                string outputPath = mOutputPathStripped + "_" + fileCounterSStream.str() + ".png";
                string outputPathHC = mOutputPathStripped + "_hc_" + fileCounterSStream.str() + ".png";
                string outputPathBR = mOutputPathStripped + "_br_" + fileCounterSStream.str() + ".png";
                string outputPathBRHC = mOutputPathStripped + "_brhc_" + fileCounterSStream.str() + ".png";

                SaveAndClearImage(mOutputImage, outputPath);
                SaveAndClearImage(mOutputImageContrast, outputPathHC);
                SaveAndClearImage(mOutputImageBlueRed, outputPathBR);
                SaveAndClearImage(mOutputImageBlueRedContrast, outputPathBRHC);
            }

            if (mIsXMLOutputEnabled)
            {
                string outputPath = mOutputPathStripped + "_" + fileCounterSStream.str() + ".xml";
                SaveAndClearXML(mOutputXML, outputPath);
            }
        }

        mIsClosed = true;
    }
    else
    {
        LOG(WARNING) << "Attempted to close already closed output module for: " << mOutputPath;
    }
}


void OutputModule::WriteText(const string& inputFilename)
{
    if (mIsTextOutputEnabled)
    {
        const float* begin = (float*)mBlob->data()->cpu_data();
        const float* end = begin + mBlob->num() * mBlob->channels() * mBlob->width() * mBlob->height();
        vector<float> feature(begin, end);

        mOutputStream << inputFilename << ":";
        for (size_t i = 0; i < feature.size(); ++i)
        {
            mOutputStream << feature[i] << ";";
        }
        mOutputStream << endl;
    }
}


void OutputModule::WriteImage(const string& inputFilename)
{
    if (mIsImageOutputEnabled || mIsXMLOutputEnabled)
    {
        // wrap blob data using cv::Mat
        const Mat featureImage(1, mBlob->num() * mBlob->channels() * mBlob->width() * mBlob->height(),
            CV_32FC1, mBlob->data()->mutable_cpu_data());

        // split images if needed
        if (mImageMaxHeight > 0 && mOutputImage.rows == mImageMaxHeight)
        {
        	std::stringstream fileCounterSStream;
        	fileCounterSStream << mFileCounter;

            if (mIsImageOutputEnabled)
            {
                string outputPath = mOutputPathStripped + "_" + fileCounterSStream.str() + ".png";
                string outputPathHC = mOutputPathStripped + "_hc_" + fileCounterSStream.str() + ".png";
                string outputPathBR = mOutputPathStripped + "_br_" + fileCounterSStream.str() + ".png";
                string outputPathBRHC = mOutputPathStripped + "_brhc_" + fileCounterSStream.str() + ".png";

                SaveAndClearImage(mOutputImage, outputPath);
                SaveAndClearImage(mOutputImageContrast, outputPathHC);
                SaveAndClearImage(mOutputImageBlueRed, outputPathBR);
                SaveAndClearImage(mOutputImageBlueRedContrast, outputPathBRHC);
            }

            if (mIsXMLOutputEnabled)
            {
                string outputPath = mOutputPathStripped + "_" + fileCounterSStream.str() + ".xml";
                SaveAndClearXML(mOutputXML, outputPath);
            }
            mFileCounter++;
        }

        // append feature into image buffers
        NormalizeAndSaveFeature(featureImage);
    }
}


void OutputModule::NormalizeAndSaveFeature(const Mat& featureImage)
{
    double min, max;
    minMaxLoc(featureImage, &min, &max);
    min = (min > 0) ? 0 : min;

    // normalized image to range 0..1
    Mat normalized;
    Mat normalizedHighContrast;
    featureImage.copyTo(normalized);
    normalized += abs(min);
    normalized *= 1.0 / abs(max - min);
    normalized.copyTo(normalizedHighContrast);
    normalized.convertTo(normalized, CV_8U, 255);
    // normalized image to range 0..1 high contrast
    normalizedHighContrast = (normalizedHighContrast != 0);


    // normalized image to range -1..+1
    Mat normalizedBoth;
    featureImage.copyTo(normalizedBoth);
    normalizedBoth.convertTo(normalizedBoth, CV_32FC3);
    if (abs(min) < max)
    {
        normalizedBoth *= 1.0 / max;
    }
    else
    {
        normalizedBoth *= 1.0 / abs(min);
    }
    Mat positiveMask = (normalizedBoth > 0);
    Mat negativeMask = (normalizedBoth < 0);
    Mat positive, negative;
    normalizedBoth.copyTo(positive, positiveMask);
    normalizedBoth.copyTo(negative, negativeMask);
    negative *= -1;
    vector<Mat> channels;
    Mat zeros = Mat::zeros(normalizedBoth.rows, normalizedBoth.cols, CV_32FC1);
    channels.push_back(negative);
    channels.push_back(zeros);
    channels.push_back(positive);
    Mat normalizedBlueRed;
    merge(channels, normalizedBlueRed);
    normalizedBlueRed.convertTo(normalizedBlueRed, CV_8U, 255);


    // normalized image to range -1..+1 high contrast
    Mat normalizedBlueRedHighContrast;
    channels.resize(0);
    zeros = Mat::zeros(normalizedBoth.rows, normalizedBoth.cols, CV_8UC1);
    channels.push_back(negativeMask);
    channels.push_back(zeros);
    channels.push_back(positiveMask);
    merge(channels, normalizedBlueRedHighContrast);


    // raw float values saved as XML
    mOutputXML.push_back(featureImage);
    mOutputImage.push_back(normalized);
    mOutputImageContrast.push_back(normalizedHighContrast);
    mOutputImageBlueRed.push_back(normalizedBlueRed);
    mOutputImageBlueRedContrast.push_back(normalizedBlueRedHighContrast);
}

}  // namespace caffe
