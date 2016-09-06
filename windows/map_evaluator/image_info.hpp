#ifndef CAFFE_MAP_EVALUATOR_IMAGE_INFO_HPP
#define CAFFE_MAP_EVALUATOR_IMAGE_INFO_HPP

/**
* @brief Structure containing info which is used in MAP computation.
*/
struct ImageInfo
{
    int id;
    int classId;
    bool isQuery;
    int classCount;
    double distance = -1;

    bool operator < (const ImageInfo& imageInfo) const
    {
        return (distance < imageInfo.distance);
    }
};

#endif
