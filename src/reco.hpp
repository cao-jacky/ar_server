#ifndef RECO_HPP
#define RECO_HPP

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cudaSift.h"

#define RECO_W_OFFSET 160 
#define RECO_H_OFFSET 0 

union charint {
    char b[4];
    int i;
};

union charfloat {
    char b[4];
    float f;
};

struct frameBuffer {
    int frmID;
    int dataType;
    int bufferSize;
    char* buffer;
};

struct resBuffer {
    charint resID;
    charint resType;
    charint markerNum;
    char* buffer;
};

struct recognizedMarker {
    charint markerID;
    charint height, width;
    cv::Point2f corners[4];
    std::string markername;
};

struct cacheItem {
    std::vector<float> fv;
    SiftData data;
    frameBuffer curFrame;
    recognizedMarker curMarker;
};

double wallclock();
void loadImages(std::vector<char *> onlineImages); 
void trainParams(); 
void trainCacheParams(); 
void loadParams();
void encodeDatabase(int factor, int nn);
void test();
bool query(cv::Mat queryImage, recognizedMarker &marker);
bool cacheQuery(cv::Mat queryImage, recognizedMarker &marker);
void addCacheItem(frameBuffer curFrame, resBuffer curRes);
void freeParams();
void scalabilityTest();
#endif
