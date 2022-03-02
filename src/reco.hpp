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

struct receiver_arg_struct {
   int port; 
};

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

struct inter_service_buffer {
    charint frame_id;
    charint previous_service;
    charint buffer_size;
    unsigned char* buffer;
};

double wallclock();
void loadImages(std::vector<char *> onlineImages); 
void trainParams(); 
void trainCacheParams(); 
void loadParams();
void encodeDatabase(int factor, int nn);
void test();
std::tuple<int, char*, char*> sift_processing(cv::Mat image, SiftData &siftData, std::vector<float> &enc_vec, bool online, bool isColorImage);
std::tuple<int, char*> encoding(float* sift_resg, int sift_result);
std::tuple<int, char*> lsh_nn(std::vector<float> enc_vec);
bool matching(std::vector<int> result, SiftData &tData, recognizedMarker &marker); 
bool query(cv::Mat queryImage, recognizedMarker &marker);
bool cacheQuery(cv::Mat queryImage, recognizedMarker &marker);
void addCacheItem(frameBuffer curFrame, resBuffer curRes);
void freeParams();
void scalabilityTest();
#endif
