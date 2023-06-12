#ifndef RECO_HPP
#define RECO_HPP

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cudaSift.h"

#define RECO_W_OFFSET 160
#define RECO_H_OFFSET 0

struct receiver_arg_struct
{
    int port;
};

union charint
{
    char b[4];
    int i;
};

union charfloat
{
    char b[4];
    float f;
};

struct frame_buffer
{
    std::string client_id;
    std::string client_ip;
    int client_port;
    int frame_no;
    int data_type;
    int buffer_size;
    char *buffer;
    int sift_buffer_size;
    char *sift_buffer;
};

struct resBuffer
{
    charint resID;
    charint resType;
    charint markerNum;
    char *buffer;
};

struct recognizedMarker
{
    charint markerID;
    charint height, width;
    cv::Point2f corners[4];
    std::string markername;
};

struct cacheItem
{
    std::vector<float> fv;
    SiftData data;
    frame_buffer curFrame;
    recognizedMarker curMarker;
};

struct matchingSiftItem
{
    char *client_id;
    int frame_no;
    SiftData data;
};

struct matching_args
{
    int udp_socket;
    matchingSiftItem sift_data;
};

struct inter_service_buffer
{
    std::string client_id;
    std::string client_ip;
    charint client_port;
    charint frame_no;
    charint data_type;
    charint previous_service;
    charint buffer_size;
    unsigned char *buffer;
    char *results_buffer;
    char *image_buffer;
    charint sift_buffer_size;
    char *sift_buffer;
};

struct matching_sift
{
    char *sift_data;
    int frame_no;
    char *client_id;
};
struct matching_item
{
    char *client_id;
    char *client_ip;
    int client_port;
    int frame_no;
    std::vector<int> lsh_result;
};

void print_log(std::string service_name, std::string client_id, std::string frame_no, std::string message);

double wallclock();
void load_images(std::string binary_directory, std::vector<char *> onlineImages);
void trainParams();
void trainCacheParams();
void load_params();
void encodeDatabase(int factor, int nn);
void test();

std::tuple<int, char*> lsh_nn(std::vector<float> enc_vec);
std::tuple<int, float *> sift_gpu(cv::Mat img, float **sift_res, float **sift_frame, SiftData &sift_data, int &w, int &h, bool online, bool is_color_image);

// void sift_processing(int &sift_points, char **sift_data_buffer, char **raw_sift_data, cv::Mat image, SiftData &siftData);
// std::tuple<int, char*> encoding(float* siftresg, int siftResult, std::vector<float> &enc_vec, bool cache, char** enc_vector);
// bool matching(std::vector<int> result, SiftData &tData, recognizedMarker &marker);
// bool query(cv::Mat queryImage, recognizedMarker &marker);
// bool cacheQuery(cv::Mat queryImage, recognizedMarker &marker);
// void addCacheItem(frame_buffer curFrame, resBuffer curRes);
void free_params();
// void scalabilityTest();

// bool query_sift(cv::Mat queryImage, recognizedMarker &marker);

#endif
