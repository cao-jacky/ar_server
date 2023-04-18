#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "sys/times.h"
//#include "sys/vtimes.h"

#include "reco.hpp"
#include <fstream>
#include <ctime>
#include <set>
#include <iterator>
#include <dirent.h>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>
#include <thread>
#include <algorithm>
#include <chrono>
#include <random>
#include <atomic>
#include <tuple>
#include <iomanip>

#include "cudaImage.h"
#include "cuda_files.h"
#include "Eigen/Dense"
#include "falconn/lsh_nn_table.h"
extern "C"
{
#include "vl/generic.h"
#include "vl/gmm.h"
#include "vl/fisher.h"
#include "vl/mathop.h"
}

using namespace std;
using namespace falconn;
using namespace Eigen;
using namespace cv;

#define NUM_CLUSTERS 32
#define SIZE 5248 // 82 * 2 * 32
#define ROWS 200
#define TYPE float
#define DST_DIM 80
#define NUM_HASH_TABLES 20
#define NUM_HASH_BITS 24
#define SUB_DATASET 110
//#define SUB_DATASET 1000
//#define FEATURE_CHECK
#define MATCH_ONE_ONLY
//#define TEST

int querysizefactor, nn_num;
float *means, *covariances, *priors, *projectionCenter, *projection;
vector<char *> whole_list;
vector<SiftData> trainData;
vector<DenseVector<float>> lsh;
unique_ptr<LSHNearestNeighborTable<DenseVector<float>>> tablet;
unique_ptr<LSHNearestNeighborQuery<DenseVector<float>>> table;
vector<cacheItem> cacheItems;
atomic<int> totalTime;

long double a[4], b[4], loadavg;
FILE *fp;
char dump[50];

void print_log(string service_name, string client_id, string frame_no, string message)
{
    cout << "{\\\"service_name\\\": \\\"" << service_name << "\\\", \\\"client_id\\\": \\\"" << client_id;
    cout << "\\\", \\\"frame_no\\\": \\\"" << frame_no << "\\\", \\\"timestamp\\\": \\\"" << setprecision(15) << wallclock() * 1000;
    cout << "\\\", \\\"message\\\": \\\"" << message << "\\\"}" << endl;
}

int parseLine(char *line)
{
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char *p = line;
    while (*p < '0' || *p > '9')
        p++;
    line[i - 3] = '\0';
    i = atoi(p);
    return i;
}

int getValueVirtualMem()
{ // Note: this value is in KB!
    FILE *file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL)
    {
        if (strncmp(line, "VmSize:", 7) == 0)
        {
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

int getValuePhysicalMem()
{ // Note: this value is in KB!
    FILE *file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL)
    {
        if (strncmp(line, "VmRSS:", 6) == 0)
        {
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

double wallclock(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int msec = tv.tv_usec / 1000;
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

char *export_siftdata(SiftData &data_struct, char **sift_data_array)
{
    int spf = 15; // sift point features number

    int num_points = data_struct.numPts;
    int max_points = data_struct.maxPts;

    SiftPoint *cpu_data = data_struct.h_data;

    // data appears to be mostly stored on the CPU, so below returns nothing
    // SiftPoint *gpu_data = data_struct.d_data;

    // allocating memory into char*, taking two array as the last two features
    // are arrays of size 3 and 128 respectively
    // int sd_size = num_points * (spf - 2 + 3 + 128);
    int sd_size = num_points * (4 * (spf + 3 + 128));
    // char *sift_data = (char *)calloc(sd_size, sizeof(float));
    // char *sift_data = new char[sd_size];

    *sift_data_array = (char*)malloc(sd_size);
    char *sift_data = *sift_data_array;

    int curr_posn = 0; // current position in char array

    // inserting data for num_points and max_points
    charint sd_num_points;
    sd_num_points.i = num_points;
    memcpy(&(sift_data[curr_posn]), sd_num_points.b, 4);
    curr_posn += 4;

    charint sd_max_points;
    sd_max_points.i = max_points;
    memcpy(&(sift_data[curr_posn]), sd_max_points.b, 4);
    curr_posn += 4;

    for (int i = 0; i < num_points; i++)
    {
        SiftPoint *curr_data = (&cpu_data[i]);

        // going through all parts of the SiftPoint structure and storing into char*
        charfloat cd_xpos;
        cd_xpos.f = curr_data->xpos;
        memcpy(&(sift_data[curr_posn]), cd_xpos.b, 4);
        curr_posn += 4;

        charfloat cd_ypos;
        cd_ypos.f = curr_data->ypos;
        memcpy(&(sift_data[curr_posn]), cd_ypos.b, 4);
        curr_posn += 4;

        charfloat cd_scale;
        cd_scale.f = curr_data->scale;
        memcpy(&(sift_data[curr_posn]), cd_scale.b, 4);
        curr_posn += 4;

        charfloat cd_sharpness;
        cd_sharpness.f = curr_data->sharpness;
        memcpy(&(sift_data[curr_posn]), cd_sharpness.b, 4);
        curr_posn += 4;

        charfloat cd_edgeness;
        cd_edgeness.f = curr_data->edgeness;
        memcpy(&(sift_data[curr_posn]), cd_edgeness.b, 4);
        curr_posn += 4;

        charfloat cd_orientation;
        cd_orientation.f = curr_data->orientation;
        memcpy(&(sift_data[curr_posn]), cd_orientation.b, 4);
        curr_posn += 4;

        charfloat cd_score;
        cd_score.f = curr_data->score;
        memcpy(&(sift_data[curr_posn]), cd_score.b, 4);
        curr_posn += 4;

        charfloat cd_ambiguity;
        cd_ambiguity.f = curr_data->ambiguity;
        memcpy(&(sift_data[curr_posn]), cd_ambiguity.b, 4);
        curr_posn += 4;

        charint cd_match;
        cd_match.i = curr_data->match;
        memcpy(&(sift_data[curr_posn]), cd_match.b, 4);
        curr_posn += 4;

        charfloat cd_match_xpos;
        cd_match_xpos.f = curr_data->match_xpos;
        memcpy(&(sift_data[curr_posn]), cd_match_xpos.b, 4);
        curr_posn += 4;

        charfloat cd_match_ypos;
        cd_match_ypos.f = curr_data->match_ypos;
        memcpy(&(sift_data[curr_posn]), cd_match_ypos.b, 4);
        curr_posn += 4;

        charfloat cd_match_error;
        cd_match_error.f = curr_data->match_error;
        memcpy(&(sift_data[curr_posn]), cd_match_error.b, 4);
        curr_posn += 4;

        charfloat cd_subsampling;
        cd_subsampling.f = curr_data->subsampling;
        memcpy(&(sift_data[curr_posn]), cd_subsampling.b, 4);
        curr_posn += 4;

        // empty array with 3 elements
        for (int j = 0; j < 3; j++)
        {
            charfloat cd_empty;
            cd_empty.f = curr_data->empty[j];
            memcpy(&(sift_data[curr_posn]), cd_empty.b, 4);
            curr_posn += 4;
        }

        // data array with 128 elements
        for (int k = 0; k < 128; k++)
        {
            charfloat cd_data;
            cd_data.f = curr_data->data[k];
            memcpy(&(sift_data[curr_posn]), cd_data.b, 4);
            curr_posn += 4;
        }
    }
    // return sift_data;
}

tuple<int, float *> sift_gpu(Mat img, float **siftres, float **siftframe, SiftData &siftData, int &w, int &h, bool online, bool isColorImage)
{
    CudaImage cimg;
    int numPts;
    double start, finish, durationgmm;

    // if(online) resize(img, img, Size(), 0.5, 0.5);
    if (isColorImage)
        cvtColor(img, img, COLOR_BGR2GRAY);
    img.convertTo(img, CV_32FC1);
    start = wallclock();
    w = img.cols;
    h = img.rows;
    // cout << "Image size = (" << w << "," << h << ")" << endl;

    cimg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)img.data);
    cimg.Download();

    float initBlur = 1.0f;
    float thresh = 2.0f;
    InitSiftData(siftData, 1000, true, true);
    ExtractSift(siftData, cimg, 5, initBlur, thresh, 0.0f, false);

    numPts = siftData.numPts;
    *siftres = (float *)malloc(sizeof(float) * 128 * numPts);
    *siftframe = (float *)malloc(sizeof(float) * 2 * numPts);
    // *siftres = (float *)calloc(sizeof(float), sizeof(float)*128*numPts);
    // *siftframe = (float *)calloc(sizeof(char), sizeof(float)*2*numPts);
    float *curRes = *siftres;
    float *curframe = *siftframe;
    SiftPoint *p = siftData.h_data;

    for (int i = 0; i < numPts; i++)
    {
        memcpy(curRes, p->data, (128 + 1) * sizeof(float));
        curRes += 128;

        *curframe++ = p->xpos / w - 0.5;
        *curframe++ = p->ypos / h - 0.5;
        p++;
    }

    // char *final_sift_data = export_siftdata(siftData);

    if (!online)
        FreeSiftData(siftData); //
    // FreeSiftData(siftData);

    finish = wallclock();
    durationgmm = (double)(finish - start);
    print_log("", "0", "0", to_string(numPts) + " SIFT points extracted in " + to_string(durationgmm * 1000) + " ms");

    return make_tuple(numPts, curRes);
}

tuple<float *> sift_gpu_new(Mat img, float **siftres, float **siftframe, SiftData &siftData, int &w, int &h, bool online, bool isColorImage, int &num_points, char **raw_sift_data)
{
    CudaImage cimg;
    double start, finish, durationgmm;

    if (isColorImage)
        cvtColor(img, img, COLOR_BGR2GRAY);
    img.convertTo(img, CV_32FC1);
    start = wallclock();
    w = img.cols;
    h = img.rows;

    cimg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)img.data);
    cimg.Download();

    float initBlur = 1.0f;
    float thresh = 2.0f;
    InitSiftData(siftData, 1000, true, true);
    ExtractSift(siftData, cimg, 5, initBlur, thresh, 0.0f, false);

    num_points = siftData.numPts;
    *siftres = (float *)malloc(sizeof(float) * 128 * num_points);
    *siftframe = (float *)malloc(sizeof(float) * 2 * num_points);
    // *siftres = (float *)calloc(sizeof(float), sizeof(float)*128*num_points);
    // *siftframe = (float *)calloc(sizeof(char), sizeof(float)*2*num_points);
    float *curRes = *siftres;
    float *curframe = *siftframe;
    SiftPoint *p = siftData.h_data;

    for (int i = 0; i < num_points; i++)
    {
        memcpy(curRes, p->data, (128 + 1) * sizeof(float));
        curRes += 128;

        *curframe++ = p->xpos / w - 0.5;
        *curframe++ = p->ypos / h - 0.5;
        p++;
    }

    // *raw_sift_data = export_siftdata(siftData);
    export_siftdata(siftData, raw_sift_data);

    if (!online)
        FreeSiftData(siftData);
    // FreeSiftData(siftData);

    finish = wallclock();
    durationgmm = (double)(finish - start);
    print_log("", "0", "0", to_string(num_points) + " SIFT points extracted in " + to_string(durationgmm * 1000) + " ms");

    return make_tuple(curRes);
}

void sift_processing(int &sift_points, char **sift_data_buffer, char **raw_sift_data, Mat image, SiftData &siftData)
{
    float *siftresg;
    float *siftframe;
    int height, width;
    float *curr_res;

    auto sift_gpu_results = sift_gpu_new(image, &siftresg, &siftframe, siftData, width, height, true, false, sift_points, raw_sift_data);
    curr_res = get<0>(sift_gpu_results);

    // copying the data to the function-passed variable
    *sift_data_buffer = (char *)malloc(sift_points * 128 * 4);
    char *sdb_tmp = *sift_data_buffer;

    int buffer_count = 0;
    int bytes_count = 0;
    for (int i = 0; i < sift_points * 128; i++)
    {
        charfloat sift_curr_result;
        sift_curr_result.f = *&siftresg[i];
        memcpy(&sdb_tmp[buffer_count], sift_curr_result.b, 4);

        buffer_count += 4;
    }

    free(siftresg);
    free(siftframe);
}

tuple<int, char *> encoding(float *siftresg, int siftResult, vector<float> &enc_vec, bool cache, char** enc_vector)
{
    double start, finish;
    double durationsift, durationgmm;

    char *encoded_vector = *enc_vector;

    float enc[SIZE] = {0};

    start = wallclock();
    float *dest = (float *)malloc(siftResult * 82 * sizeof(float));
    gpu_pca_mm(projection, projectionCenter, siftresg, dest, siftResult, DST_DIM);

    finish = wallclock();
    durationgmm = (double)(finish - start);
    print_log("encoding", "0", "0", "PCA encoding took a time of " + to_string(durationgmm) + " ms");

    start = wallclock();
    gpu_gmm_1(covariances, priors, means, NULL, NUM_CLUSTERS, 82, siftResult, (82 / 2.0) * log(2.0 * VL_PI), enc, NULL, dest);

    ///////////WARNING: add the other NOOP
    float sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += enc[i] * enc[i];
    }
    for (int i = 0; i < SIZE; i++)
    {
        // WARNING: didn't use the max operation
        enc[i] /= sqrt(sum);
    }
    sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += enc[i] * enc[i];
    }
    for (int i = 0; i < SIZE; i++)
    {
        // WARNING: didn't use the max operation
        enc[i] /= sqrt(sum);
    }

    enc_vec = vector<float>(enc, enc + SIZE);
    finish = wallclock();

    durationgmm = (double)(finish - start);
    print_log("encoding", "0", "0", "Fisher Vector encoding took a time of " + to_string(durationgmm * 1000) + " ms");

    // transforming the vector of floats into a char*
    float *enc_vec_floats = &(enc_vec[0]);
    // encoded_vector = new char[4 * SIZE];
    memset(encoded_vector, 0, 4 * SIZE);
    int buffer_count = 0;
    for (float x : enc_vec)
    {
        charfloat enc_vec_result;
        enc_vec_result.f = x;
        memcpy(&(encoded_vector[buffer_count]), enc_vec_result.b, 4);
        buffer_count += 4;
    }
    free(dest);
    return make_tuple(SIZE, encoded_vector);
}

tuple<int, char *> lsh_nn(vector<float> enc_vec)
{
    vector<float> test;
    vector<int> result;
    DenseVector<float> t(SIZE);

    char *encoded_results;

    double start, finish;
    double duration_lshnn;

    test = enc_vec;

    for (int j = 0; j < SIZE; j++)
    {
        t[j] = test[j];
    }
    start = wallclock();
    table->find_k_nearest_neighbors(t, nn_num, &result);
    finish = wallclock();
    duration_lshnn = (double)(finish - start);
    print_log("lsh", "0", "0", "LSH NN search took a time of " + to_string(duration_lshnn * 1000) + " ms");

    int enc_res_size = result.size();
    encoded_results = (char *)malloc(enc_res_size * sizeof(int));
    int buffer_count = 0;
    for (int x : result)
    {
        charint enc_res;
        enc_res.i = x;
        memcpy(&(encoded_results[buffer_count]), enc_res.b, 4);
        buffer_count += 4;
    }
    return make_tuple(enc_res_size, encoded_results);
}

bool matching(vector<int> result, SiftData &tData, recognizedMarker &marker)
{
    float homography[9];
    int numMatches;

    for (int idx = 0; idx < result.size(); idx++)
    {
        print_log("matching", "0", "0", "Testing " + to_string(result[idx]) + " " + whole_list[result[idx]]);

        Mat image = imread(whole_list[result[idx]], IMREAD_COLOR);
        SiftData sData;
        int w, h;
        float *a, *b;
        sift_gpu(image, &a, &b, sData, w, h, true, true);

        print_log("matching", "0", "0", "Number of feature points: " + to_string(sData.numPts) + " " + to_string(tData.numPts));
        MatchSiftData(sData, tData);
        FindHomography(sData, homography, &numMatches, 10000, 0.00f, 0.85f, 5.0);
        int numFit = ImproveHomography(sData, homography, 5, 0.00f, 0.80f, 2.0);
        double ratio = 100.0f * numFit / min(sData.numPts, tData.numPts);
        print_log("matching", "0", "0", "Matching features: " + to_string(numFit) + " " + to_string(numMatches) + " " + to_string(ratio) + "% ");

        if (ratio > 10)
        {
            Mat H(3, 3, CV_32FC1, homography);

            vector<Point2f> obj_corners(4), scene_corners(4);
            obj_corners[0] = Point(0, 0);
            obj_corners[1] = Point(image.cols, 0);
            obj_corners[2] = Point(image.cols, image.rows);
            obj_corners[3] = Point(0, image.rows);

            try
            {
                perspectiveTransform(obj_corners, scene_corners, H);
            }
            catch (Exception)
            {
                cout << "cv exception" << endl;
                continue;
            }

            marker.markerID.i = result[idx];
            marker.height.i = image.rows;
            marker.width.i = image.cols;

            for (int i = 0; i < 4; i++)
            {
                marker.corners[i].x = scene_corners[i].x + RECO_W_OFFSET;
                marker.corners[i].y = scene_corners[i].y + RECO_H_OFFSET;
            }
            marker.markername = "gpu_recognized_image.";

            // cout<<"after matching "<<wallclock()<<endl;
            FreeSiftData(sData);
            FreeSiftData(tData);
            print_log("matching", "0", "0", "Recognised object(s)");
            return true;
        }
        else
        {
            print_log("matching", "0", "0", "No matching objects");
        }
        // free(a);
        // free(b);
    }
    FreeSiftData(tData);
}

void onlineProcessing(Mat image, SiftData &siftData, vector<float> &enc_vec, bool online, bool isColorImage, bool cache)
{
    double start, finish;
    double durationsift, durationgmm;
    int siftResult;
    float *siftresg;
    float *siftframe;
    int height, width;

    float homography[9];
    int numMatches;

    float *dest;

    auto sift_gpu_results = sift_gpu(image, &siftresg, &siftframe, siftData, width, height, online, isColorImage);
    siftResult = get<0>(sift_gpu_results);

    float enc[SIZE] = {0};

    if (cache)
    {
        start = wallclock();
        gpu_gmm_1(covariances, priors, means, NULL, NUM_CLUSTERS, 82, siftResult, (82 / 2.0) * log(2.0 * VL_PI), enc, NULL, siftresg);
    }
    else
    {
        start = wallclock();
        dest = (float *)malloc(siftResult * 82 * sizeof(float));
        gpu_pca_mm(projection, projectionCenter, siftresg, dest, siftResult, DST_DIM);

        finish = wallclock();
        durationgmm = (double)(finish - start);
        cout << "PCA encoding time: " << durationgmm << endl;
        

        start = wallclock();
        gpu_gmm_1(covariances, priors, means, NULL, NUM_CLUSTERS, 82, siftResult, (82 / 2.0) * log(2.0 * VL_PI), enc, NULL, dest);
    }

    ///////////WARNING: add the other NOOP
    float sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += enc[i] * enc[i];
    }
    for (int i = 0; i < SIZE; i++)
    {
        // WARNING: didn't use the max operation
        enc[i] /= sqrt(sum);
    }
    sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += enc[i] * enc[i];
    }
    for (int i = 0; i < SIZE; i++)
    {
        // WARNING: didn't use the max operation
        enc[i] /= sqrt(sum);
    }

    enc_vec = vector<float>(enc, enc + SIZE);

    finish = wallclock();
    durationgmm = (double)(finish - start);
    cout << "Fisher Vector encoding time: " << durationgmm << endl;

    free(dest);
    free(siftresg);
    free(siftframe);
}

void encodeDatabase(int factor, int nn)
{
    querysizefactor = factor;
    nn_num = nn;

    gpu_copy(covariances, priors, means, NUM_CLUSTERS, DST_DIM + 2);

    vector<float> train[whole_list.size()];
    // Encode train files
    for (int i = 0; i < whole_list.size(); i++)
    {
        SiftData sData;
        Mat image = imread(whole_list[i], IMREAD_COLOR);
#ifdef TEST
        onlineProcessing(image, sData, train[i], true, true, false);
        if (i < 20)
            trainData.push_back(sData);
        else
            FreeSiftData(sData);
#else
        onlineProcessing(image, sData, train[i], false, true, false);
#endif
    }

    for (int i = 0; i < whole_list.size(); i++)
    {
        DenseVector<float> FV(SIZE);
        for (int j = 0; j < SIZE; j++)
            FV[j] = train[i][j];
        lsh.push_back(FV);
    }

    LSHConstructionParameters params = get_default_parameters<DenseVector<float>>((int_fast64_t)lsh.size(), (int_fast32_t)SIZE, DistanceFunction::EuclideanSquared, true);
    params.l = 32;
    params.k = 1;

    tablet = construct_table<DenseVector<float>>(lsh, params);
    table = tablet->construct_query_object(100);
    cout << "LSH tables preparation done" << endl;
}

bool query(Mat queryImage, recognizedMarker &marker)
{
    SiftData tData;
    vector<float> test;
    vector<int> result;
    DenseVector<float> t(SIZE);
    float homography[9];
    int numMatches;

    double start, finish;
    double duration_lshnn;

    onlineProcessing(queryImage, tData, test, true, false, false);

    for (int j = 0; j < SIZE; j++)
    {
        t[j] = test[j];
    }
    start = wallclock();
    table->find_k_nearest_neighbors(t, nn_num, &result);
    finish = wallclock();
    duration_lshnn = (double)(finish - start);
    cout << "LSH NN searching time: " << duration_lshnn << endl;
    cout << "Query - time before matching: " << wallclock() << endl;

    for (int idx = 0; idx < result.size(); idx++)
    {
        cout << "Testing " << result[idx] << endl;

#ifdef TEST
        // Mat image(741, 500, CV_8UC1);
        Mat image = imread(whole_list[result[idx]], IMREAD_COLOR);
        if (result[idx] >= 100)
            break;
        SiftData sData = trainData[result[idx]];
#else
        Mat image = imread(whole_list[result[idx]], IMREAD_COLOR);
        SiftData sData;
        int w, h;
        float *a, *b;
        sift_gpu(image, &a, &b, sData, w, h, true, true);
#endif

        cout << "Number of feature points: " << sData.numPts << " " << tData.numPts << endl;
        MatchSiftData(sData, tData);
        FindHomography(sData, homography, &numMatches, 10000, 0.00f, 0.85f, 5.0);
        int numFit = ImproveHomography(sData, homography, 5, 0.00f, 0.80f, 2.0);
        double ratio = 100.0f * numFit / min(sData.numPts, tData.numPts);
        cout << "Matching features: " << numFit << " " << numMatches << " " << ratio << "% " << endl;
#ifndef TEST
        FreeSiftData(sData);
#endif

        if (ratio > 10)
        {
            Mat H(3, 3, CV_32FC1, homography);

            vector<Point2f> obj_corners(4), scene_corners(4);
            obj_corners[0] = Point(0, 0);
            obj_corners[1] = Point(image.cols, 0);
            obj_corners[2] = Point(image.cols, image.rows);
            obj_corners[3] = Point(0, image.rows);

            try
            {
                perspectiveTransform(obj_corners, scene_corners, H);
            }
            catch (Exception)
            {
                cout << "cv exception" << endl;
                continue;
            }

            marker.markerID.i = result[idx];
            marker.height.i = image.rows;
            marker.width.i = image.cols;

            for (int i = 0; i < 4; i++)
            {
                marker.corners[i].x = scene_corners[i].x + RECO_W_OFFSET;
                marker.corners[i].y = scene_corners[i].y + RECO_H_OFFSET;
            }
            marker.markername = "gpu_recognized_image.";

            FreeSiftData(tData);
            cout << "after matching " << wallclock() << endl;
            return true;
        }
#ifdef MATCH_ONE_ONLY
        break;
#endif
    }
    FreeSiftData(tData);

    return false;
}

bool cacheQuery(Mat queryImage, recognizedMarker &marker)
{
    SiftData sData, tData;
    vector<float> test;
    vector<int> result;
    float homography[9];
    int numMatches;

    if (cacheItems.size() == 0)
        return false;

    onlineProcessing(queryImage, tData, test, true, false, true);

    double minDistance = 999999999;
    int index = -1;
    for (int idx = 0; idx < cacheItems.size(); idx++)
    {
        double dis = norm(cacheItems[idx].fv, test);
        if (dis < minDistance)
        {
            minDistance = dis;
            index = idx;
        }
    }
    sData = cacheItems[index].data;
    cout << "Cache query - time before matching: " << wallclock() << endl;

    cout << "Number of feature points: " << sData.numPts << " " << tData.numPts << endl;
    MatchSiftData(sData, tData);
    FindHomography(sData, homography, &numMatches, 10000, 0.00f, 0.85f, 5.0);
    int numFit = ImproveHomography(sData, homography, 5, 0.00f, 0.80f, 2.0);
    double ratio = 100.0f * numFit / min(sData.numPts, tData.numPts);
    cout << "Matching features: " << numFit << " " << numMatches << " " << ratio << "% " << endl;
    FreeSiftData(tData);

    if (ratio > 10)
    {
        Mat H(3, 3, CV_32FC1, homography);

        vector<Point2f> obj_corners, scene_corners(4);
        for (int i = 0; i < 4; i++)
            obj_corners.push_back(cacheItems[index].curMarker.corners[i]);

        try
        {
            perspectiveTransform(obj_corners, scene_corners, H);
        }
        catch (Exception)
        {
            cout << "cv exception" << endl;
            return false;
        }

        marker.markerID.i = cacheItems[index].curMarker.markerID.i;
        marker.height.i = cacheItems[index].curMarker.height.i;
        marker.width.i = cacheItems[index].curMarker.width.i;

        for (int i = 0; i < 4; i++)
        {
            marker.corners[i].x = scene_corners[i].x + RECO_W_OFFSET;
            marker.corners[i].y = scene_corners[i].y + RECO_H_OFFSET;
        }
        marker.markername = "gpu_recognized_image.";

        return true;
    }
    else
    {
        return false;
    }
}

void addCacheItem(frame_buffer curr_frame, resBuffer curRes)
{
    SiftData tData;
    vector<float> test;

    vector<uchar> imagedata(curr_frame.buffer, curr_frame.buffer + curr_frame.buffer_size);
    Mat queryImage = imdecode(imagedata, IMREAD_GRAYSCALE);
    Mat cacheQueryImage = queryImage(Rect(RECO_W_OFFSET, RECO_H_OFFSET, 160, 270));

    onlineProcessing(cacheQueryImage, tData, test, true, false, true);

    recognizedMarker marker;
    int pointer = 0;
    memcpy(marker.markerID.b, &(curRes.buffer[pointer]), 4);
    pointer += 4;
    memcpy(marker.height.b, &(curRes.buffer[pointer]), 4);
    pointer += 4;
    memcpy(marker.width.b, &(curRes.buffer[pointer]), 4);
    pointer += 4;

    charfloat p;
    for (int j = 0; j < 4; j++)
    {
        memcpy(p.b, &(curRes.buffer[pointer]), 4);
        marker.corners[j].x = p.f;
        pointer += 4;
        memcpy(p.b, &(curRes.buffer[pointer]), 4);
        marker.corners[j].y = p.f;
        pointer += 4;
    }

    char name[40];
    memcpy(name, &(curRes.buffer[pointer]), 40);
    marker.markername = name;

    cacheItem newItem;
    newItem.fv = test;
    newItem.data = tData;
    newItem.curFrame = curr_frame;
    newItem.curMarker = marker;
    cacheItems.push_back(newItem);
    cout << "Cache item inserted at " << wallclock() << endl
         << endl;
}

bool mycompare(char *x, char *y)
{
    if (strcmp(x, y) <= 0)
        return 1;
    else
        return 0;
}

void loadImages(vector<char *> onlineImages)
{
    gpu_init();

    const char *home = "data/bk_train";
    DIR *d = opendir(home);
    struct dirent *cur_dir;
    vector<char *> paths;
    while ((cur_dir = readdir(d)) != NULL)
    {
        if ((strcmp(cur_dir->d_name, ".") != 0) && (strcmp(cur_dir->d_name, "..") != 0))
        {
            char *temppath = new char[256];
            sprintf(temppath, "%s/%s", home, cur_dir->d_name);

            paths.push_back(temppath);
        }
    }
    sort(paths.begin(), paths.end(), mycompare);

    for (int i = 0; i < onlineImages.size(); i++)
    {
        cout << "online image: " << onlineImages[i] << endl;
        whole_list.push_back(onlineImages[i]);
    }

    for (int i = 0; i < paths.size(); i++)
    {
        DIR *subd = opendir(paths[i]);
        struct dirent *cur_subdir;
        while ((cur_subdir = readdir(subd)) != NULL)
        {
            if ((strcmp(cur_subdir->d_name, ".") != 0) && (strcmp(cur_subdir->d_name, "..") != 0))
            {
                char *file = new char[256];
                sprintf(file, "%s/%s", paths[i], cur_subdir->d_name);
                if (strstr(file, "jpg") != NULL)
                {
                    whole_list.push_back(file);
#ifdef SUB_DATASET
                    if (whole_list.size() == SUB_DATASET)
                        break;
#endif
                }
            }
        }
        closedir(subd);
    }
    // sort(whole_list.begin(), whole_list.end(), mycompare);
    cout << endl
         << "---------------------in total " << whole_list.size() << " images------------------------" << endl
         << endl;
    closedir(d);
}

void trainParams()
{
    int numData;
    int dimension = DST_DIM + 2;
    float *sift_res;
    float *sift_frame;

    float *final_res = (float *)malloc(ROWS * whole_list.size() * 128 * sizeof(float));
    float *final_frame = (float *)malloc(ROWS * whole_list.size() * 128 * sizeof(float));
    Mat training_descriptors(0, 128, CV_32FC1);
    //////////////////train encoder //////////////// //////// STEP 0: obtain sample image descriptors set<int>::iterator iter; double start_time = wallclock();

    for (int i = 0; i != whole_list.size(); ++i)
    {
        char imagename[256], imagesizename[256];
        int height, width;
        float *pca_desc;
        // get descriptors
        cout << "Train file " << i << ": " << whole_list[i] << endl;
        //    if(count == 2)
        SiftData siftData;
        Mat image = imread(whole_list[i], IMREAD_COLOR);

        auto sift_gpu_results = sift_gpu(image, &sift_res, &sift_frame, siftData, width, height, false, true);
        int pre_size = get<0>(sift_gpu_results);

#ifdef FEATURE_CHECK
        if (pre_size < ROWS)
            remove(whole_list[i]);
        continue;
#endif

        srand(1);
        set<int> indices;
        for (int it = 0; it < ROWS; it++)
        {
            int rand_t = rand() % pre_size;
            pair<set<int>::iterator, bool> ret = indices.insert(rand_t);
            while (ret.second == false)
            {
                rand_t = rand() % pre_size;
                ret = indices.insert(rand_t);
            }
        }
        set<int>::iterator iter;
        int it = 0;
        for (iter = indices.begin(); iter != indices.end(); iter++)
        {
            Mat descriptor = Mat(1, 128, CV_32FC1, sift_res + (*iter) * 128);
            training_descriptors.push_back(descriptor);

            for (int k = 0; k < 128; k++)
            {
                final_res[(ROWS * i + it) * 128 + k] = sift_res[(*iter) * 128 + k];
            }
            for (int k = 0; k < 2; k++)
            {
                final_frame[(ROWS * i + it) * 2 + k] = sift_frame[(*iter) * 2 + k];
            }

            it++;
        }
        free(sift_res);
        free(sift_frame);
    }
    /////////////STEP 1: PCA
    projectionCenter = (float *)malloc(128 * sizeof(float));
    projection = (float *)malloc(128 * 80 * sizeof(float));

    PCA pca(training_descriptors, Mat(), PCA::DATA_AS_ROW, 80);
    for (int i = 0; i < 128; i++)
    {
        projectionCenter[i] = pca.mean.at<float>(0, i);
    }
    for (int i = 0; i < 80; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            projection[i * 128 + j] = pca.eigenvectors.at<float>(i, j);
        }
    }
    cout << "================ PCA training finished ===================" << endl;

    ///////STEP 2  (optional): geometrically augment the features
    int pre_size = training_descriptors.rows;
    float *dest = (float *)malloc(pre_size * (DST_DIM + 2) * sizeof(float));

    gpu_pca_mm(projection, projectionCenter, final_res, dest, pre_size, DST_DIM);
    for (int i = 0; i < pre_size; i++)
    {
        dest[i * (DST_DIM + 2) + DST_DIM] = final_frame[i * 2];
        dest[i * (DST_DIM + 2) + DST_DIM + 1] = final_frame[i * 2 + 1];
    }
    free(final_frame);
    free(final_res);

    //////////////////////STEP 3  learn a GMM vocabulary
    numData = pre_size;
    // vl_twister
    VlRand *rand;
    rand = vl_get_rand();
    vl_rand_seed(rand, 1);

    VlGMM *gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, NUM_CLUSTERS);
    ///////////////////////WARNING: should set these parameters
    vl_gmm_set_initialization(gmm, VlGMMKMeans);
    // Compute V
    double denom = pre_size - 1;
    double xbar[82], V[82];
    for (int i = 0; i < dimension; i++)
    {
        xbar[i] = 0.0;
        for (int j = 0; j < numData; j++)
        {
            xbar[i] += (double)dest[j * dimension + i];
        }
        xbar[i] /= (double)pre_size;
    }
    for (int i = 0; i < dimension; i++)
    {
        double absx = 0.0;
        for (int j = 0; j < numData; j++)
        {
            double tempx = (double)dest[j * dimension + i] - xbar[i];
            absx += abs(tempx) * abs(tempx);
        }
        V[i] = absx / denom;
    }

    // Get max(V)
    double maxNum = V[0];
    for (int i = 1; i < dimension; i++)
    {
        if (V[i] > maxNum)
        {
            maxNum = V[i];
        }
    }
    maxNum = maxNum * 0.0001;
    vl_gmm_set_covariance_lower_bound(gmm, (double)maxNum);
    cout << "Lower bound " << maxNum << endl;
    vl_gmm_set_verbosity(gmm, 1);
    vl_gmm_set_max_num_iterations(gmm, 100);
    printf("vl_gmm: data type = %s\n", vl_get_type_name(vl_gmm_get_data_type(gmm)));
    printf("vl_gmm: data dimension = %d\n", dimension);
    printf("vl_gmm: num. data points = %d\n", numData);
    printf("vl_gmm: num. Gaussian modes = %d\n", NUM_CLUSTERS);
    printf("vl_gmm: lower bound on covariance = [");
    printf(" %f %f ... %f]\n",
           vl_gmm_get_covariance_lower_bounds(gmm)[0],
           vl_gmm_get_covariance_lower_bounds(gmm)[1],
           vl_gmm_get_covariance_lower_bounds(gmm)[dimension - 1]);

    double gmmres = vl_gmm_cluster(gmm, dest, numData);
    free(dest);
    cout << "GMM ending cluster." << endl;

    priors = (TYPE *)vl_gmm_get_priors(gmm);
    means = (TYPE *)vl_gmm_get_means(gmm);
    covariances = (TYPE *)vl_gmm_get_covariances(gmm);
    cout << "End of encoder " << endl;
    // cout << "Training time " << wallclock() - start_time << endl;
    ///////////////END train encoer//////////

    ofstream out1("params/priors", ios::out | ios::binary);
    out1.write((char *)priors, sizeof(float) * NUM_CLUSTERS);
    out1.close();

    ofstream out2("params/means", ios::out | ios::binary);
    out2.write((char *)means, sizeof(float) * dimension * NUM_CLUSTERS);
    out2.close();

    ofstream out3("params/covariances", ios::out | ios::binary);
    out3.write((char *)covariances, sizeof(float) * dimension * NUM_CLUSTERS);
    out3.close();

    ofstream out4("params/projection", ios::out | ios::binary);
    out4.write((char *)projection, sizeof(float) * 80 * 128);
    out4.close();

    ofstream out5("params/projectionCenter", ios::out | ios::binary);
    out5.write((char *)projectionCenter, sizeof(float) * 128);
    out5.close();
}

void trainCacheParams()
{
    int numData;
    int dimension = 128;
    float *sift_res;
    float *sift_frame;

    float *final_res = (float *)malloc(ROWS * whole_list.size() * 128 * sizeof(float));
    float *final_frame = (float *)malloc(ROWS * whole_list.size() * 128 * sizeof(float));
    // float *final_res = (float *)calloc(sizeof(float), ROWS * whole_list.size() * 128 * sizeof(float));
    // float *final_frame = (float *)calloc(sizeof(float), ROWS * whole_list.size() * 128 * sizeof(float));
    Mat training_descriptors(0, 128, CV_32FC1);
    //////////////////train encoder ////////////////
    //////// STEP 1: obtain sample image descriptors
    set<int>::iterator iter;
    double start_time = wallclock();

    for (int i = 0; i != whole_list.size(); ++i)
    {
        char imagename[256], imagesizename[256];
        int height, width;
        // get descriptors
        cout << "Train file " << i << ": " << whole_list[i] << endl;
        SiftData siftData;
        Mat image = imread(whole_list[i], IMREAD_COLOR);
        auto sift_gpu_results = sift_gpu(image, &sift_res, &sift_frame, siftData, width, height, false, true);
        int pre_size = get<0>(sift_gpu_results);
#ifdef FEATURE_CHECK
        if (pre_size < ROWS)
            remove(whole_list[i]);
        continue;
#endif

        srand(1);
        set<int> indices;
        for (int it = 0; it < ROWS; it++)
        {
            int rand_t = rand() % pre_size;
            pair<set<int>::iterator, bool> ret = indices.insert(rand_t);
            while (ret.second == false)
            {
                rand_t = rand() % pre_size;
                ret = indices.insert(rand_t);
            }
        }
        set<int>::iterator iter;
        int it = 0;
        for (iter = indices.begin(); iter != indices.end(); iter++)
        {
            Mat descriptor = Mat(1, 128, CV_32FC1, sift_res + (*iter) * 128);
            training_descriptors.push_back(descriptor);

            for (int k = 0; k < 128; k++)
            {
                final_res[(ROWS * i + it) * 128 + k] = sift_res[(*iter) * 128 + k];
            }
            for (int k = 0; k < 2; k++)
            {
                final_frame[(ROWS * i + it) * 2 + k] = sift_frame[(*iter) * 2 + k];
            }

            it++;
        }
        free(sift_res);
        free(sift_frame);
    }

    //////////////////////STEP 2  learn a GMM vocabulary
    int pre_size = training_descriptors.rows;
    numData = pre_size;
    // vl_twister
    VlRand *rand;
    rand = vl_get_rand();
    vl_rand_seed(rand, 1);

    VlGMM *gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, NUM_CLUSTERS);
    ///////////////////////WARNING: should set these parameters
    vl_gmm_set_initialization(gmm, VlGMMKMeans);
    // Compute V
    double denom = pre_size - 1;
    double xbar[128], V[128];
    for (int i = 0; i < dimension; i++)
    {
        xbar[i] = 0.0;
        for (int j = 0; j < numData; j++)
        {
            xbar[i] += (double)final_res[j * dimension + i];
        }
        xbar[i] /= (double)pre_size;
    }
    for (int i = 0; i < dimension; i++)
    {
        double absx = 0.0;
        for (int j = 0; j < numData; j++)
        {
            double tempx = (double)final_res[j * dimension + i] - xbar[i];
            absx += abs(tempx) * abs(tempx);
        }
        V[i] = absx / denom;
    }

    // Get max(V)
    double maxNum = V[0];
    for (int i = 1; i < dimension; i++)
    {
        if (V[i] > maxNum)
        {
            maxNum = V[i];
        }
    }
    maxNum = maxNum * 0.0001;
    vl_gmm_set_covariance_lower_bound(gmm, (double)maxNum);
    cout << "Lower bound " << maxNum << endl;
    vl_gmm_set_verbosity(gmm, 1);
    vl_gmm_set_max_num_iterations(gmm, 100);
    printf("vl_gmm: data type = %s\n", vl_get_type_name(vl_gmm_get_data_type(gmm)));
    printf("vl_gmm: data dimension = %d\n", dimension);
    printf("vl_gmm: num. data points = %d\n", numData);
    printf("vl_gmm: num. Gaussian modes = %d\n", NUM_CLUSTERS);
    printf("vl_gmm: lower bound on covariance = [");
    printf(" %f %f ... %f]\n",
           vl_gmm_get_covariance_lower_bounds(gmm)[0],
           vl_gmm_get_covariance_lower_bounds(gmm)[1],
           vl_gmm_get_covariance_lower_bounds(gmm)[dimension - 1]);

    double gmmres = vl_gmm_cluster(gmm, final_res, numData);
    free(final_res);
    free(final_frame);
    cout << "GMM ending cluster." << endl;

    // Rename to be cachePriors...etc
    priors = (TYPE *)vl_gmm_get_priors(gmm);
    means = (TYPE *)vl_gmm_get_means(gmm);
    covariances = (TYPE *)vl_gmm_get_covariances(gmm);
    cout << *covariances << endl;
    cout << "End of encoder " << endl;
    cout << "Training time " << wallclock() - start_time << endl;
    ///////////////END train encoder//////////
}

void loadParams()
{
    int dimension = DST_DIM + 2;
    priors = (TYPE *)vl_malloc(sizeof(float) * NUM_CLUSTERS);
    means = (TYPE *)vl_malloc(sizeof(float) * dimension * NUM_CLUSTERS);
    covariances = (TYPE *)vl_malloc(sizeof(float) * dimension * NUM_CLUSTERS);
    projection = (float *)malloc(128 * sizeof(float) * 80);
    projectionCenter = (float *)malloc(128 * sizeof(float));

    ifstream in1("params/priors", ios::in | ios::binary);
    in1.read((char *)priors, sizeof(float) * NUM_CLUSTERS);
    in1.close();

    ifstream in2("params/means", ios::in | ios::binary);
    in2.read((char *)means, sizeof(float) * dimension * NUM_CLUSTERS);
    in2.close();

    ifstream in3("params/covariances", ios::in | ios::binary);
    in3.read((char *)covariances, sizeof(float) * dimension * NUM_CLUSTERS);
    in3.close();

    ifstream in4("params/projection", ios::in | ios::binary);
    in4.read((char *)projection, sizeof(float) * 128 * 80);
    in4.close();

    ifstream in5("params/projectionCenter", ios::in | ios::binary);
    in5.read((char *)projectionCenter, sizeof(float) * 128);
    in5.close();
}

void freeParams()
{
    gpu_free();
    free(projection);
    free(projectionCenter);
    free(priors);
    free(means);
    free(covariances);
}

#define REQUEST 512
void distribution(int *order)
{
    const int nstars = REQUEST; // maximum number of stars to distribute

    std::default_random_engine generator;
    std::poisson_distribution<int> distribution(24.5);

    int p[50] = {};

    for (int i = 0; i < nstars; ++i)
    {
        int number = distribution(generator);
        if (number < 50)
            ++p[number];
    }
    for (int i = 0; i < 50; i++)
        cout << p[i] << " ";
    cout << endl
         << endl;

    int index = 0;
    for (int i = 0; i < 50; i++)
    {
        for (int j = 0; j < p[i]; j++)
        {
            order[index++] = i;
        }
    }

    for (int i = 0; i < REQUEST; i++)
        cout << order[i] << " ";
    cout << endl
         << endl;

    random_shuffle(order, order + REQUEST);
    for (int i = 0; i < REQUEST; i++)
        cout << order[i] << " ";
    cout << endl
         << endl;
}

void ThreadQueryFunction()
{
    Mat image = imread("/home/jacky/Desktop/mobile_ar_system/ar_server/data/demo/test/fantastic.jpg", IMREAD_GRAYSCALE);
    recognizedMarker marker;

    for (int i = 0; i < 10; i++)
    {
        double t0 = wallclock();
        query(image, marker);
        // this_thread::sleep_for(chrono::milliseconds(10));
        double t1 = wallclock();
        int mills = (t1 - t0) * 1000;
        totalTime += mills;
        this_thread::sleep_for(chrono::milliseconds(1000 - mills));
    }
}

void scalabilityTest()
{
    thread handlerThread[REQUEST];
    int order[REQUEST] = {0};
    distribution(order);

    for (int loop = 0; loop < 10; loop++)
    {
        totalTime = 0;
        int requestNum = pow(2, loop);
        int requestTable[50] = {0};
        int threadIndex = 0;

        for (int i = 0; i < requestNum; i++)
            requestTable[order[i]]++;

        for (int i = 0; i < 50; i++)
        {
            int curRequest = requestTable[i];

            for (int j = 0; j < curRequest; j++)
            {
                handlerThread[threadIndex++] = thread(ThreadQueryFunction);
            }
            this_thread::sleep_for(chrono::milliseconds(20));
        }

        for (int i = 0; i < threadIndex; i++)
            handlerThread[i].join();

        cout << endl
             << "===================================================================" << endl;
        cout << "average for loop " << loop << ": " << totalTime / 10.0 / threadIndex << " with thread #: " << threadIndex << endl;
        cout << "===================================================================" << endl
             << endl;
    }
}