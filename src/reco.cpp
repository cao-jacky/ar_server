#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "sys/times.h"
#include "reco.hpp"

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
#define MATCH_ONE_ONLY

int query_size_factor, nn_num;
float *means, *covariances, *priors, *projection_center, *projection;
vector<char *> whole_list;
vector<DenseVector<float>> lsh;
unique_ptr<LSHNearestNeighborTable<DenseVector<float>>> tablet;
unique_ptr<LSHNearestNeighborQuery<DenseVector<float>>> table;

void print_log(string service_name, string client_id, string frame_no, string message)
{
    cout << "{\\\"service_name\\\": \\\"" << service_name << "\\\", \\\"client_id\\\": \\\"" << client_id;
    cout << "\\\", \\\"frame_no\\\": \\\"" << frame_no << "\\\", \\\"timestamp\\\": \\\"" << setprecision(15) << wallclock() * 1000;
    cout << "\\\", \\\"message\\\": \\\"" << message << "\\\"}" << endl;
}

double wallclock(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int msec = tv.tv_usec / 1000;
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

bool mycompare(char *x, char *y)
{
    if (strcmp(x, y) <= 0)
        return 1;
    else
        return 0;
}

void load_images(vector<char *> online_images)
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

    for (int i = 0; i < online_images.size(); i++)
    {
        print_log("", "0", "0", "Online image: " + string(online_images[i]));
        whole_list.push_back(online_images[i]);
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
    print_log("", "0", "0", "Total images: " + to_string(whole_list.size()));

    closedir(d);
}

void load_params()
{
    int dimension = DST_DIM + 2;
    priors = (TYPE *)vl_malloc(sizeof(float) * NUM_CLUSTERS);
    means = (TYPE *)vl_malloc(sizeof(float) * dimension * NUM_CLUSTERS);
    covariances = (TYPE *)vl_malloc(sizeof(float) * dimension * NUM_CLUSTERS);
    projection = (float *)malloc(128 * sizeof(float) * 80);
    projection_center = (float *)malloc(128 * sizeof(float));

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

    ifstream in5("params/projection_center", ios::in | ios::binary);
    in5.read((char *)projection_center, sizeof(float) * 128);

    in5.close();
}

void free_params()
{
    gpu_free();
    free(projection);
    free(projection_center);
    free(priors);
    free(means);
    free(covariances);
}

tuple<int, float *> sift_gpu(Mat img, float **sift_res, float **sift_frame, SiftData &sift_data, int &w, int &h, bool online, bool is_color_image)
{
    CudaImage cimg;
    int num_pts;
    double start, finish, duration_gmm;

    // if(online) resize(img, img, Size(), 0.5, 0.5);
    if (is_color_image)
        cvtColor(img, img, COLOR_BGR2GRAY);
    img.convertTo(img, CV_32FC1);
    start = wallclock();
    w = img.cols;
    h = img.rows;
    print_log("", "0", "0", "Image size is (" + to_string(w) + "," + to_string(h) + ")");

    cimg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)img.data);
    cimg.Download();

    float initBlur = 1.0f;
    float thresh = 2.0f;
    InitSiftData(sift_data, 1000, true, true);
    ExtractSift(sift_data, cimg, 5, initBlur, thresh, 0.0f, false);

    num_pts = sift_data.numPts;
    *sift_res = (float *)malloc(sizeof(float) * 128 * num_pts);
    *sift_frame = (float *)malloc(sizeof(float) * 2 * num_pts);
    float *curr_res = *sift_res;
    float *curframe = *sift_frame;
    SiftPoint *p = sift_data.h_data;

    for (int i = 0; i < num_pts; i++)
    {
        memcpy(curr_res, p->data, (128 + 1) * sizeof(float));
        curr_res += 128;

        *curframe++ = p->xpos / w - 0.5;
        *curframe++ = p->ypos / h - 0.5;
        p++;
    }

    if (!online)
        FreeSiftData(sift_data); //

    finish = wallclock();
    duration_gmm = (double)(finish - start);
    print_log("", "0", "0", to_string(num_pts) + " SIFT points extracted in " + to_string(duration_gmm * 1000) + " ms");

    return make_tuple(num_pts, curr_res);
}

void onlineProcessing(Mat image, SiftData &sift_data, vector<float> &enc_vec, bool online, bool is_color_image, bool cache)
{
    double start, finish;
    double durationsift, duration_gmm;
    int sift_result;
    float *sift_resg;
    float *sift_frame;
    int height, width;

    float homography[9];
    int numMatches;

    float *dest;

    auto sift_gpu_results = sift_gpu(image, &sift_resg, &sift_frame, sift_data, width, height, online, is_color_image);
    sift_result = get<0>(sift_gpu_results);

    float enc[SIZE] = {0};

    if (cache)
    {
        start = wallclock();
        gpu_gmm_1(covariances, priors, means, NULL, NUM_CLUSTERS, 82, sift_result, (82 / 2.0) * log(2.0 * VL_PI), enc, NULL, sift_resg);
    }
    else
    {
        start = wallclock();
        dest = (float *)malloc(sift_result * 82 * sizeof(float));
        gpu_pca_mm(projection, projection_center, sift_resg, dest, sift_result, DST_DIM);

        finish = wallclock();
        duration_gmm = (double)(finish - start);
        print_log("", "0", "0", "PCA encoding time is " + to_string(duration_gmm));

        start = wallclock();
        gpu_gmm_1(covariances, priors, means, NULL, NUM_CLUSTERS, 82, sift_result, (82 / 2.0) * log(2.0 * VL_PI), enc, NULL, dest);
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
    duration_gmm = (double)(finish - start);
    print_log("", "0", "0", "Fisher Vector encoding time is " + to_string(duration_gmm));

    free(dest);
    free(sift_resg);
    free(sift_frame);
}

void encodeDatabase(int factor, int nn)
{
    query_size_factor = factor;
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
    print_log("", "0", "0", "LSH tables prepartion done");
}
