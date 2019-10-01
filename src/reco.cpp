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

#include "cudaImage.h"
#include "cuda_files.h"
#include "Eigen/Dense"
#include "falconn/lsh_nn_table.h"
extern "C" {
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
#define SIZE 5248 //82 * 2 * 32
#define ROWS 200
#define TYPE float
#define DST_DIM 80
#define NUM_HASH_TABLES 20
#define NUM_HASH_BITS 24
#define SUB_DATASET 110
//#define SUB_DATASET 1000
//#define FEATURE_CHECK
#define MATCH_ONE_ONLY
#define TEST

int querysizefactor, nn_num;
float *means, *covariances, *priors, *projectionCenter, *projection;
vector<char *> whole_list;
vector<SiftData> trainData;
vector<DenseVector<float>> lsh;
unique_ptr<LSHNearestNeighborTable<DenseVector<float>>> tablet;
unique_ptr<LSHNearestNeighborQuery<DenseVector<float>>> table;
vector<cacheItem> cacheItems;
atomic<int> totalTime;

double wallclock (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int msec = tv.tv_usec / 1000;
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

int sift_gpu(Mat img, float **siftres, float **siftframe, SiftData &siftData, int &w, int &h, bool online, bool isColorImage)
{
    CudaImage cimg;
    int numPts;
    double start, finish, durationgmm;

    //if(online) resize(img, img, Size(), 0.5, 0.5);
    if(isColorImage) cvtColor(img, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32FC1);
    start = wallclock();
    w = img.cols;
    h = img.rows;
    cout << "Image size = (" << w << "," << h << ")" << endl;

    cimg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)img.data);
    cimg.Download();

    float initBlur = 1.0f;
    float thresh = 2.0f;
    InitSiftData(siftData, 1000, true, true);
    ExtractSift(siftData, cimg, 5, initBlur, thresh, 0.0f, false);

    numPts = siftData.numPts;
    *siftres = (float *)malloc(sizeof(float)*128*numPts);
    *siftframe = (float *)malloc(sizeof(float)*2*numPts);
    float* curRes = *siftres;
    float* curframe = *siftframe;
    SiftPoint* p = siftData.h_data;

    for(int i = 0; i < numPts; i++) {
        memcpy(curRes, p->data, 128*sizeof(float));
        curRes+=128;

        *curframe++ = p->xpos/w - 0.5;
        *curframe++ = p->ypos/h - 0.5;
        p++;
    }

    if(!online) FreeSiftData(siftData);

    finish = wallclock();
    durationgmm = (double)(finish - start);
    cout << "sift res " << numPts << " with time: " << durationgmm << endl;
    return numPts;
}

void onlineProcessing(Mat image, SiftData &siftData, vector<float> &enc_vec, bool online, bool isColorImage)
{
    double start, finish;
    double durationsift, durationgmm;
    int siftResult;
    //cout << "Process " << image << endl;
    float *siftresg;
    float *siftframe;
    int height, width;

    siftResult = sift_gpu(image, &siftresg, &siftframe, siftData, width, height, online, isColorImage);

    start = wallclock();

    float *dest = (float *)malloc(siftResult*82*sizeof(float));
    gpu_pca_mm(projection, projectionCenter, siftresg, dest, siftResult, DST_DIM);

    finish = wallclock();
    durationgmm = (double)(finish - start);
    cout << "pca encoding time: " << durationgmm << endl;
    start = wallclock();

    float enc[SIZE] = {0};
    gpu_gmm_1(covariances, priors, means, NULL, NUM_CLUSTERS, 82, siftResult, (82/2.0)*log(2.0*VL_PI), enc, NULL, dest);

    ///////////WARNING: add the other NOOP
    float sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += enc[i] * enc[i];
    }
    for (int i = 0; i < SIZE; i++)
    {
        //WARNING: didn't use the max operation
        enc[i] /= sqrt(sum);
    }
    sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
      sum += enc[i] * enc[i];
    }
    for (int i = 0; i < SIZE; i++)
    {
        //WARNING: didn't use the max operation
        enc[i] /= sqrt(sum);
    }
  
    enc_vec = vector<float>(enc, enc+SIZE);

    finish = wallclock();
    durationgmm = (double)(finish - start);
    cout << "fisher encoding time: " << durationgmm << endl;

    free(dest);
    free(siftresg);
    free(siftframe);
}

void onlineCacheProcessing(Mat image, SiftData &siftData, vector<float> &enc_vec, bool online, bool isColorImage)
{
    int siftResult;
    float *siftresg;
    float *siftframe;
    int height, width;

    siftResult = sift_gpu(image, &siftresg, &siftframe, siftData, width, height, online, isColorImage);

    float enc[SIZE] = {0};
    gpu_gmm_1(covariances, priors, means, NULL, NUM_CLUSTERS, 128, siftResult, (128/2.0)*log(2.0*VL_PI), enc, NULL, siftresg);

    ///////////WARNING: add the other NOOP
    float sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += enc[i] * enc[i];
    }
    for (int i = 0; i < SIZE; i++)
    {
        //WARNING: didn't use the max operation
        enc[i] /= sqrt(sum);
    }
    sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
      sum += enc[i] * enc[i];
    }
    for (int i = 0; i < SIZE; i++)
    {
        //WARNING: didn't use the max operation
        enc[i] /= sqrt(sum);
    }
  
    enc_vec = vector<float>(enc, enc+SIZE);

    free(siftresg);
    free(siftframe);
}

void encodeDatabase(int factor, int nn)
{
    querysizefactor = factor;
    nn_num = nn;

    gpu_copy(covariances, priors, means, NUM_CLUSTERS, DST_DIM+2);

    vector<float> train[whole_list.size()];
    //Encode train files
    for (int i = 0; i < whole_list.size(); i++) {
        SiftData sData;
        Mat image = imread(whole_list[i], CV_LOAD_IMAGE_COLOR);
#ifdef TEST
        onlineProcessing(image, sData, train[i], true, true);
        if (i < 20) trainData.push_back(sData);
        else FreeSiftData(sData);
#else 
        onlineProcessing(image, sData, train[i], false, true);
#endif         
    }

    for(int i = 0; i < whole_list.size(); i++) {
        DenseVector<float> FV(SIZE);
        for(int j = 0; j < SIZE; j++) FV[j] = train[i][j];
        lsh.push_back(FV);
    }

    LSHConstructionParameters params = get_default_parameters<DenseVector<float>>((int_fast64_t)lsh.size(), (int_fast32_t)SIZE, DistanceFunction::EuclideanSquared, true);
    params.l = 32;
    params.k = 1;

    tablet = construct_table<DenseVector<float>>(lsh, params);
    table = tablet->construct_query_object(100);
    cout << "lsh tables preparing done" << endl;
}

void test()
{
    double dist;
    int correct = 0;
    double start, finish, duration;
    double t_s, t_f, t_d;
    SiftData tData[103];
    vector<float> test;
    vector<char *> test_list;

#if 1
    test_list.push_back("data/demo/test/aquaman.jpg");
    test_list.push_back("data/demo/test/fantastic.jpg");
    test_list.push_back("data/demo/test/smallfoot.jpg");
#endif

    const char *test_path = "data/crop";
    DIR *d = opendir(test_path);
    struct dirent *cur_dir;
    while ((cur_dir = readdir(d)) != NULL) {
        if ((strcmp(cur_dir->d_name, ".") != 0) && (strcmp(cur_dir->d_name, "..") != 0)) {
            char *file = new char[256];
            sprintf(file, "%s/%s", test_path, cur_dir->d_name);
            if (strstr(file, "jpg") != NULL)
                test_list.push_back(file);
        }
    }
    cout << endl << "-------------testing " << test_list.size() << " images---------------" <<endl << endl;
    closedir(d);

    t_s = wallclock();
    //for (int i = 0; i < test_list.size(); i++) {
    for (int i = 0; i < 20; i++) {
        start = wallclock();

        cout << endl << test_list[i] << endl;
        Mat image = imread(test_list[i], CV_LOAD_IMAGE_COLOR);
        onlineProcessing(image, tData[i], test, true, true);
        vector<int> result;
        DenseVector<float> t(SIZE);
        for(int j = 0; j < SIZE; j++) t[j] = test[j];
        table->find_k_nearest_neighbors(t, nn_num, &result);

        int flag = 0;
        for(int idx = 0; idx < result.size(); idx++) {
            cout << result[idx] << " ";
            if(result[idx] == i) {
                correct++;
                flag = 1;
                cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!>>" << endl;
                break;
            }
        } 

#ifdef TEST 
        if(flag) {
            MatchSiftData(trainData[i], tData[i]);
            float homography[9];
            int numMatches;
            FindHomography(trainData[i], homography, &numMatches, 10000, 0.00f, 0.85f, 5.0);
            int numFit = ImproveHomography(trainData[i], homography, 5, 0.00f, 0.80f, 2.0);
            //cout << "Matching features: " << numFit << " " << numMatches << endl;
            double ratio = 100.0f*numFit/min(trainData[i].numPts, tData[i].numPts);
            cout << "Matching features: " << numFit << " " << numMatches << " " << ratio << "% " << endl;

            if(ratio > 10) {
                Mat H(3, 3, CV_32FC1, homography);
    
                vector<Point2f> obj_corners(4), scene_corners(4);
                obj_corners[0] = cvPoint(0, 0);
                obj_corners[1] = cvPoint(330, 0);
                obj_corners[2] = cvPoint(330, 500);
                obj_corners[3] = cvPoint(0, 500);

                try {
                    perspectiveTransform(obj_corners, scene_corners, H);
                } catch (Exception) {
                    cout << "cv exception" << endl;
                    continue;
                }
            }
        }
#endif
        FreeSiftData(tData[i]);

        finish = wallclock();
        duration = (double)(finish - start);
        cout << "query time: " << duration << endl << endl;
    }
    t_f = wallclock();
    t_d = (double)(t_f - t_s)/20.0;
    cout << "correct: " <<correct << " in average time: " << t_d << endl;
}

bool query(Mat queryImage, recognizedMarker &marker)
{
    SiftData tData;
    vector<float> test;
    vector<int> result;
    DenseVector<float> t(SIZE);
    float homography[9];
    int numMatches;

    onlineProcessing(queryImage, tData, test, true, false);

    for(int j = 0; j < SIZE; j++) t[j] = test[j];
    table->find_k_nearest_neighbors(t, nn_num, &result);
    cout << "=====================time before matching: " << wallclock() << endl;

    for(int idx = 0; idx < result.size(); idx++) {
        cout << "testing " << result[idx] << endl;

#ifdef TEST
        //Mat image(741, 500, CV_8UC1);
        Mat image = imread(whole_list[result[idx]], CV_LOAD_IMAGE_COLOR);
        if(result[idx] >= 100) break;
        SiftData sData = trainData[result[idx]];
#else
        Mat image = imread(whole_list[result[idx]], CV_LOAD_IMAGE_COLOR);
        SiftData sData;
        int w, h;
        float *a, *b;
        sift_gpu(image, &a, &b, sData, w, h, true, true);
#endif
    
	cout << "features num: " << sData.numPts << " " << tData.numPts << endl;
        MatchSiftData(sData, tData);
        FindHomography(sData, homography, &numMatches, 10000, 0.00f, 0.85f, 5.0);
        int numFit = ImproveHomography(sData, homography, 5, 0.00f, 0.80f, 2.0);
        double ratio = 100.0f*numFit/min(sData.numPts, tData.numPts);
        cout << "Matching features: " << numFit << " " << numMatches << " " << ratio << "% " << endl;
#ifndef TEST
        FreeSiftData(sData);
#endif
       
        if(ratio > 10) {
            Mat H(3, 3, CV_32FC1, homography);

            vector<Point2f> obj_corners(4), scene_corners(4);
            obj_corners[0] = cvPoint(0, 0); 
            obj_corners[1] = cvPoint(image.cols, 0);
            obj_corners[2] = cvPoint(image.cols, image.rows); 
            obj_corners[3] = cvPoint(0, image.rows);

            try {
                perspectiveTransform(obj_corners, scene_corners, H);
            } catch (Exception) {
                cout << "cv exception" << endl;
                continue;
            }

            marker.markerID.i = result[idx];
            marker.height.i = image.rows;
            marker.width.i = image.cols;

            for (int i = 0; i < 4; i++) {
                marker.corners[i].x = scene_corners[i].x + RECO_W_OFFSET;
                marker.corners[i].y = scene_corners[i].y + RECO_H_OFFSET;
            }
            marker.markername = "gpu_recognized_image.";

            FreeSiftData(tData);
            cout<<"after matching "<<wallclock()<<endl;
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

    if(cacheItems.size() == 0) return false;

    onlineCacheProcessing(queryImage, tData, test, true, false);

    double minDistance = 999999999;
    int index = -1;
    for(int idx = 0; idx < cacheItems.size(); idx++) {
        double dis = norm(cacheItems[idx].fv, test);
        if(dis < minDistance) {
            minDistance = dis;
            index = idx;
        }
    }
    sData = cacheItems[index].data;
    cout << "=====================time before matching: " << wallclock() << endl;
    
    cout << "features num: " << sData.numPts << " " << tData.numPts << endl;
    MatchSiftData(sData, tData);
    FindHomography(sData, homography, &numMatches, 10000, 0.00f, 0.85f, 5.0);
    int numFit = ImproveHomography(sData, homography, 5, 0.00f, 0.80f, 2.0);
    double ratio = 100.0f*numFit/min(sData.numPts, tData.numPts);
    cout << "Matching features: " << numFit << " " << numMatches << " " << ratio << "% " << endl;
    FreeSiftData(tData);
       
    if(ratio > 10) {
        Mat H(3, 3, CV_32FC1, homography);

        vector<Point2f> obj_corners, scene_corners(4);
        for(int i = 0; i < 4; i++)
            obj_corners.push_back(cacheItems[index].curMarker.corners[i]);

        try {
            perspectiveTransform(obj_corners, scene_corners, H);
        } catch (Exception) {
            cout << "cv exception" << endl;
            return false;
        }

        marker.markerID.i = cacheItems[index].curMarker.markerID.i;
        marker.height.i = cacheItems[index].curMarker.height.i;
        marker.width.i = cacheItems[index].curMarker.width.i;

        for (int i = 0; i < 4; i++) {
            marker.corners[i].x = scene_corners[i].x + RECO_W_OFFSET;
            marker.corners[i].y = scene_corners[i].y + RECO_H_OFFSET;
        }
        marker.markername = "gpu_recognized_image.";

        return true; 
    } else {
        return false;
    }
}

void addCacheItem(frameBuffer curFrame, resBuffer curRes) 
{
    SiftData tData;
    vector<float> test;

    vector<uchar> imagedata(curFrame.buffer, curFrame.buffer + curFrame.bufferSize);
    Mat queryImage = imdecode(imagedata, CV_LOAD_IMAGE_GRAYSCALE);
    onlineCacheProcessing(queryImage, tData, test, true, false);

    recognizedMarker marker;
    int pointer = 0;
    memcpy(marker.markerID.b, &(curRes.buffer[pointer]), 4);
    pointer += 4;
    memcpy(marker.height.b, &(curRes.buffer[pointer]), 4);
    pointer += 4;
    memcpy(marker.width.b, &(curRes.buffer[pointer]), 4);
    pointer += 4;

    charfloat p;
    for(int j = 0; j < 4; j++) {
        memcpy(p.b, &(curRes.buffer[pointer]), 4);
        marker.corners[j].x = p.f;
        pointer+=4;
        memcpy(p.b, &(curRes.buffer[pointer]), 4);
        marker.corners[j].y = p.f;
        pointer+=4;
    }

    char name[40];
    memcpy(name, &(curRes.buffer[pointer]), 40);
    marker.markername = name;

    cacheItem newItem;
    newItem.fv = test;
    newItem.data = tData;
    newItem.curFrame = curFrame;
    newItem.curMarker = marker;
    cacheItems.push_back(newItem);
    cout<<"cache item inserted at "<<wallclock()<<endl<<endl;
}

bool mycompare(char* x, char* y) 
{
    if(strcmp(x, y)<=0) return 1;
    else return 0;
}

void loadImages(vector<char *> onlineImages) 
{
    gpu_init();

    const char *home = "data/bk_train"; 
    //const char *home = "/home/symlab/Downloads/dataset/paintings/shell"; 
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

    for (int i = 0; i < onlineImages.size(); i++) {
        cout<<"online image: "<<onlineImages[i]<<endl;
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
                    if(whole_list.size() == SUB_DATASET) break;
#endif
                }
            }
        }
        closedir(subd);
    }
    //sort(whole_list.begin(), whole_list.end(), mycompare);
    cout << endl << "---------------------in total " << whole_list.size() << " images------------------------" << endl << endl;
    closedir(d);
}

void trainParams() {
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
        //get descriptors
        cout << "Train file " << i << ": " << whole_list[i] << endl;
        //    if(count == 2)
        SiftData siftData;
        Mat image = imread(whole_list[i], CV_LOAD_IMAGE_COLOR);
        int pre_size = sift_gpu(image, &sift_res, &sift_frame, siftData, width, height, false, true);

#ifdef FEATURE_CHECK
        if(pre_size < ROWS) remove(whole_list[i]);
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
            Mat descriptor = Mat(1, 128, CV_32FC1, sift_res+(*iter)*128);
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

    PCA pca(training_descriptors, Mat(), CV_PCA_DATA_AS_ROW, 80);
    for (int i = 0; i < 128; i++)
    {
        projectionCenter[i] = pca.mean.at<float>(0,i);
    }
    for (int i = 0; i < 80; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            projection[i * 128 + j] = pca.eigenvectors.at<float>(i,j);
        }
    }
    cout<<"================pca training finished==================="<<endl;

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
    //vl_twister
    VlRand *rand;
    rand = vl_get_rand();
    vl_rand_seed(rand, 1);

    VlGMM *gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, NUM_CLUSTERS);
    ///////////////////////WARNING: should set these parameters
    vl_gmm_set_initialization(gmm, VlGMMKMeans);
    //Compute V
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

    //Get max(V)
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
    //cout << "Training time " << wallclock() - start_time << endl;
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

void trainCacheParams() {
    int numData;
    int dimension = 128;
    float *sift_res;
    float *sift_frame;

    float *final_res = (float *)malloc(ROWS * whole_list.size() * 128 * sizeof(float));
    float *final_frame = (float *)malloc(ROWS * whole_list.size() * 128 * sizeof(float));
    Mat training_descriptors(0, 128, CV_32FC1);
    //////////////////train encoder ////////////////
    //////// STEP 1: obtain sample image descriptors
    set<int>::iterator iter;
    double start_time = wallclock();

    for (int i = 0; i != whole_list.size(); ++i)
    {
        char imagename[256], imagesizename[256];
        int height, width;
        //get descriptors
        cout << "Train file " << i << ": " << whole_list[i] << endl;
        SiftData siftData;
        Mat image = imread(whole_list[i], CV_LOAD_IMAGE_COLOR);
        int pre_size = sift_gpu(image, &sift_res, &sift_frame, siftData, width, height, false, true);

#ifdef FEATURE_CHECK
        if(pre_size < ROWS) remove(whole_list[i]);
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
            Mat descriptor = Mat(1, 128, CV_32FC1, sift_res+(*iter)*128);
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
    //vl_twister
    VlRand *rand;
    rand = vl_get_rand();
    vl_rand_seed(rand, 1);

    VlGMM *gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, NUM_CLUSTERS);
    ///////////////////////WARNING: should set these parameters
    vl_gmm_set_initialization(gmm, VlGMMKMeans);
    //Compute V
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

    //Get max(V)
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

    priors = (TYPE *)vl_gmm_get_priors(gmm);
    means = (TYPE *)vl_gmm_get_means(gmm);
    covariances = (TYPE *)vl_gmm_get_covariances(gmm);
    cout << "End of encoder " << endl;
    cout << "Training time " << wallclock() - start_time << endl;
    ///////////////END train encoer//////////
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
void distribution(int *order) {
    const int nstars = REQUEST;   // maximum number of stars to distribute

    std::default_random_engine generator;
    std::poisson_distribution<int> distribution(24.5);

    int p[50]={};

    for (int i=0; i<nstars; ++i) {
        int number = distribution(generator);
        if (number<50) ++p[number];
    }
    for(int i = 0; i < 50; i++)
        cout<<p[i]<<" ";
    cout<<endl<<endl;

    int index = 0;
    for(int i = 0; i < 50; i++) {
        for(int j = 0; j < p[i]; j++) {
            order[index++] = i;
        }
    }

    for(int i = 0; i < REQUEST; i++)
        cout<<order[i]<<" ";
    cout<<endl<<endl;

    random_shuffle(order, order+REQUEST);
    for(int i = 0; i < REQUEST; i++)
        cout<<order[i]<<" ";
    cout<<endl<<endl;
}

void ThreadQueryFunction() {
    Mat image = imread("/home/symlab/Downloads/dataset/paintings/eval2/20190609_183259.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    recognizedMarker marker;

    for(int i = 0; i < 10; i++) {
        double t0 = wallclock();
        query(image, marker);
        //this_thread::sleep_for(chrono::milliseconds(10));
        double t1 = wallclock();
        int mills = (t1 - t0) * 1000;
        totalTime += mills;
        this_thread::sleep_for(chrono::milliseconds(1000-mills));
    }
}

void scalabilityTest() {
    thread handlerThread[REQUEST];
    int order[REQUEST] = {0};
    distribution(order);

    for(int loop = 0; loop < 10; loop++) {
        totalTime = 0;
        int requestNum = pow(2, loop);
        int requestTable[50] = {0};
        int threadIndex = 0;

        for(int i = 0; i < requestNum; i++)
            requestTable[order[i]]++;

        for(int i = 0; i < 50; i++) {
            int curRequest = requestTable[i];

            for(int j = 0; j < curRequest; j++) {
                handlerThread[threadIndex++] = thread(ThreadQueryFunction);
            }
            this_thread::sleep_for(chrono::milliseconds(20));
        }

        for(int i = 0; i < threadIndex; i++)
            handlerThread[i].join();

	cout<<endl<<"==================================================================="<<endl;
        cout<<"average for loop "<<loop<<": "<<totalTime/10.0/threadIndex<<" with thread #: "<<threadIndex<<endl;
	cout<<"==================================================================="<<endl<<endl;
    }
}

