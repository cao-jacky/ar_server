#include "services.hpp"
#include "../reco.hpp"

#include "../cuda_files.h"
extern "C"
{
#include "vl/generic.h"
#include "vl/gmm.h"
#include "vl/fisher.h"
#include "vl/mathop.h"
}

extern queue<frame_buffer> frames;
extern queue<inter_service_buffer> inter_service_data;

// defining constants
#define SIZE 5248 // 82 * 2 * 32
#define DST_DIM 80
#define NUM_CLUSTERS 32

extern float *projection;
extern float *projection_center;
extern float *covariances;
extern float *priors;
extern float *means;

tuple<int, char *> encoding(float *siftresg, int siftResult, vector<float> &enc_vec, bool cache, char **enc_vector)
{
    double start, finish;
    double durationsift, durationgmm;

    char *encoded_vector = *enc_vector;

    float enc[SIZE] = {0};

    start = wallclock();
    float *dest = (float *)malloc(siftResult * 82 * sizeof(float));
    gpu_pca_mm(projection, projection_center, siftresg, dest, siftResult, DST_DIM);

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

void encoding_processing(string service, int service_order, frame_buffer curr_frame, inter_service_buffer &results_frame)
{
    char tmp[4];

    string client_id = curr_frame.client_id;
    int frame_no = curr_frame.frame_no;
    int frame_data_type = curr_frame.data_type;
    int frame_size = curr_frame.buffer_size;

    string client_ip = curr_frame.client_ip;
    int client_port = curr_frame.client_port;
    char *frame_data = curr_frame.buffer;

    vector<float> test;

    memcpy(tmp, &(frame_data[0]), 4);
    int sift_result = *(int *)tmp;

    char *sift_resg = (char *)malloc(frame_size);
    memset(sift_resg, 0, frame_size);
    memcpy(sift_resg, &(frame_data[4]), frame_size);

    float *siftres = new float[128 * sift_result];

    int data_index = 0;
    for (int i = 0; i < sift_result * 128; i++)
    {
        memcpy(tmp, &(sift_resg[data_index]), 4);
        float curr_float = *(float *)tmp;
        siftres[i] = curr_float;

        data_index += 4;
    }

    char *encoded_vec = (char *)malloc(4 * 5248);
    memset(encoded_vec, 0, 4 * 5248);
    auto encoding_results = encoding(siftres, sift_result, test, false, &encoded_vec);

    charint encoded_size;
    encoded_size.i = get<0>(encoding_results);
    int encoding_buffer_size = 4 * encoded_size.i; // size of char values

    char *encoded_vector = get<1>(encoding_results);

    results_frame.client_id = client_id;
    results_frame.frame_no.i = frame_no;
    // results_frame.data_type.i = MSG_DATA_TRANSMISSION;
    results_frame.buffer_size.i = 4 + encoding_buffer_size;
    results_frame.client_ip = client_ip;
    results_frame.client_port.i = client_port;
    results_frame.previous_service.i = service_order;
    results_frame.buffer = (unsigned char *)malloc(4 + encoding_buffer_size);
    memset(results_frame.buffer, 0, 4 + encoding_buffer_size);

    // memset(results_frame.buffer, 0, strlen((char *)results_frame.buffer) + 1);
    memcpy(&(results_frame.buffer[0]), encoded_size.b, 4);
    memcpy(&(results_frame.buffer[4]), encoded_vector, encoding_buffer_size);

    // copy sift data into results_frame buffer
    results_frame.sift_buffer_size.i = curr_frame.sift_buffer_size;
    results_frame.sift_buffer = curr_frame.sift_buffer;

    print_log(service, client_id, to_string(frame_no), "Performed encoding on received sift data");

    delete[] siftres;
    // delete[] encoded_vec;

    // free(frame_data);
    free(encoded_vec);
    free(sift_resg);
}