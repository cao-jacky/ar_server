#include "services.hpp"
#include "../reco.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;

extern queue<frame_buffer> frames;
extern queue<inter_service_buffer> inter_service_data;

void export_siftdata(SiftData &data_struct, char **sift_data_array)
{
    int spf = 15; // sift point features number

    int num_points = data_struct.numPts;
    int max_points = data_struct.maxPts;

    SiftPoint *cpu_data = data_struct.h_data;

    int sd_size = num_points * (4 * (spf + 3 + 128));

    *sift_data_array = (char *)malloc(sd_size);
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
}

tuple<float *> services_sift_gpu(Mat img, float **siftres, float **siftframe, SiftData &siftData, int &w, int &h, bool online, bool isColorImage, int &num_points, char **raw_sift_data)
{
    CudaImage cimg;
    double start, finish, durationgmm;

    // if (isColorImage)
    //     cvtColor(img, img, COLOR_BGR2GRAY);
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

    export_siftdata(siftData, raw_sift_data);

    if (!online)
        FreeSiftData(siftData);

    finish = wallclock();
    durationgmm = (double)(finish - start);
    print_log("", "0", "0", to_string(num_points) + " SIFT points extracted in " + to_string(durationgmm * 1000) + " ms");

    return make_tuple(curRes);
}

void sift_analysis(int &sift_points, char **sift_data_buffer, char **raw_sift_data, Mat image, SiftData &siftData)
{
    float *siftresg;
    float *siftframe;
    int height, width;
    float *curr_res;

    auto sift_gpu_results = services_sift_gpu(image, &siftresg, &siftframe, siftData, width, height, true, false, sift_points, raw_sift_data);

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

void sift_processing(string service, int service_order, frame_buffer curr_frame)
{
    SiftData tData;
    float sift_array[2];
    int sift_result;
    float sift_resg;
    int height, width;
    vector<float> test;

    int sift_points;        // number of SIFT points
    char *sift_data_buffer; // raw data of the SIFT analysis
    char *raw_sift_data;    // raw SIFT data needed by matching

    // selecting out data
    string client_id = curr_frame.client_id;

    int frame_no = curr_frame.frame_no;
    int frame_data_type = curr_frame.data_type;
    int frame_size = curr_frame.buffer_size;

    string client_ip = curr_frame.client_ip;
    int client_port = curr_frame.client_port;
    char *frame_data = curr_frame.buffer;

    vector<uchar> imgdata(frame_data, frame_data + frame_size);
    Mat img_scene = imdecode(imgdata, IMREAD_GRAYSCALE);
    // imwrite("query.jpg", img_scene);
    Mat detect = img_scene(Rect(RECO_W_OFFSET, RECO_H_OFFSET, 160, 270));

    sift_analysis(sift_points, &sift_data_buffer, &raw_sift_data, detect, tData);

    char tmp[4];
    memcpy(tmp, &(raw_sift_data[0]), 4);
    int sd_num_pts = *(int *)tmp;

    charint siftresult;
    siftresult.i = sift_points;
    int sift_buffer_size = 128 * 4 * siftresult.i; // size of char values

    inter_service_buffer item;

    // push data required for next service
    item.client_id = client_id;
    item.frame_no.i = frame_no;
    // item.data_type.i = MSG_SIFT_TO_ENCODING;
    item.buffer_size.i = 4 + sift_buffer_size;
    item.client_ip = client_ip;
    item.client_port.i = client_port;
    item.previous_service.i = service_order;

    // item.buffer = new unsigned char[4 + sift_buffer_size];
    item.buffer = (unsigned char *)malloc(4 + sift_buffer_size);
    // memset(item.buffer, 0, 4 + sift_buffer_size);
    memset(item.buffer, 0, 4 + sift_buffer_size);
    memcpy(&(item.buffer[0]), siftresult.b, 4);
    memcpy(&(item.buffer[4]), sift_data_buffer, sift_buffer_size);

    // storing SIFT data for retrieval by the matching service
    int sift_data_size = 4 * siftresult.i * (15 + 3 + 128); // taken from export_siftdata

    print_log(service, client_id, to_string(frame_no), "Expected size of SIFT data buffer to pack for Frame " + to_string(frame_no) + " is " + to_string(sift_data_size) + " Bytes");

    item.sift_buffer_size.i = sift_data_size;
    item.sift_buffer = raw_sift_data;

    inter_service_data.push(item);

    FreeSiftData(tData);
    free(frame_data);
    free(sift_data_buffer);
}