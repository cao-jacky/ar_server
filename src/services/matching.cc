#include "services.hpp"
#include "../reco.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;

extern queue<frame_buffer> frames;
extern queue<inter_service_buffer> inter_service_data;
extern queue<inter_service_buffer> results_frames;

#define BOUNDARY 3

extern vector<char *> whole_list;

void siftdata_reconstructor(string service, char *sd_char_array, SiftData reconstructed_sift_data, frame_buffer curr_frame)
{
    SiftData reconstructed_data;

    string client_id = curr_frame.client_id;
    int frame_no = curr_frame.frame_no;

    char tmp[4];
    int curr_posn = 0;

    memcpy(tmp, &(sd_char_array[curr_posn]), 4);
    int sd_num_pts = *(int *)tmp;
    reconstructed_data.numPts = sd_num_pts;
    curr_posn += 4;

    memcpy(tmp, &(sd_char_array[curr_posn]), 4);
    reconstructed_data.maxPts = *(int *)tmp;
    curr_posn += 4;

    SiftPoint *cpu_data = (SiftPoint *)calloc(sd_num_pts, sizeof(SiftPoint));

    for (int i = 0; i < sd_num_pts; i++)
    {
        SiftPoint *curr_data = (&cpu_data[i]);

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->xpos = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->ypos = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->scale = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->sharpness = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->edgeness = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->orientation = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->score = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->ambiguity = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->match = *(int *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->match_xpos = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->match_ypos = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->match_error = *(float *)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->subsampling = *(float *)tmp;
        curr_posn += 4;

        // re-creating the empty array
        for (int j = 0; j < 3; j++)
        {
            memcpy(tmp, &(sd_char_array[curr_posn]), 4);
            curr_data->empty[j] = *(float *)tmp;
            curr_posn += 4;
        }

        for (int k = 0; k < 128; k++)
        {
            memcpy(tmp, &(sd_char_array[curr_posn]), 4);
            curr_data->data[k] = *(float *)tmp;
            curr_posn += 4;
        }
    }

    reconstructed_data.h_data = cpu_data; // inserting data into reconstructed data structure
    reconstructed_sift_data = reconstructed_data;
    print_log(service, client_id, to_string(frame_no), "SiftData has been reconstructed from sift service, example data: " + to_string(sd_num_pts) + " SIFT points");
}

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

bool matching(string service, vector<int> result, SiftData &tData, recognizedMarker &marker, frame_buffer curr_frame)
{
    float homography[9];
    int numMatches;

    string client_id = curr_frame.client_id;
    int frame_no = curr_frame.frame_no;

    for (int idx = 0; idx < result.size(); idx++)
    {
        print_log(service, client_id, to_string(frame_no), "Testing " + to_string(result[idx]) + " " + whole_list[result[idx]]);

        Mat image = imread(whole_list[result[idx]], IMREAD_COLOR);
        SiftData sData;
        int w, h;
        float *a, *b;
        sift_gpu(image, &a, &b, sData, w, h, true, true);

        print_log(service, client_id, to_string(frame_no), "Number of feature points: " + to_string(sData.numPts) + " " + to_string(tData.numPts));
        MatchSiftData(sData, tData);
        FindHomography(sData, homography, &numMatches, 10000, 0.00f, 0.85f, 5.0);
        int numFit = ImproveHomography(sData, homography, 5, 0.00f, 0.80f, 2.0);
        double ratio = 100.0f * numFit / min(sData.numPts, tData.numPts);
        print_log(service, client_id, to_string(frame_no), "Matching features: " + to_string(numFit) + " " + to_string(numMatches) + " " + to_string(ratio) + "% ");

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
            print_log(service, client_id, to_string(frame_no), "Recognised object(s)");            
            return true;
        }
        else
        {
            print_log(service, client_id, to_string(frame_no), "No matching objects");
            return false;
        }
    }
    
}

void matching_processing(string service, int service_order, frame_buffer curr_frame)
{
    int recognised_marker_id;
    char tmp[4];
    SiftData reconstructed_sift_data;

    recognizedMarker marker;
    bool marker_detected = false;
    inter_service_buffer curRes;

    string client_id = curr_frame.client_id;
    int frame_no = curr_frame.frame_no;
    int frame_data_type = curr_frame.data_type;
    int frame_size = curr_frame.buffer_size;

    string client_ip = curr_frame.client_ip;
    int client_port = curr_frame.client_port;
    char *frame_data = curr_frame.buffer;

    char *sift_data = curr_frame.sift_buffer;

    vector<int> result;

    memcpy(tmp, &(frame_data[0]), 4);
    int result_size = *(int *)tmp;

    char *results_char = new char[result_size * 4];
    memset(results_char, 0, result_size * 4);
    memcpy(results_char, &(frame_data[4]), result_size * 4);

    int data_index = 0;
    for (int i = 0; i < result_size; i++)
    {
        memcpy(tmp, &(results_char[data_index]), 4);
        int *curr_int = (int *)tmp;
        result.push_back(*curr_int);

        data_index += 4;
    }

    siftdata_reconstructor(service, sift_data, reconstructed_sift_data, curr_frame);
    marker_detected = matching(service, result, reconstructed_sift_data, marker, curr_frame);

    if (marker_detected)
    {
        curRes.client_id = client_id;
        curRes.frame_no.i = frame_no;
        // curRes.data_type.i = MSG_DATA_TRANSMISSION;
        curRes.buffer_size.i = 1;
        curRes.client_ip = client_ip;
        curRes.client_port.i = client_port;
        curRes.previous_service.i = BOUNDARY;

        curRes.results_buffer = new char[100 * curRes.buffer_size.i];

        int pointer = 0;
        memcpy(&(curRes.results_buffer[pointer]), marker.markerID.b, 4);
        pointer += 4;
        memcpy(&(curRes.results_buffer[pointer]), marker.height.b, 4);
        pointer += 4;
        memcpy(&(curRes.results_buffer[pointer]), marker.width.b, 4);
        pointer += 4;

        charfloat p;
        for (int j = 0; j < 4; j++)
        {
            p.f = marker.corners[j].x;
            memcpy(&(curRes.results_buffer[pointer]), p.b, 4);
            pointer += 4;
            p.f = marker.corners[j].y;
            memcpy(&(curRes.results_buffer[pointer]), p.b, 4);
            pointer += 4;
        }

        memcpy(&(curRes.results_buffer[pointer]), marker.markername.data(), marker.markername.length());

        recognised_marker_id = marker.markerID.i;

        print_log(service, client_id, to_string(frame_no), "matching analysis is complete, will pass to client forwarder");
    }
    else
    {
        curRes.frame_no.i = frame_no;
        curRes.buffer_size.i = 0;
    }
    results_frames.push(curRes);
}