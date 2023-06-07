#include "services.hpp"
#include "../reco.hpp"

void encoding_processing(string service, int service_order, frame_buffer curr_frame)
{
    vector<float> test;

    memcpy(tmp, &(frame_data[0]), 4);
    int sift_result = *(int *)tmp;

    char *sift_resg = (char *)malloc(frame_size);
    memset(sift_resg, 0, frame_size);
    memcpy(sift_resg, &(frame_data[4]), frame_size);

    float *siftres = new float[128 * sift_result];
    // float *siftres = (float*)malloc(128*sift_result);

    int data_index = 0;
    for (int i = 0; i < sift_result * 128; i++)
    {
        memcpy(tmp, &(sift_resg[data_index]), 4);
        float curr_float = *(float *)tmp;
        siftres[i] = curr_float;

        data_index += 4;
    }

    char *encoded_vec = new char[4 * 5248];
    auto encoding_results = encoding(siftres, sift_result, test, false, &encoded_vec);

    charint encoded_size;
    encoded_size.i = get<0>(encoding_results);
    int encoding_buffer_size = 4 * encoded_size.i; // size of char values

    char *encoded_vector = get<1>(encoding_results);

    string client_id_string = client_id;
    string client_id_corr = client_id_string.substr(0, 4);
    char *client_id_ptr = &client_id_corr[0];

    item.client_id = client_id_ptr;
    item.frame_no.i = frame_no;
    item.data_type.i = MSG_DATA_TRANSMISSION;
    item.buffer_size.i = 4 + encoding_buffer_size;
    item.client_ip = client_ip;
    item.client_port.i = client_port;
    item.previous_service.i = service_value;
    item.buffer = new unsigned char[4 + encoding_buffer_size];
    memset(item.buffer, 0, strlen((char *)item.buffer) + 1);
    memcpy(&(item.buffer[0]), encoded_size.b, 4);
    memcpy(&(item.buffer[4]), encoded_vector, encoding_buffer_size);

    item.sift_ip = curr_frame.sift_ip;
    item.sift_port.i = curr_frame.sift_port;

    inter_service_data.push(item);
    print_log(service, string(client_id_ptr), to_string(frame_no), "Performed encoding on received 'sift' data");

    delete[] siftres;
    delete[] encoded_vec;

    free(frame_data);
    free(sift_resg);
}