#include "services.hpp"
#include "../reco.hpp"

extern queue<frame_buffer> frames;
extern queue<inter_service_buffer> inter_service_data;

inter_service_buffer lsh_processing(string service, int service_order, frame_buffer curr_frame)
{
    inter_service_buffer item;
    char tmp[4];

    string client_id = curr_frame.client_id;
    int frame_no = curr_frame.frame_no;
    int frame_data_type = curr_frame.data_type;
    int frame_size = curr_frame.buffer_size;

    string client_ip = curr_frame.client_ip;
    int client_port = curr_frame.client_port;
    char *frame_data = curr_frame.buffer;

    vector<float> enc_vec;

    memcpy(tmp, &(frame_data[0]), 4);
    int enc_size = *(int *)tmp;

    char *enc_vec_char = (char *)malloc(enc_size * 4);
    memcpy(enc_vec_char, &(frame_data[4]), 4 * enc_size);

    // looping through char array to convert data back into floats
    // at i = 0, index should begin at 4
    int data_index = 0;
    for (int i = 0; i < enc_size; i++)
    {
        memcpy(tmp, &(enc_vec_char[data_index]), 4);
        float *curr_float = (float *)tmp;
        enc_vec.push_back(*curr_float);

        data_index += 4;
    }
    auto results_returned = lsh_nn(enc_vec);

    charint results_size;
    results_size.i = get<0>(results_returned);
    int results_buffer_size = 4 * results_size.i; // size of char values

    char *results_vector = get<1>(results_returned);

    item.client_id = client_id;
    item.frame_no.i = frame_no;
    // item.data_type.i = MSG_DATA_TRANSMISSION;
    item.buffer_size.i = 4 + results_buffer_size;
    item.client_ip = client_ip;
    item.client_port.i = client_port;
    item.previous_service.i = service_order;

    item.buffer = (unsigned char *)malloc(4 + results_buffer_size);
    memset(item.buffer, 0, 4 + results_buffer_size);
    memcpy(&(item.buffer[0]), results_size.b, 4);
    memcpy(&(item.buffer[4]), results_vector, results_buffer_size);

    // copy sift data into item buffer
    item.sift_buffer_size.i = curr_frame.sift_buffer_size;
    item.sift_buffer = curr_frame.sift_buffer;

    print_log(service, string(client_id), to_string(frame_no), "Performed analysis on received 'encoding' data");
    return item;
}