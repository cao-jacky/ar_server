#include "services.hpp"
#include "../reco.hpp"

extern queue<frame_buffer> frames;
extern queue<inter_service_buffer> inter_service_data;

void client_echo(string service, frame_buffer curr_frame)
{
    // if an echo message from the client
    string client_ip = curr_frame.client_ip;
    int client_port = curr_frame.client_port;

    print_log(service, string(curr_frame.client_id), "0", "Received an initial echo message from client with IP " + client_ip + " and port " + to_string(client_port));
}

frame_buffer client_preprocessing_request(string service, frame_buffer curr_frame, char *buffer)
{
    print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no), "Frame " + to_string(curr_frame.frame_no) + " received and has a filesize of " + to_string(curr_frame.buffer_size) + " Bytes");

    // copy frame image data into buffer
    curr_frame.buffer = (char *)malloc(curr_frame.buffer_size);
    memset(curr_frame.buffer, 0, curr_frame.buffer_size);
    memcpy(curr_frame.buffer, &(buffer[16]), curr_frame.buffer_size);

    print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no), "Added frame into internal buffer for processing");
    return curr_frame;
}

void primary_processing(string service, int service_order, frame_buffer curr_frame, inter_service_buffer &results_frame)
{
    string client_id = curr_frame.client_id;
    int frame_no = curr_frame.frame_no;
    int frame_size = curr_frame.buffer_size;

    print_log(service, string(client_id), to_string(frame_no), "Image from Frame has been reduced from size " + to_string(frame_size) + " to a Mat object of size " + to_string(frame_size));

    results_frame.client_id = client_id;
    results_frame.frame_no.i = curr_frame.frame_no;
    results_frame.data_type.i = curr_frame.data_type;
    results_frame.buffer_size.i = curr_frame.buffer_size;
    results_frame.client_ip = curr_frame.client_ip;
    results_frame.client_port.i = curr_frame.client_port;
    results_frame.previous_service.i = service_order;
    results_frame.image_buffer = curr_frame.buffer;
}