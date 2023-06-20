#include "services.hpp"
#include "../reco.hpp"

extern queue<frame_buffer> frames;
extern queue<inter_service_buffer> inter_service_data;

void primary_processing(string service, int service_order, frame_buffer curr_frame, inter_service_buffer &results_frame)
{
    string client_id = curr_frame.client_id;
    int frame_no = curr_frame.frame_no;
    int frame_size = curr_frame.buffer_size;

    print_log(service, string(client_id), to_string(frame_no), "Image from Frame " + to_string(frame_no) +  " has been reduced from size " + to_string(frame_size) + " to a Mat object of size " + to_string(frame_size));

    results_frame.client_id = client_id;
    results_frame.frame_no.i = curr_frame.frame_no;
    results_frame.data_type.i = curr_frame.data_type;
    results_frame.buffer_size.i = curr_frame.buffer_size;
    results_frame.client_ip = curr_frame.client_ip;
    results_frame.client_port.i = curr_frame.client_port;
    results_frame.previous_service.i = service_order;
    results_frame.image_buffer = curr_frame.buffer;
}