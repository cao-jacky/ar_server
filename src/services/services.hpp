#include "../reco.hpp"

#include <iostream>
#include <queue>

using namespace std;

struct service_data
{
    string name;
    int order;
    string ip;
    int port;
    int udp_socket;
};

// primary
void client_echo(string service, frame_buffer curr_frame);
void client_preprocessing_request(string service, frame_buffer curr_frame, char* buffer);
inter_service_buffer primary_processing(string service, int service_order, frame_buffer curr_frame);
inter_service_buffer sift_processing(string service, int service_order, frame_buffer curr_frame);
inter_service_buffer encoding_processing(string service, int service_order, frame_buffer curr_frame);
inter_service_buffer lsh_processing(string service, int service_order, frame_buffer curr_frame);
inter_service_buffer matching_processing(string service, int service_order, frame_buffer curr_frame);