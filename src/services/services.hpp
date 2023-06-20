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

void primary_processing(string service, int service_order, frame_buffer curr_frame, inter_service_buffer &results_frame);
void sift_processing(string service, int service_order, frame_buffer curr_frame, inter_service_buffer &results_frame);
void encoding_processing(string service, int service_order, frame_buffer curr_frame, inter_service_buffer &results_frame);
void lsh_processing(string service, int service_order, frame_buffer curr_frame, inter_service_buffer &results_frame);
void matching_processing(string service, int service_order, frame_buffer curr_frame, inter_service_buffer &results_frame);