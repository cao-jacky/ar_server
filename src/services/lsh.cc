#include "services.hpp"
#include "../reco.hpp"

extern queue<frame_buffer> frames;
extern queue<inter_service_buffer> inter_service_data;

void encoding_processing(string service, int service_order, frame_buffer curr_frame)
{
    cout << "LSH PROCESSING" << endl;
}