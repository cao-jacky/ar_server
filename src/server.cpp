#include "reco.hpp"
#include "cuda_files.h"

#include <map>

#include <opencv2/opencv.hpp>

#include <nlohmann/json.hpp>

#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <chrono>
#include <thread>
#include <iomanip>
#include <queue>
#include <fstream>
#include <tuple>
#include <math.h>

#include <numeric>

#include <errno.h>
#include <cerrno>

#define MAIN_PORT 50000

#define MESSAGE_ECHO 0
#define MESSAGE_REGISTER 1
#define MESSAGE_NEXT_SERVICE_IP 2
#define DATA_TRANSMISSION 3
#define CLIENT_REGISTRATION 4
#define SIFT_TO_MATCHING 5

#define BOUNDARY 3

// message_type definitions
#define MSG_ECHO 0
#define MSG_SERVICE_REGISTER 11
#define MSG_PRIMARY_SERVICE 12
#define MSG_DATA_TRANSMISSION 13
#define MSG_MATCHING_SIFT 14
#define MSG_SIFT_TO_ENCODING 15
#define MSG_CLIENT_FRAME_DETECT 2

#define PACKET_SIZE 60000
#define MAX_PACKET_SIZE 60000
#define RES_SIZE 512
//#define TRAIN
#define UDP

using namespace std;
using namespace cv;
using json = nlohmann::json;

using namespace std::this_thread; // sleep_for, sleep_until
using namespace std::chrono;      // nanoseconds, system_clock, seconds

struct sockaddr_in localAddr;
struct sockaddr_in remoteAddr;
struct sockaddr_in main_addr;
struct sockaddr_in next_service_addr;
struct sockaddr_in sift_rec_addr;
struct sockaddr_in sift_rec_remote_addr;
struct sockaddr_in matching_rec_addr;
struct sockaddr_in matching_addr;
struct sockaddr_in client_addr;
struct sockaddr_in sift_to_matching_addr;

socklen_t addrlen = sizeof(remoteAddr);
socklen_t mrd_len = sizeof(matching_rec_addr);
bool isClientAlive = false;

queue<frame_buffer> frames, offloadframes;
queue<resBuffer> results;
int recognizedMarkerID;

vector<char *> onlineImages;
vector<char *> onlineAnnotations;

SiftData reconstructed_data;
matchingSiftItem receivedSiftData;
bool isSiftReconstructed = false;

// declaring variables needed for distributed operation
string service;
int service_value;
queue<inter_service_buffer> inter_service_data;

string next_service;

json sift_buffer_details;
deque<sift_data_item> sift_items;
int sbd_max = 100;

json matching_buffer_details;
deque<matching_item> matching_items;
int mbd_max = 100;

// hard coding the maps for each service, nothing clever needed about this
std::map<string, int> service_map = {
    {"primary", 1},
    {"sift", 2},
    {"encoding", 3},
    {"lsh", 4},
    {"matching", 5}};

std::map<int, string> service_map_reverse = {
    {1, "primary"},
    {2, "sift"},
    {3, "encoding"},
    {4, "lsh"},
    {5, "matching"}};

std::map<string, string> registered_services;

// json services = {
//    {"primary", {"10.30.100.1", "50001"}},
//      {"sift", {"10.30.101.1", "50002"}},
//      {"encoding", {"10.30.102.1", "50003"}},
//      {"lsh", {"10.30.103.1", "50004"}},
//      {"matching", {"10.30.104.1", "50005"}}};

json services = {
   {"primary", {"35.157.216.194", "50001"}},
   {"sift", {"35.157.216.194", "50002"}},
   {"encoding", {"35.157.216.194", "50003"}},
   {"lsh", {"35.157.216.194", "50004"}},
   {"matching", {"35.157.216.194", "50005"}}};

json services_primary_knowledge;

json services_outline = {
    {"name_val", {{"primary", 1}, {"sift", 2}, {"encoding", 3}, {"lsh", 4}, {"matching", 5}}},
    {"val_name", {{"1", "primary"}, {"2", "sift"}, {"3", "encoding"}, {"4", "lsh"}, {"5", "matching"}}}};

char *ip_to_bytes(char *client_ip)
{
    unsigned short a, b, c, d;
    sscanf(client_ip, "%hu.%hu.%hu.%hu", &a, &b, &c, &d);

    char *ip_buffer = new char[16];
    memset(ip_buffer, 0, sizeof(ip_buffer));

    charint ib_a;
    ib_a.i = (int)a;
    memcpy(ip_buffer, ib_a.b, 4);

    charint ib_b;
    ib_b.i = (int)b;
    memcpy(&(ip_buffer[4]), ib_b.b, 4);

    charint ib_c;
    ib_c.i = (int)c;
    memcpy(&(ip_buffer[8]), ib_c.b, 4);

    charint ib_d;
    ib_d.i = (int)d;
    memcpy(&(ip_buffer[12]), ib_d.b, 4);

    return ip_buffer;
}

char *bytes_to_ip(char *client_ip)
{
    char tmp[4];

    memcpy(tmp, &(client_ip[0]), 4);
    int ib_a = *(int *)tmp;

    memcpy(tmp, &(client_ip[4]), 4);
    int ib_b = *(int *)tmp;

    memcpy(tmp, &(client_ip[8]), 4);
    int ib_c = *(int *)tmp;

    memcpy(tmp, &(client_ip[12]), 4);
    int ib_d = *(int *)tmp;

    string final_ip_string = to_string(ib_a) + "." + to_string(ib_b) + "." + to_string(ib_c) + "." + to_string(ib_d);

    char *final_ip = new char[final_ip_string.length()];
    strcpy(final_ip, final_ip_string.c_str());

    return final_ip;
}

void registerService(int sock)
{
    charint register_id;
    register_id.i = service_value; // ID of service registering itself

    charint message_type;
    message_type.i = MSG_SERVICE_REGISTER; // message to primary to register service

    char registering[16];
    memcpy(&(registering[8]), message_type.b, 4);
    memcpy(&(registering[12]), register_id.b, 4);

    int udp_status = sendto(sock, registering, sizeof(registering), 0, (struct sockaddr *)&main_addr, sizeof(main_addr));
    if (udp_status == -1)
    {
        cout << "Error sending: " << strerror(errno) << endl;
    }
    print_log(service, "0", "0", "Service " + string(service) + " is attempting to register with the primary service");
}

void *ThreadUDPReceiverFunction(void *socket)
{
    print_log(service, "0", "0", "UDP receiver thread created");

    char tmp[4];
    char buffer[60 + PACKET_SIZE];
    int sock = *((int *)socket);

    char *sift_res_buffer;
    int sift_res_buffer_size;

    int curr_recv_packet_no;
    int prev_recv_packet_no = 0;
    int total_packets_no;

    int last_frame_no = 0;

    int packet_tally;
    int sift_data_count;

    char *results_buffer;

    char *previous_client;

    if (service != "primary")
    {
        registerService(sock); // when first called, try to register with primary service
    }

    while (1)
    {
        memset(buffer, 0, sizeof(buffer));
        recvfrom(sock, buffer, PACKET_SIZE, 0, (struct sockaddr *)&remoteAddr, &addrlen);

        char device_id[4];
        char client_id[4];
        char *device_ip = inet_ntoa(remoteAddr.sin_addr);
        int device_port = htons(remoteAddr.sin_port);

        // copy client frames into frames buffer if main service
        frame_buffer curr_frame;
        memcpy(client_id, buffer, 4);
        memcpy(device_id, buffer, 4);

        curr_frame.client_id = (char *)device_id;

        //char curr_client[4];
        //memcpy(&(curr_client[0]), device_id, 4);
        char *curr_client = (char*)device_id;

        memcpy(tmp, &(buffer[4]), 4);
        curr_frame.frame_no = *(int *)tmp;

        memcpy(tmp, &(buffer[8]), 4);
        curr_frame.data_type = *(int *)tmp;

        memcpy(tmp, &(buffer[12]), 4);
        curr_frame.buffer_size = *(int *)tmp;

        if (service == "primary")
        {
            if (curr_frame.data_type == MSG_ECHO)
            {
                // if an echo message from the client
                string device_ip_print = device_ip;
                print_log(service, string(curr_frame.client_id), "0", "Received an echo message from client with IP " + device_ip_print + " and port " + to_string(device_port));
                charint echoID;
                echoID.i = curr_frame.frame_no;
                char echo[4];
                memcpy(echo, echoID.b, 4);
                sendto(sock, echo, sizeof(echo), 0, (struct sockaddr *)&remoteAddr, addrlen);
                print_log(service, string(curr_frame.client_id), "0", "Sent an echo reply");

                int client_ip_strlen = strlen(device_ip);

                char client_registration[16 + client_ip_strlen];

                charint client_reg_frame_no;
                client_reg_frame_no.i = 0; // no frame
                memcpy(&(client_registration[0]), client_reg_frame_no.b, 4);

                charint client_reg_id;
                client_reg_id.i = CLIENT_REGISTRATION;
                memcpy(&(client_registration[4]), client_reg_id.b, 4);

                charint device_ip_len;
                device_ip_len.i = client_ip_strlen;
                memcpy(&(client_registration[12]), device_ip_len.b, 4);
                memcpy(&(client_registration[16]), device_ip, client_ip_strlen);

                continue;
            }
            else if (curr_frame.data_type == MSG_SERVICE_REGISTER)
            {
                // when a service comes online it tells primary and "registers"
                string service_id;

                // at position 12 of the frame there is the ID value of the service to be registered
                int service_val = curr_frame.buffer_size;
                json val_names = (services_outline["val_name"]);

                auto so_service = val_names.find(to_string(service_val));
                string service_to_register = *so_service;
                string service_ip = device_ip;
                services_primary_knowledge[service_to_register] = {device_ip, device_port};
                print_log(service, "0", "0", "Service " + service_to_register + " on IP " + device_ip + " has come online and registered with primary");

                registered_services.insert({service_to_register, device_ip});

                charint message_type;
                message_type.i = MSG_SERVICE_REGISTER; // message to primary to register service

                charint register_status;
                register_status.i = 1;

                char registering[16];
                memcpy(&(registering[8]), message_type.b, 4);
                memcpy(&(registering[12]), register_status.b, 4);

                sendto(sock, registering, sizeof(registering), 0, (struct sockaddr *)&remoteAddr, sizeof(remoteAddr));
                print_log(service, "0", "0", "Sending confirmation to service " + service_to_register + " that service is now registered with primary");
            }
            else if (curr_frame.data_type == MSG_CLIENT_FRAME_DETECT)
            {
                print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no),
                          "Frame " + to_string(curr_frame.frame_no) + " received and has a filesize of " +
                              to_string(curr_frame.buffer_size) + " Bytes");

                // copy frame image data into buffer
                curr_frame.buffer = (char *)malloc(curr_frame.buffer_size);
                memset(curr_frame.buffer, 0, curr_frame.buffer_size);
                memcpy(curr_frame.buffer, &(buffer[16]), curr_frame.buffer_size);

                // copy client ip and port into buffer
                curr_frame.client_ip = device_ip;
                curr_frame.client_port = device_port;

                frames.push(curr_frame);
            }
        }
        else
        {
            if (curr_frame.data_type == MSG_SERVICE_REGISTER)
            {
                int primary_service_status = curr_frame.buffer_size;
                if (primary_service_status != 1)
                {
                    break;
                }

                print_log(service, "0", "0", "Received confirmation from primary that this service is now logged as online and active");
            }
            else if (curr_frame.data_type == MSG_DATA_TRANSMISSION)
            {
                // performing logic to check that received data is supposed to be sent on
                memcpy(tmp, &(buffer[36]), 4);
                int previous_service_val = *(int *)tmp;
                if (previous_service_val == service_value - 1)
                {
                    // if the data received is from previous service, proceed
                    // with copying out the data
                    char tmp_ip[16];
                    memcpy(tmp_ip, &(buffer[16]), 16);
                    curr_frame.client_ip = (char *)tmp_ip;
                    
                    memcpy(tmp, &(buffer[32]), 4);
                    curr_frame.client_port = *(int *)tmp;

                    // only copy if not sift
                    if (service_value != 2) {
                        // copy sift details out
                        char sift_tmp_ip[16];
                        memcpy(sift_tmp_ip, &(buffer[40]), 16);
                        curr_frame.sift_ip = (char *)sift_tmp_ip;
                        cout << "SIFT DATA1 " << curr_frame.sift_ip << endl;

                        memcpy(tmp, &(buffer[56]), 4);
                        curr_frame.sift_port = *(int *)tmp;
                    }

                    // if matching service, proceed to request the corresponding data from sift
                    if (service_value == 5)
                    {
                        // copy sift details out
                        char sift_tmp_ip2[16];
                        memcpy(sift_tmp_ip2, &(buffer[40]), 16);
                        curr_frame.sift_ip = (char *)sift_tmp_ip2;

                        char* sift_ip = curr_frame.sift_ip;  
                        string sift_ip_string = sift_ip;  

                        json sift_ns = services["sift"];
                        string sift_port_string = sift_ns[1];
                        int sift_port = stoi(sift_port_string);

                        char ms_buffer[12];
                        memset(ms_buffer, 0, sizeof(ms_buffer));
                        memcpy(ms_buffer, curr_frame.client_id, 4);
                        
                        string client_id_string = curr_frame.client_id;
                        string client_id_corr = client_id_string.substr(0, 4);
                        curr_frame.client_id = &client_id_corr[0];

                        //memcpy(&(ms_buffer[0]), client_id, sizeof(client_id));

                        charint matching_sift_fno;
                        matching_sift_fno.i = curr_frame.frame_no;
                        memcpy(&(ms_buffer[4]), matching_sift_fno.b, 4);

                        charint matching_sift_id;
                        matching_sift_id.i = MSG_MATCHING_SIFT;
                        memcpy(&(ms_buffer[8]), matching_sift_id.b, 4);

                        inet_pton(AF_INET, sift_ip_string.c_str(), &(sift_rec_addr.sin_addr));
                        sift_rec_addr.sin_port = htons(sift_port);

                        int udp_status = sendto(sock, ms_buffer, sizeof(ms_buffer), 0, (struct sockaddr *)&sift_rec_addr, sizeof(sift_rec_addr));
                        
                        print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no), "Requested data from sift service with details " +  sift_ip_string + " " + to_string(sift_port) + ", request packet has size " + to_string(udp_status));
                    }
                    // curr_frame.buffer = new char[curr_frame.buffer_size];
                    curr_frame.buffer = (char *)malloc(curr_frame.buffer_size);
                    memset(curr_frame.buffer, 0, curr_frame.buffer_size);
                    memcpy(curr_frame.buffer, &(buffer[60]), curr_frame.buffer_size);

                    frames.push(curr_frame);
                }
                print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no),
                          "Received data from '" + services_outline["val_name"][to_string(previous_service_val)] + "' service and will now proceed with analysis");
            }
            else if (curr_frame.data_type == CLIENT_REGISTRATION)
            {
                memcpy(tmp, &(buffer[8]), 4);
                int client_port = *(int *)tmp;

                memcpy(tmp, &(buffer[12]), 4);
                int client_ip_len = *(int *)tmp;

                char client_ip_tmp[client_ip_len];
                memcpy(client_ip_tmp, &(buffer[16]), client_ip_len + 1);

                // creating client object to return data to
                inet_pton(AF_INET, client_ip_tmp, &(client_addr.sin_addr));
                client_addr.sin_port = htons(client_port);
                print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no),
                          "Received client registration details from main of IP '" + string(client_ip_tmp) + "' and port " + to_string(client_port));
            }
            else if (curr_frame.data_type == MSG_SIFT_TO_ENCODING)
            {
                char tmp_ip[16];
                memcpy(tmp_ip, &(buffer[16]), 16);
                curr_frame.client_ip = (char *)tmp_ip;

                memcpy(tmp, &(buffer[32]), 4);
                curr_frame.client_port = *(int *)tmp;

                memcpy(tmp, &(buffer[44]), 4);
                int curr_packet_no = *(int *)tmp;

                memcpy(tmp, &(buffer[40]), 4);
                total_packets_no = *(int *)tmp;

                // store the sift IP and port details 
                curr_frame.sift_ip = device_ip;
                curr_frame.sift_port = device_port;

                memcpy(tmp, &(buffer[12]), 4);
                sift_res_buffer_size = *(int *)tmp;

                // logic to check for if first packet from a client or if the client does not match the previous
                if (curr_packet_no == 0 || curr_client != previous_client)
                {
                    // if the first packet received
                    print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no),
                              "Receiving SIFT data in packets for Frame " + to_string(curr_frame.frame_no) +
                                  " with an expected total number of packets of " + to_string(total_packets_no) +
                                  " and total bytes of " + to_string(sift_res_buffer_size));

                    sift_res_buffer = new char[sift_res_buffer_size];
                    memset(sift_res_buffer, 0, sift_res_buffer_size);
                    packet_tally = 0;
                    sift_data_count = 0;
                    last_frame_no = curr_frame.frame_no;
                }

                if (curr_client == previous_client) {
                    cout << last_frame_no << " " << curr_frame.frame_no << endl;
                    if (curr_frame.frame_no >= last_frame_no) {
                        int to_copy = MAX_PACKET_SIZE;
                        int copy_index = curr_packet_no * MAX_PACKET_SIZE;
                        if (curr_packet_no + 1 == total_packets_no)
                        {
                            to_copy = sift_res_buffer_size - sift_data_count;
                        }
                        memcpy(&(sift_res_buffer[copy_index]), &(buffer[48]), to_copy);
                        print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no),
                                    "For Frame " + to_string(curr_frame.frame_no) + " received packet with packet number of " + to_string(curr_packet_no));
                        packet_tally++;
                        sift_data_count += MAX_PACKET_SIZE;

                        if (curr_packet_no + 1 == total_packets_no)
                        {
                            // need to add logic to check whether all of the packets were received,
                            // and whether they were in the correct order

                            if (packet_tally == total_packets_no)
                            {
                                // curr_frame.buffer = new char[curr_frame.buffer_size];
                                curr_frame.buffer = (char *)malloc(curr_frame.buffer_size);
                                memset(curr_frame.buffer, 0, curr_frame.buffer_size);
                                memcpy(curr_frame.buffer, sift_res_buffer, curr_frame.buffer_size);

                                frames.push(curr_frame);
                                print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no),
                                            "All packets received for Frame " + to_string(curr_frame.frame_no) + " and will now pass the data to the encoding functions");
                            }
                        }
                        last_frame_no = curr_frame.frame_no;
                    }
                }
                previous_client = curr_client;
            }
            else if (curr_frame.data_type == MSG_MATCHING_SIFT)
            {
                // a request sent from matching to retrieve sift data
                print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no),
                          "Received request from 'matching' for sift data for frame " +
                              to_string(curr_frame.frame_no) + " and client " + string(curr_frame.client_id));

                // find where in the JSON array is the matching frame
                string frame_to_find = string(curr_frame.client_id) + "_" + to_string(curr_frame.frame_no);
                int sbd_val = 0;
                int sbd_loc;
                for (auto sbd_it : sift_buffer_details)
                {
                    // "it" is of type json::reference and has no key() member
                    if (sbd_it == frame_to_find)
                    {
                        sbd_loc = sbd_val;
                    }
                    sbd_val++;
                }

                // copy out the sift data from the deque
                sift_data_item msd = sift_items[sbd_loc];
                char *msd_client_id = msd.client_id;
                int msd_frame_no = msd.frame_no.i;
                float msd_data_size = (float)msd.sift_data_size.i;
                char *msd_data_buffer = msd.sift_data;

                double max_packets = ceil(msd_data_size / MAX_PACKET_SIZE);
                if (max_packets > 1)
                {
                    print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no),
                              "Packet payload of " + to_string((int)msd_data_size) + " will be greater than " + to_string(MAX_PACKET_SIZE) +
                                  " B, therefore, the data will be sent in " + to_string((int)max_packets) + " packets");

                    // preparing the buffer of the packets to be sent
                    char to_m_buffer[20 + MAX_PACKET_SIZE];
                    memset(to_m_buffer, 0, 20 + MAX_PACKET_SIZE);
                    memcpy(to_m_buffer, curr_frame.client_id, 4);

                    charint curr_frame_no;
                    curr_frame_no.i = msd_frame_no;
                    memcpy(&(to_m_buffer[4]), curr_frame_no.b, 4);

                    charint total_packets;
                    total_packets.i = max_packets;
                    memcpy(&(to_m_buffer[12]), total_packets.b, 4);

                    charint total_size;
                    total_size.i = msd_data_size;
                    memcpy(&(to_m_buffer[16]), total_size.b, 4);

                    // setting index to copy data from
                    int initial_index = 0;
                    for (int i = 0; i < max_packets; i++)
                    {
                        // setting packet number to be read to account for out-of-order delivery
                        charint curr_packet;
                        curr_packet.i = i;

                        int to_copy = MAX_PACKET_SIZE;
                        if (i + 1 == max_packets)
                        {
                            to_copy = (int)msd_data_size - initial_index;
                        }

                        memcpy(&(to_m_buffer[8]), curr_packet.b, 4);
                        memcpy(&(to_m_buffer[20]), &(msd_data_buffer)[initial_index], to_copy);

                        int udp_status = sendto(sock, to_m_buffer, sizeof(to_m_buffer), 0, (struct sockaddr *)&sift_rec_remote_addr, sizeof(sift_rec_remote_addr));
                        print_log(service, string(curr_frame.client_id), to_string(curr_frame.frame_no),
                                  "Sent packet #" + to_string(i + 1) + " of " + to_string((int)max_packets) + " to matching" +
                                      " with the following number of characters " + to_string(udp_status));
                        if (udp_status == -1)
                        {
                            cout << "Error sending: " << strerror(errno) << endl;
                        }
                        else
                        {
                            close(udp_status);
                        }
                        initial_index += MAX_PACKET_SIZE;
                        sleep_for(nanoseconds(10000000));
                    }
                }

                // free(msd_data_buffer);
            }
        }
    }
}

void siftdata_reconstructor(char *sd_char_array, matchingSiftItem receivedSiftData)
{
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
    receivedSiftData.data = reconstructed_data;
    print_log(service, "0", "0", "SiftData has been reconstructed from sift service, example data: " + to_string(sd_num_pts) + " SIFT points");
}

void *udp_sift_data_listener(void *socket)
{
    print_log(service, "0", "0", "Created thread to listen for SIFT data packets for the matching service");

    int sock = *((int *)socket);
    char tmp[4];
    char client_id[4];

    char *client_id_ptr = client_id;

    char *sift_data_buffer;
    int curr_recv_packet_no;
    int prev_recv_packet_no = 0;
    int total_packets_no;

    int packet_tally;
    int sift_data_count;
    int complete_data_size;

    recognizedMarker marker;
    bool markerDetected = false;

    int last_packet_frame_no;

    while (1)
    {
        // THERE'S AN ISSUE WITH THE MEMORY COPYING HERE!
        char packet_buffer[20 + MAX_PACKET_SIZE];
        memset(packet_buffer, 0, sizeof(packet_buffer));

        recvfrom(sock, packet_buffer, PACKET_SIZE + 20, 0, (struct sockaddr *)&matching_rec_addr, &mrd_len);

        memcpy(client_id, packet_buffer, 4);
        
        string client_id_string = client_id;
        string client_id_corr = client_id_string.substr(0, 4);
        client_id_ptr = &client_id_corr[0];

        memcpy(tmp, &(packet_buffer[4]), 4);
        int frame_no = *(int *)tmp;

        memcpy(tmp, &(packet_buffer[8]), 4);
        int curr_packet_no = *(int *)tmp;

        if (curr_packet_no == 0)
        {
            // if the first packet received
            memcpy(tmp, &(packet_buffer[16]), 4);
            complete_data_size = *(int *)tmp;

            memcpy(tmp, &(packet_buffer[12]), 4);
            total_packets_no = *(int *)tmp;

            print_log(service, string(client_id_ptr), to_string(frame_no),
                      "Receiving SIFT data in packets for Frame " + to_string(frame_no) +
                          " with an expected total number of packets of " + to_string(total_packets_no) +
                          " and total bytes of " + to_string(complete_data_size));

            sift_data_buffer = new char[complete_data_size];
            memset(sift_data_buffer, 0, complete_data_size);
            packet_tally = 0;
            sift_data_count = 0;

            last_packet_frame_no = frame_no;
        }

        int to_copy = MAX_PACKET_SIZE;
        int copy_index = curr_packet_no * MAX_PACKET_SIZE;
        if (curr_packet_no + 1 == total_packets_no)
        {
            to_copy = complete_data_size - copy_index;
        }
        memcpy(&(sift_data_buffer[copy_index]), &(packet_buffer[20]), to_copy);
        print_log(service, string(client_id_ptr), to_string(frame_no), "For Frame " + to_string(frame_no) + " received " + to_string(curr_packet_no + 1) + " out of " + to_string(total_packets_no) + " packets");

        packet_tally++;
        sift_data_count += MAX_PACKET_SIZE;

        if (curr_packet_no + 1 == total_packets_no)
        {
            // need to add logic to check whether all of the packets were received,
            // and whether they were in the correct order

            if (packet_tally == total_packets_no)
            {
                print_log(service, string(client_id_ptr), to_string(frame_no),
                          "All packets received for Frame " + to_string(frame_no) + " and will attempt to reconstruct into a SiftData struct");
                siftdata_reconstructor(sift_data_buffer, receivedSiftData);
                receivedSiftData.frame_no = frame_no;

                // find where in the JSON array is the matching frame
                string frame_to_find = string(client_id_ptr) + "_" + to_string(frame_no);
                int mbd_val = 0;
                int mbd_loc;
                for (auto mbd_it : matching_buffer_details)
                {
                    // "it" is of type json::reference and has no key() member
                    if (mbd_it == frame_to_find)
                    {
                        mbd_loc = mbd_val;
                    }
                    mbd_val++;
                }

                // copy out the matching data from the deque
                matching_item md = matching_items[mbd_loc];
                char *md_client_id = md.client_id;
                char *md_client_ip = md.client_ip;
                int md_client_port = md.client_port;
                int md_frame_no = md.frame_no;
                vector<int> result = md.lsh_result;

                recognizedMarker marker;
                markerDetected = matching(result, reconstructed_data, marker);

                inter_service_buffer curRes;
                if (markerDetected)
                {
                    charfloat p;
                    curRes.client_id = md_client_id;
                    curRes.frame_no.i = md_frame_no;
                    curRes.data_type.i = MSG_DATA_TRANSMISSION;
                    curRes.buffer_size.i = 1;
                    curRes.client_ip = md_client_ip;
                    curRes.client_port.i = md_client_port;
                    curRes.previous_service.i = BOUNDARY;

                    //curRes.buffer = new unsigned char[100 * curRes.buffer_size.i];
                    curRes.buffer = (unsigned char*)malloc(100 * curRes.buffer_size.i);

                    int pointer = 0;
                    memcpy(&(curRes.buffer[pointer]), marker.markerID.b, 4);
                    pointer += 4;
                    memcpy(&(curRes.buffer[pointer]), marker.height.b, 4);
                    pointer += 4;
                    memcpy(&(curRes.buffer[pointer]), marker.width.b, 4);
                    pointer += 4;

                    for (int j = 0; j < 4; j++)
                    {
                        p.f = marker.corners[j].x;
                        memcpy(&(curRes.buffer[pointer]), p.b, 4);
                        pointer += 4;
                        p.f = marker.corners[j].y;
                        memcpy(&(curRes.buffer[pointer]), p.b, 4);
                        pointer += 4;
                    }

                    memcpy(&(curRes.buffer[pointer]), marker.markername.data(), marker.markername.length());

                    recognizedMarkerID = marker.markerID.i;
                    inter_service_data.push(curRes);

                    // cout << recognizedMarkerID << endl;
                }
                else
                {
                    curRes.frame_no.i = frame_no;
                    curRes.buffer_size.i = 0;
                }

                //matching_items.erase(matching_items.begin()+mbd_loc);
                //matching_buffer_details.erase(mbd_loc);
            }
        }

    }
}

void *ThreadUDPSenderFunction(void *socket)
{
    print_log(service, "0", "0", "UDP sender thread created");

    char buffer[RES_SIZE];
    int sock = *((int *)socket);

    socklen_t next_service_addrlen = sizeof(next_service_addr);

    if (service_value < 5)
    {
        next_service = service_map_reverse.at(service_value + 1);
    }

    while (1)
    {
        if (inter_service_data.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        // generate new socket everytime data is needed to be sent
        int next_service_socket;
        struct sockaddr_in next_service_sock;
        if ((next_service_socket = ::socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        {
            perror("socket creation failed");
            exit(EXIT_FAILURE);
        }
        memset(&next_service_sock, 0, sizeof(next_service_sock));

        // Filling server information
        next_service_sock.sin_family = AF_INET;
        next_service_sock.sin_port = htons(0);
        next_service_sock.sin_addr.s_addr = INADDR_ANY;

        // Forcefully attaching socket to the port 8080
        if (bind(next_service_socket, (struct sockaddr*)&next_service_sock,
                sizeof(next_service_sock))
            < 0) {
            print_log(service, "0", "0", "ERROR: Unable to bind UDP");
            exit(1);
        }

        if (service == "primary")
        {
            inter_service_buffer curr_item = inter_service_data.front();
            inter_service_data.pop();

            char buffer[60 + curr_item.buffer_size.i];
            memset(buffer, 0, sizeof(buffer));

            memcpy(buffer, curr_item.client_id, 4);
            memcpy(&(buffer[4]), curr_item.frame_no.b, 4);
            memcpy(&(buffer[8]), curr_item.data_type.b, 4);
            memcpy(&(buffer[12]), curr_item.buffer_size.b, 4);
            memcpy(&(buffer[16]), curr_item.client_ip, 16);
            memcpy(&(buffer[32]), curr_item.client_port.b, 4);
            memcpy(&(buffer[36]), curr_item.previous_service.b, 4);
            memcpy(&(buffer[60]), &(curr_item.image_buffer)[0], curr_item.buffer_size.i + 1);
            sendto(next_service_socket, buffer, sizeof(buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);
            close(next_service_socket);
            print_log(service, string(curr_item.client_id), to_string(curr_item.frame_no.i),
                      "Frame " + to_string(curr_item.frame_no.i) + " sent to " + next_service +
                          " service for processing with a payload size of " + to_string(curr_item.buffer_size.i));
        }
        else if (service == "sift")
        {
            // client_addr
            inter_service_buffer curr_item = inter_service_data.front();
            inter_service_data.pop();

            char sift_buffer[48 + MAX_PACKET_SIZE];
            memset(sift_buffer, 0, sizeof(sift_buffer));

            memcpy(sift_buffer, curr_item.client_id, 4);
            memcpy(&(sift_buffer[4]), curr_item.frame_no.b, 4);
            memcpy(&(sift_buffer[8]), curr_item.data_type.b, 4);
            memcpy(&(sift_buffer[12]), curr_item.buffer_size.b, 4);
            memcpy(&(sift_buffer[16]), curr_item.client_ip, 16);
            memcpy(&(sift_buffer[32]), curr_item.client_port.b, 4);
            memcpy(&(sift_buffer[36]), curr_item.previous_service.b, 4);

            float total_packet_size = (float)curr_item.buffer_size.i;
            double max_packets = ceil(total_packet_size / MAX_PACKET_SIZE);
            if (max_packets > 1)
            {
                print_log(service, string(curr_item.client_id), to_string(curr_item.frame_no.i),
                          "Packet payload of " + to_string(curr_item.buffer_size.i) + " will be greater than " + to_string(MAX_PACKET_SIZE) +
                              " B, therefore, the data will be sent in " + to_string((int)max_packets) + " packets");

                // preparing the payload buffer of the packets to be sent
                charint total_packets;
                total_packets.i = (int)max_packets;
                memcpy(&(sift_buffer[40]), total_packets.b, 4);

                int initial_index = 0;
                for (int i = 0; i < max_packets; i++)
                {
                    charint curr_packet;
                    curr_packet.i = (int)(i);
                    memcpy(&(sift_buffer[44]), curr_packet.b, 4);

                    int to_copy = MAX_PACKET_SIZE;
                    if (i + 1 == max_packets)
                    {
                        to_copy = total_packet_size - initial_index;
                    }
                    memcpy(&(sift_buffer[48]), &(curr_item.buffer)[initial_index], to_copy);
                    int udp_status = sendto(next_service_socket, sift_buffer, sizeof(sift_buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);
                    print_log(service, string(curr_item.client_id), to_string(curr_item.frame_no.i),
                              "Sent packet #" + to_string(i + 1) + " of " + to_string((int)max_packets) + " to encoding" +
                                  " with the following number of characters " + to_string(udp_status));
                    char *ip = inet_ntoa(next_service_sock.sin_addr);
                    string ip_string = ip;
                    if (udp_status == -1)
                    {
                        cout << "Error sending: " << strerror(errno) << endl;
                    }
                    initial_index += MAX_PACKET_SIZE;
                    sleep_for(nanoseconds(5000000));
                    // sleep_until(system_clock::now() + seconds(1));
                }
                close(next_service_socket);
            }
            free(curr_item.buffer);
        }
        else if (service == "matching")
        {
            // client_addr
            inter_service_buffer curr_res = inter_service_data.front();
            inter_service_data.pop();

            int buffer_size = curr_res.buffer_size.i;

            char buffer[16 + buffer_size];
            memset(buffer, 0, sizeof(buffer));

            memcpy(buffer, curr_res.client_id, 4);
            memcpy(&(buffer[4]), curr_res.frame_no.b, 4);
            memcpy(&(buffer[12]), curr_res.buffer_size.b, 4);
            if (buffer_size != 0)
                memcpy(&(buffer[16]), curr_res.buffer, 100 * buffer_size);

            string client_return_ip = curr_res.client_ip;
            inet_pton(AF_INET, client_return_ip.c_str(), &(client_addr.sin_addr));
            int client_return_port = curr_res.client_port.i;
            client_addr.sin_port = htons(client_return_port);

            int udp_status = sendto(next_service_socket, buffer, sizeof(buffer), 0, (struct sockaddr *)&client_addr, sizeof(client_addr));
            close(next_service_socket);
            if (udp_status == -1)
            {
                printf("Error sending: %i\n", errno);
            }
            char *client_device_ip = inet_ntoa(client_addr.sin_addr);
            //string client_device_ip_string = client_device_ip;
            int client_device_port = htons(client_addr.sin_port);

            print_log(service, string(curr_res.client_id), to_string(curr_res.frame_no.i),
                      "Results for Frame " + to_string(curr_res.frame_no.i) +
                          " sent to client with number of markers of " + to_string(buffer_size));

            cout << "[DEBUG] client has IP of " << client_device_ip << " and port " << to_string(client_device_port) << endl;
        }
        else
        {
            inter_service_buffer curr_item = inter_service_data.front();
            inter_service_data.pop();

            int item_data_size = curr_item.buffer_size.i;
            char buffer[60 + item_data_size];

            memset(buffer, 0, sizeof(buffer));
            memcpy(buffer, curr_item.client_id, 4);
            memcpy(&(buffer[4]), curr_item.frame_no.b, 4);
            memcpy(&(buffer[8]), curr_item.data_type.b, 4);
            memcpy(&(buffer[12]), curr_item.buffer_size.b, 4);
            memcpy(&(buffer[16]), curr_item.client_ip, 16);
            memcpy(&(buffer[32]), curr_item.client_port.b, 4);
            memcpy(&(buffer[36]), curr_item.previous_service.b, 4);
            memcpy(&(buffer[40]), curr_item.sift_ip, 16);
            cout << "SIFT DATA1 " << curr_item.sift_ip << endl;
            memcpy(&(buffer[56]), curr_item.sift_port.b, 4);
            memcpy(&(buffer[60]), &(curr_item.buffer)[0], curr_item.buffer_size.i);

            int udp_status = sendto(next_service_socket, buffer, sizeof(buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);
            close(next_service_socket);
            if (udp_status == -1)
            {
                printf("Error sending: %i\n", errno);
            }

            print_log(service, string(curr_item.client_id), to_string(curr_item.frame_no.i),
                      "Forwarded frame " + to_string(curr_item.frame_no.i) + " for client " +
                          string(curr_item.client_id) + " to '" + next_service + "' service for processing" +
                          " with a payload size of " + to_string(curr_item.buffer_size.i));
        }
    }
}

void *ThreadProcessFunction(void *param)
{
    print_log(service, "0", "0", "Processing thread created");

    inter_service_buffer item;
    char tmp[4];

    while (1)
    {
        if (frames.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        frame_buffer curr_frame = frames.front();
        frames.pop();

        char *client_id = curr_frame.client_id;
        int frame_no = curr_frame.frame_no;
        int frame_data_type = curr_frame.data_type;
        int frame_size = curr_frame.buffer_size;

        char *client_ip = curr_frame.client_ip;
        int client_port = curr_frame.client_port;
        char *frame_data = curr_frame.buffer;

        if (frame_data_type == MSG_CLIENT_FRAME_DETECT)
        {
            if (service == "primary")
            {
                print_log(service, string(client_id), to_string(frame_no),
                          "Image from frame has been reduced from size " + to_string(frame_size) +
                              " to a Mat object of size " + to_string(frame_size));

                item.client_id = client_id;
                item.frame_no.i = frame_no;
                item.data_type.i = MSG_DATA_TRANSMISSION;
                item.buffer_size.i = frame_size;
                item.client_ip = client_ip;
                item.client_port.i = client_port;
                item.previous_service.i = service_value;
                item.image_buffer = frame_data;

                inter_service_data.push(item);
            }
        }
        else if (frame_data_type == MSG_DATA_TRANSMISSION || frame_data_type == MSG_SIFT_TO_ENCODING)
        {
            if (service == "sift")
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

                vector<uchar> imgdata(frame_data, frame_data + frame_size);
                Mat img_scene = imdecode(imgdata, CV_LOAD_IMAGE_GRAYSCALE);
                imwrite("query.jpg", img_scene);
                Mat detect = img_scene(Rect(RECO_W_OFFSET, RECO_H_OFFSET, 160, 270));

                sift_processing(sift_points, &sift_data_buffer, &raw_sift_data, detect, tData);

                charint siftresult;
                siftresult.i = sift_points;
                int sift_buffer_size = 128 * 4 * siftresult.i; // size of char values

                // push data required for next service
                item.client_id = client_id;
                item.frame_no.i = frame_no;
                item.data_type.i = MSG_SIFT_TO_ENCODING;
                item.buffer_size.i = 4 + sift_buffer_size;
                item.client_ip = client_ip;
                item.client_port.i = client_port;
                item.previous_service.i = service_value;

                // item.buffer = new unsigned char[4 + sift_buffer_size];
                item.buffer = (unsigned char *)malloc(4 + sift_buffer_size);
                memset(item.buffer, 0, 4 + sift_buffer_size);
                memcpy(&(item.buffer[0]), siftresult.b, 4);
                memcpy(&(item.buffer[4]), sift_data_buffer, sift_buffer_size);

                inter_service_data.push(item);

                // storing SIFT data for retrieval by the matching service
                int sift_data_size = 4 * siftresult.i * (15 + 3 + 128); // taken from export_siftdata
                print_log(service, string(client_id), to_string(frame_no),
                          "Expected size of SIFT data buffer to store for frame " + to_string(frame_no) +
                              " is " + to_string(sift_data_size) + " Bytes");

                // create a buffer to store for matching to retrieve
                sift_data_item curr_sdi;
                curr_sdi.client_id = item.client_id;
                curr_sdi.frame_no.i = item.frame_no.i;
                curr_sdi.sift_data_size.i = sift_data_size;
                curr_sdi.sift_data = raw_sift_data;

                int sbd_count = sift_items.size();
                if (sbd_count == sbd_max)
                {
                    sift_items.pop_front(); // pop front item if 10 items
                }
                sift_items.push_back(curr_sdi); // append to end of the 10 items
                deque<sift_data_item>::iterator it;
                for (it = sift_items.begin(); it != sift_items.end(); ++it)
                {
                    sift_data_item curr_item = *it;
                }
                if (int(sift_buffer_details.size()) == sbd_max)
                {
                    sift_buffer_details.erase(0);
                }
                sift_buffer_details.push_back(string(item.client_id) + "_" + to_string(item.frame_no.i));

                print_log(service, string(client_id), to_string(frame_no),
                          "Storing SIFT data for client " + string(item.client_id) + " and frame " +
                              to_string(frame_no) + " in SIFT data buffer for collection by matching");
            }
            else if (service == "encoding")
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

                auto encoding_results = encoding(siftres, sift_result, test, false);

                charint encoded_size;
                encoded_size.i = get<0>(encoding_results);
                int encoding_buffer_size = 4 * encoded_size.i; // size of char values

                char *encoded_vector = get<1>(encoding_results);

                item.client_id = client_id;
                item.frame_no.i = frame_no;
                item.data_type.i = MSG_DATA_TRANSMISSION;
                item.buffer_size.i = 4 + encoding_buffer_size;
                item.client_ip = client_ip;
                item.client_port.i = client_port;
                item.previous_service.i = service_value;
                item.buffer = new unsigned char[4 + encoding_buffer_size];
                memset(item.buffer, 0, 4 + encoding_buffer_size);
                memcpy(&(item.buffer[0]), encoded_size.b, 4);
                memcpy(&(item.buffer[4]), encoded_vector, encoding_buffer_size);

                item.sift_ip = curr_frame.sift_ip;
                item.sift_port.i = curr_frame.sift_port;

                inter_service_data.push(item);
                print_log(service, string(client_id), to_string(frame_no), "Performed encoding on received 'sift' data");

                delete[] siftres;
            }
            else if (service == "lsh")
            {
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
                item.data_type.i = MSG_DATA_TRANSMISSION;
                item.buffer_size.i = 4 + results_buffer_size;
                item.client_ip = client_ip;
                item.client_port.i = client_port;
                item.previous_service.i = service_value;

                item.buffer = (unsigned char *)malloc(4 + results_buffer_size);
                memset(item.buffer, 0, 4 + results_buffer_size);
                memcpy(&(item.buffer[0]), results_size.b, 4);
                memcpy(&(item.buffer[4]), results_vector, results_buffer_size);

                char* sift_ip = curr_frame.sift_ip;
                int sift_port = curr_frame.sift_port;

                item.sift_ip = sift_ip;
                item.sift_port.i = sift_port;

                inter_service_data.push(item);
                print_log(service, string(client_id), to_string(frame_no), "Performed analysis on received 'encoding' data");
            }
            else if (service == "matching")
            {
                vector<int> result;

                memcpy(tmp, &(frame_data[0]), 4);
                int result_size = *(int *)tmp;
                    
                //char *results_char = new char[result_size * 4];
                char *results_char = (char*)malloc(result_size*4);
                memset(results_char, 0, result_size*4);
                memcpy(results_char, &(frame_data[4]), result_size*4);

                int data_index = 0;
                for (int i = 0; i < result_size; i++)
                {
                    memcpy(tmp, &(results_char[data_index]), 4);
                    int *curr_int = (int *)tmp;
                    result.push_back(*curr_int);

                    data_index += 4;
                }

                // sleep_until(system_clock::now() + nanoseconds(100000000));
                // cout << "Last reconstructed data was " << receivedSiftData.frame_no << " and current is " << frame_no << endl;
                // if (receivedSiftData.frame_no == frame_no) {
                //     markerDetected = matching(result, reconstructed_data, marker);
                // }

                // create buffer to store for when sift data is reconstructed
                matching_item curr_mi;
                curr_mi.client_id = client_id;
                curr_mi.client_ip = client_ip;
                curr_mi.client_port = client_port;
                curr_mi.frame_no = frame_no;
                curr_mi.lsh_result = result;

                int mi_count = matching_items.size();
                // if ((mi_count-mbd_max)==0)
                // {
                //     matching_items.pop_front();
                // }
                matching_items.push_back(curr_mi); // append to end of the 10 items
                deque<matching_item>::iterator it;
                for (it = matching_items.begin(); it != matching_items.end(); ++it)
                {
                    matching_item curr_item = *it;
                }
                // if (int(matching_buffer_details.size()) == mbd_max)
                // {
                //     matching_buffer_details.erase(0);
                // }
                matching_buffer_details.push_back(string(client_id) + "_" + to_string(frame_no));
                //delete[] results_char;
            }
        }
    }
}

void runServer(int port, string service)
{
    pthread_t senderThread, receiverThread, imageProcessThread, processThread;
    pthread_t sift_listen_thread;
    char buffer[PACKET_SIZE];
    char fileid[4];
    int status = 0;
    int sockUDP;
    int sl_udp_sock;

    memset((char *)&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = htons(port);

    if ((sockUDP = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        print_log(service, "0", "0", "ERROR: Unable to open UDP socket");
        exit(1);
    }
    if (bind(sockUDP, (struct sockaddr *)&localAddr, sizeof(localAddr)) < 0)
    {
        print_log(service, "0", "0", "ERROR: Unable to bind UDP");
        exit(1);
    }
    print_log(service, "0", "0", "Server UDP port for service " + service + " is bound to " + to_string(port));

    isClientAlive = true;
    pthread_create(&receiverThread, NULL, ThreadUDPReceiverFunction, (void *)&sockUDP);
    pthread_create(&senderThread, NULL, ThreadUDPSenderFunction, (void *)&sockUDP);
    pthread_create(&imageProcessThread, NULL, ThreadProcessFunction, NULL);

    // if primary service, set the details of the next service
    if (service == "primary" || service == "matching")
    {
        json sift_ns = services["sift"];
        string sift_ip = sift_ns[0];
        string sift_port_string = sift_ns[1];
        int sift_port = stoi(sift_port_string);
        if (service == "primary")
        {
            //inet_pton(AF_INET, sift_ip.c_str(), &(next_service_addr.sin_addr));
            next_service_addr.sin_family = AF_INET;
            next_service_addr.sin_addr.s_addr = inet_addr(sift_ip.c_str());
            next_service_addr.sin_port = htons(sift_port);
        }
        else if (service == "matching")
        {
            memset((char *)&matching_rec_addr, 0, sizeof(matching_rec_addr));
            matching_rec_addr.sin_family = AF_INET;
            matching_rec_addr.sin_addr.s_addr = htonl(INADDR_ANY);
            matching_rec_addr.sin_port = htons(51005);

            if ((sl_udp_sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
            {
                cout << "[ERROR] Unable to open UDP socket" << endl;
                exit(1);
            }
            if (bind(sl_udp_sock, (struct sockaddr *)&matching_rec_addr, sizeof(matching_rec_addr)) < 0)
            {
                cout << "[ERROR] Unable to bind UDP " << endl;
                exit(1);
            }

            pthread_create(&sift_listen_thread, NULL, udp_sift_data_listener, (void *)&sl_udp_sock);
        }
    }
    else if (service == "sift")
    {
        json matching_ns = services["matching"];
        string matching_ip = matching_ns[0];
        inet_pton(AF_INET, matching_ip.c_str(), &(sift_rec_remote_addr.sin_addr));
        sift_rec_remote_addr.sin_port = htons(51005);
    }

    // check if there is a following service, and attempt to contact
    json val_names = (services_outline["val_name"]);
    int next_service_val = service_value + 1;
    auto nsv = val_names.find(to_string(next_service_val));
    if ((nsv != val_names.end()) == true)
    {
        // if there is a following service, set the details of it
        string next_service = *nsv;
        json next_service_details = services[next_service];

        string next_service_ip = next_service_details[0];
        string next_service_port_string = next_service_details[1];
        int next_service_port = stoi(next_service_port_string);
        
        inet_pton(AF_INET, next_service_ip.c_str(), &(next_service_addr.sin_addr));
            print_log(service, "0", "0", "Setting the details of the next service '" + next_service + "' to have an IP of " + next_service_ip + " and port " + to_string(next_service_port));
        next_service_addr.sin_port = htons(next_service_port);
    }

    pthread_join(receiverThread, NULL);
    pthread_join(senderThread, NULL);
    pthread_join(imageProcessThread, NULL);

    if (service == "matching")
    {
        pthread_join(sift_listen_thread, NULL);
    }
}

void loadOnline()
{
    ifstream file("data/onlineData.dat");
    string line;
    int i = 0;
    while (getline(file, line))
    {
        char *fileName = new char[256];
        strcpy(fileName, line.c_str());

        if (i % 2 == 0)
            onlineImages.push_back(fileName);
        else
            onlineAnnotations.push_back(fileName);
        ++i;
    }
    file.close();
}

inline string getCurrentDateTime(string s)
{
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    if (s == "now")
        strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
    else if (s == "date")
        strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);
    return string(buf);
};

int main(int argc, char *argv[])
{
    int querysizefactor, nn_num;

    // current service name and value in the service map
    service = string(argv[1]);
    service_value = service_map.at(argv[1]);

    print_log(service, "0", "0", "Selected service is: " + string(argv[1]));
    print_log(service, "0", "0", "IP of the primary module provided is " + string(argv[2]));

    int pp_req[4]{2, 3, 4, 5}; // pre-processing required

    if (find(begin(pp_req), end(pp_req), service_value) != end(pp_req))
    {
        // performing initial variable loading and encoding
        loadOnline();
        loadImages(onlineImages);
        loadParams();

        // arbitrarily encoding the above variables
        querysizefactor = 3;
        nn_num = 5;
        encodeDatabase(querysizefactor, nn_num);
    }

    // setting the specified host IP address and the hardcoded port
    inet_pton(AF_INET, argv[2], &(main_addr.sin_addr));
    main_addr.sin_port = htons(50000 + int(service_map.at("primary")));

    int port = MAIN_PORT + service_value; // hardcoding the initial port

    runServer(port, service);

    freeParams();
    return 0;
}
