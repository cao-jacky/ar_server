#include "reco.hpp"
#include "cuda_files.h"

#include <map>

#include <opencv2/opencv.hpp>

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

#define FEATURES 1
#define IMAGE_DETECT 2 // to change here and in the client
#define BOUNDARY 3
#define PACKET_SIZE 60000
#define MAX_PACKET_SIZE 50000
#define RES_SIZE 512
//#define TRAIN
#define UDP

using namespace std;
using namespace cv;

struct sockaddr_in localAddr;
struct sockaddr_in remoteAddr;
struct sockaddr_in main_addr;
struct sockaddr_in next_service_addr;
struct sockaddr_in sift_rec_addr;
struct sockaddr_in sift_rec_remote_addr;
struct sockaddr_in matching_addr;
struct sockaddr_in client_addr;

socklen_t addrlen = sizeof(remoteAddr);
bool isClientAlive = false;

queue<frameBuffer> frames, offloadframes;
queue<resBuffer> results;
int recognizedMarkerID;

vector<char *> onlineImages;
vector<char *> onlineAnnotations;

SiftData reconstructed_data;

// declaring variables needed for distributed operation
string service;
int service_value;
queue<inter_service_buffer> inter_service_data;

char* matching_ip = "0.0.0.0";
int matching_port = 50005;

// hard coding the maps for each service, nothing clever needed about this
std::map<string, int> service_map = {
    {"main", 1},
    {"sift", 2},
    {"encoding", 3},
    {"lsh", 4},
    {"matching", 5}
};

std::map<int, string> service_map_reverse = {
    {1, "main"},
    {2, "sift"},
    {3, "encoding"},
    {4, "lsh"},
    {5, "matching"}
};

std::map<string, string> registered_services;

void registerService(int sock) {
    charint register_id;
    register_id.i = service_value; // ID of service registering itself

    charint message_type;
    message_type.i = MESSAGE_REGISTER;

    char registering[8];
    memcpy(registering, register_id.b, 4);
    memcpy(&(registering[4]), message_type.b, 4);
    sendto(sock, registering, sizeof(registering), 0, (struct sockaddr *)&main_addr, sizeof(main_addr));
    cout << "[STATUS: " << service <<  "] Service " << service << " is attempting to register with main module ";
    cout << MAIN_PORT+int(service_map.at("main")) << endl;
}

void *ThreadUDPReceiverFunction(void *socket) {
    cout << "[STATUS: " << service <<  "] UDP receiver thread created" << endl;
    char tmp[4];
    char buffer[PACKET_SIZE];
    int sock = *((int*)socket);

    if (service != "main") {
        // when first called, try to register with server 
        registerService(sock);
    }   

    while (1) {
        memset(buffer, 0, sizeof(buffer));
        recvfrom(sock, buffer, PACKET_SIZE, 0, (struct sockaddr *)&remoteAddr, &addrlen);
        char *device_ip = inet_ntoa(remoteAddr.sin_addr);
        int device_port = htons(remoteAddr.sin_port);

        // copy client frames into frames buffer if main 
        frameBuffer curFrame;    
        memcpy(tmp, buffer, 4);
        curFrame.frmID = *(int*)tmp;        
        memcpy(tmp, &(buffer[4]), 4);
        curFrame.dataType = *(int*)tmp;

        if (service == "main") {
            if(curFrame.dataType == MESSAGE_ECHO) {
                cout << "[STATUS: " << service <<  "] Received an echo message" << endl;
                charint echoID;
                echoID.i = curFrame.frmID;
                char echo[4];
                memcpy(echo, echoID.b, 4);
                sendto(sock, echo, sizeof(echo), 0, (struct sockaddr *)&remoteAddr, addrlen);
                cout << "[STATUS: " << service <<  "] Sent an echo reply" << endl;

                // forward client details to matching service
                inet_pton(AF_INET, matching_ip, &(matching_addr.sin_addr));
                matching_addr.sin_port = htons(matching_port);

                int client_ip_strlen = strlen(device_ip);

                char client_registration[16+client_ip_strlen];

                charint client_reg_frame_id;
                client_reg_frame_id.i = 0; // no frame 
                memcpy(&(client_registration[0]), client_reg_frame_id.b, 4);

                charint client_reg_id;
                client_reg_id.i = CLIENT_REGISTRATION;
                memcpy(&(client_registration[4]), client_reg_id.b, 4);

                charint matching_port_char;
                matching_port_char.i = device_port;
                memcpy(&(client_registration[8]), matching_port_char.b, 4);

                charint device_ip_len;
                device_ip_len.i = client_ip_strlen;
                memcpy(&(client_registration[12]), device_ip_len.b, 4);

                memcpy(&(client_registration[16]), device_ip, client_ip_strlen);

                int main_to_matching = sendto(sock, client_registration, sizeof(client_registration), 0, (struct sockaddr *)&matching_addr, sizeof(matching_addr));
                cout << "[STATUS: " << service <<  "] Sending client details to matching service " << endl;
                if(main_to_matching == -1) {
                    cout << "Error sending: " << strerror(errno) << endl;
                }

                continue;
            } else if (curFrame.dataType == MESSAGE_REGISTER) {
                string service_to_register = service_map_reverse.at(curFrame.frmID);
                cout << service_to_register << endl;

                if (service_to_register == "sift") {
                    // main should assign next service IP from current stage
                    inet_pton(AF_INET, device_ip, &(next_service_addr.sin_addr));
                    next_service_addr.sin_port = htons(MAIN_PORT+service_map.at(service_to_register));
                }

                cout << "[STATUS: " << service <<  "] Received a register request from service " << service_to_register;
                cout << " located on IP " << device_ip << endl; 

                registered_services.insert({service_to_register, device_ip});
                cout << "[STATUS: " << service <<  "] Service " << service_to_register << " is now registered" << endl;

                if (service_to_register == "matching") {
                    matching_ip = &string(registered_services.at("matching"))[0];     
                }

                // check whether the service which follows the newly regisetered is actually registered
                string next_service;
                if (service_to_register != "matching") {
                    next_service = service_map_reverse.at(curFrame.frmID+1);
                }

                if (registered_services.find(next_service) == registered_services.end()) {
                    // not registered, telling the newly registered to wait
                    cout << "[STATUS: " << service <<  "] Next service " << next_service;
                    cout << " is not registered, telling " << service_to_register;
                    cout << " to wait"  << endl;
                } else {
                    // service is registered, providing service with associated IP
                    char *next_ser_ip = &string(registered_services.at(next_service))[0];           

                    charint register_id;
                    register_id.i = service_value; // ID of service registering itself

                    charint message_type;
                    message_type.i = MESSAGE_NEXT_SERVICE_IP;

                    charint size_next_ip;
                    size_next_ip.i = strlen(next_ser_ip);

                    char nsi_array[12+strlen(next_ser_ip)];
                    memcpy(nsi_array, register_id.b, 4);
                    memcpy(&(nsi_array[4]), message_type.b, 4);
                    memcpy(&(nsi_array[8]), size_next_ip.b, 4);
                    memcpy(&(nsi_array[12]), next_ser_ip, size_next_ip.i);
                    sendto(sock, nsi_array, sizeof(nsi_array), 0, (struct sockaddr *)&remoteAddr, addrlen);

                    cout << "[STATUS: " << service <<  "] Service " << next_service;
                    cout << " is registered, providing " << service_to_register;
                    cout << " with the IP " << next_ser_ip << endl;   

                    // if the service to register is sift, assumption is that matching is already registered
                    // therefore provide sift with matching's details
                    if (service_to_register == "sift") {
                        // selecting the IPs from the array on main
                        charint register_id;
                        register_id.i = 0; // information not needed, will keep value at zero

                        charint message_type;
                        message_type.i = SIFT_TO_MATCHING;

                        charint size_matching_ip;
                        size_matching_ip.i = strlen(matching_ip);

                        char mi_array[12+strlen(matching_ip)];
                        memcpy(mi_array, register_id.b, 4);
                        memcpy(&(mi_array[4]), message_type.b, 4);
                        memcpy(&(mi_array[8]), size_matching_ip.b, 4);
                        memcpy(&(mi_array[12]), matching_ip, size_matching_ip.i);
                        int sift_ip_status = sendto(sock, mi_array, sizeof(mi_array), 0, (struct sockaddr *)&remoteAddr, sizeof(remoteAddr));
                        cout << "[STATUS: " << service <<  "] Sent registered IP of matching (" << matching_ip << ") to sift" << endl;  
                        if(sift_ip_status == -1) {
                            cout << "Error sending: " << strerror(errno) << endl;
                        }
                    }
                }
            } else if (curFrame.dataType == IMAGE_DETECT){
                memcpy(tmp, &(buffer[8]), 4);
                curFrame.bufferSize = *(int*)tmp;
                cout << "[STATUS: " << service <<  "] Frame " << curFrame.frmID << " received, filesize: ";
                cout << curFrame.bufferSize << " at "<< setprecision(15)<<wallclock();
                cout << " from device with IP " << device_ip << endl;
                curFrame.buffer = new char[curFrame.bufferSize];
                memset(curFrame.buffer, 0, curFrame.bufferSize);
                memcpy(curFrame.buffer, &(buffer[12]), curFrame.bufferSize);
                
                frames.push(curFrame);
            }
        } else {
            if (curFrame.dataType == MESSAGE_NEXT_SERVICE_IP) {
                // store the IP for the next service into a specific variable
                string next_service;
                if (service != "matching") {
                    next_service = service_map_reverse.at(service_value+1);
                }

                memcpy(tmp, &(buffer[8]), 4);
                int next_ip_data_size = *(int*)tmp;

                // cout << next_ip_data_size << " " << sizeof(buffer) << endl;

                char next_tmp_ip[next_ip_data_size];
                memcpy(next_tmp_ip, &(buffer[12]), next_ip_data_size+1);
                // cout << next_tmp_ip << endl;

                inet_pton(AF_INET, next_tmp_ip, &(next_service_addr.sin_addr)); 
                next_service_addr.sin_port = htons(MAIN_PORT+service_value+1);
                cout << "[STATUS: " << service <<  "] Received IP for next service " << next_service << ": " << next_tmp_ip; 
                cout << ", then assigning address object that should have the same IP: ";
                cout << inet_ntoa(next_service_addr.sin_addr) << endl;
            } else if (curFrame.dataType == DATA_TRANSMISSION) {
                // performing logic to check that received data is supposed to be sent on
                memcpy(tmp, &(buffer[8]), 4);
                int previous_service_val = *(int*)tmp; 

                if (previous_service_val == service_value - 1) {
                    // if the data received is from previous service, proceed
                    // with copying out the data
                    memcpy(tmp, &(buffer[12]), 4);
                    curFrame.bufferSize = *(int*)tmp;

                    curFrame.buffer = new char[curFrame.bufferSize];
                    memset(curFrame.buffer, 0, curFrame.bufferSize);
                    memcpy(curFrame.buffer, &(buffer[20]), curFrame.bufferSize);

                    memcpy(tmp, &(buffer[20]), 4);
                    int sift_result = *(int*)tmp;

                    frames.push(curFrame);
                }
                cout << "[STATUS: " << service <<  "] Received data from previous service" << endl;
            } else if (curFrame.dataType == CLIENT_REGISTRATION) {
                cout << "[STATUS: " << service <<  "] Received client registration details from main" << endl;

                memcpy(tmp, &(buffer[8]), 4);
                int client_port = *(int*)tmp;

                memcpy(tmp, &(buffer[12]), 4);
                int client_ip_len = *(int*)tmp;

                char client_ip_tmp[client_ip_len];
                memcpy(client_ip_tmp, &(buffer[16]), client_ip_len);

                // creating client object to return data to
                inet_pton(AF_INET, client_ip_tmp, &(client_addr.sin_addr));
                client_addr.sin_port = htons(client_port);
            } else if (curFrame.dataType == SIFT_TO_MATCHING) {
                memcpy(tmp, &(buffer[8]), 4);
                int matching_ip_len = *(int*)tmp;

                cout << matching_ip_len << endl;

                char matching_ip_tmp[matching_ip_len];
                memcpy(matching_ip_tmp, &(buffer[12]), matching_ip_len+1);
                matching_ip = matching_ip_tmp;

                cout << "[STATUS: " << service <<  "] Received matching details from main, matching has an IP of " << matching_ip << endl;
            }
        } 
    }
}

void siftdata_reconstructor(char* sd_char_array) {
    char tmp[4];
    // SiftPoint *cpu_data;

    // SiftData reconstructed_data;

    int curr_posn = 0;

    memcpy(tmp, &(sd_char_array[curr_posn]), 4);
    int sd_num_pts = *(int*)tmp;
    reconstructed_data.numPts = sd_num_pts;
    curr_posn += 4;

    memcpy(tmp, &(sd_char_array[curr_posn]), 4);
    reconstructed_data.maxPts = *(int*)tmp;
    curr_posn += 4;
    
    // reconstructed_data.h_data = cpu_data;
    SiftPoint *cpu_data = (SiftPoint*)calloc(sd_num_pts, sizeof(SiftPoint));

    for (int i=0; i<sd_num_pts; i++) {
        SiftPoint *curr_data = (&cpu_data[i]);

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->xpos = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->ypos = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->scale = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->sharpness = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->edgeness = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->orientation = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->score = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->ambiguity = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->match = *(int*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->match_xpos = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->match_ypos = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->match_error = *(float*)tmp;
        curr_posn += 4;

        memcpy(tmp, &(sd_char_array[curr_posn]), 4);
        curr_data->subsampling = *(float*)tmp;
        curr_posn += 4;

        // re-creating the empty array
        for (int j=0; j<3; j++) {
            memcpy(tmp, &(sd_char_array[curr_posn]), 4);
            curr_data->empty[j] = *(float*)tmp;
            curr_posn += 4;
        }

        for (int k=0; k<128; k++) {
            memcpy(tmp, &(sd_char_array[curr_posn]), 4);
            curr_data->data[k] = *(float*)tmp;
            curr_posn += 4;            
        }

    }

    // inserting data into reconstructed data structure
    reconstructed_data.h_data = cpu_data;
    cout << "[STATUS: " << service <<  "] SiftData has been reconstructed from sift service." << endl;

}

void *udp_sift_data_listener(void *socket) {
    cout << "[STATUS: " << service <<  "] Created thread to listen for SIFT data packets for the matching service" << endl;
    int sock = *((int*)socket);
    char packet_buffer[PACKET_SIZE];
    char tmp[4];

    char* sift_data_buffer;
    int curr_recv_packet_no;
    int prev_recv_packet_no = 0; 
    int total_packets_no;

    int packet_tally;

    while (1) {
        memset(packet_buffer, 0, sizeof(packet_buffer));
        recvfrom(sock, packet_buffer, PACKET_SIZE, 0, (struct sockaddr *)&remoteAddr, &addrlen);
    
        memcpy(tmp, packet_buffer, 4);
        int frame_no =  *(int*)tmp; 

        memcpy(tmp, &(packet_buffer[4]), 4);
        int curr_packet_no =  *(int*)tmp; 

        if (curr_packet_no == 0) {
            // if the first packet received
            memcpy(tmp, &(packet_buffer[12]), 4);
            int complete_data_size =  *(int*)tmp;

            memcpy(tmp, &(packet_buffer[8]), 4);
            total_packets_no =  *(int*)tmp;

            cout << "[STATUS: " << service <<  "] Receiving SIFT data in packets for Frame " << frame_no;
            cout <<  " with an expected total number of packets of ";
            cout << total_packets_no << " and total bytes of " << complete_data_size << endl;

            sift_data_buffer = (char*)calloc(complete_data_size, sizeof(char));
            packet_tally = 0;
        }

        memcpy(&(sift_data_buffer[curr_packet_no*MAX_PACKET_SIZE]), &(packet_buffer[16]), MAX_PACKET_SIZE);
        cout << "[STATUS: " << service <<  "] For Frame " << frame_no << " received packet with packet number of " << curr_packet_no << endl;
    
        packet_tally++;

        if (curr_packet_no+1 == total_packets_no) {
            // need to add logic to check whether all of the packets were received,
            // and whether they were in the correct order

            if (packet_tally == total_packets_no) {
                cout << "[STATUS: " << service <<  "] All packets received for Frame " << frame_no;
                cout << " will attempt to reconstruct into a SiftData struct" << endl;
                siftdata_reconstructor(sift_data_buffer);
            }
        }

    }
}

void *ThreadUDPSenderFunction(void *socket) {
    cout << "[STATUS: " << service <<  "] UDP sender thread created" << endl;
    char buffer[RES_SIZE];
    int sock = *((int*)socket);
    string next_service;

    socklen_t next_service_addrlen = sizeof(next_service_addr);

    if (service_value != 5) {
        next_service = service_map_reverse.at(service_value+1);
    } 

    if (service == "sift") {
        // assign the address for the matching service and create the remote connection
        inet_pton(AF_INET, matching_ip, &(sift_rec_remote_addr.sin_addr));
        sift_rec_remote_addr.sin_port = htons(51005);
    }

    while (1) {
        if(inter_service_data.empty()) {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        if (service == "main") {
            inter_service_buffer curr_item = inter_service_data.front();
            inter_service_data.pop();

            charint message_type;
            message_type.i = DATA_TRANSMISSION;

            char buffer[20+curr_item.buffer_size.i];

            memset(buffer, 0, sizeof(buffer));
            memcpy(buffer, curr_item.frame_id.b, 4);
            memcpy(&(buffer[4]), message_type.b, 4);
            memcpy(&(buffer[8]), curr_item.previous_service.b, 4);
            memcpy(&(buffer[12]), curr_item.buffer_size.b, 4);
            memcpy(&(buffer[20]), &(curr_item.buffer)[0], curr_item.buffer_size.i+1);
            sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);

            cout << "[STATUS: " << service <<  "] Frame " << curr_item.frame_id.i << " sent to ";
            cout << next_service  << " service for processing ";
            cout << "at " << setprecision(15) << wallclock() << " and payload size is ";
            cout << curr_item.buffer_size.i << endl;
        } else if (service == "matching") {
            // client_addr
            inter_service_buffer curRes = inter_service_data.front();
            inter_service_data.pop();

            memset(buffer, 0, sizeof(buffer));
            memcpy(buffer, curRes.frame_id.b, 4);
            memcpy(&(buffer[4]), curRes.previous_service.b, 4);
            memcpy(&(buffer[8]), curRes.buffer_size.b, 4);
            if(curRes.buffer_size.i != 0)
                memcpy(&(buffer[12]), curRes.buffer, 100 * curRes.buffer_size.i);
            sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&client_addr, sizeof(client_addr));
            cout << "[STATUS: " << service <<  "] Frame "<<curRes.frame_id.i<<" res sent, marker#: "<<curRes.buffer_size.i;
            cout << " at " << setprecision(15) << wallclock() << endl<<endl;
        } else {
            inter_service_buffer curr_item = inter_service_data.front();
            inter_service_data.pop();

            charint message_type;
            message_type.i = DATA_TRANSMISSION;

            int item_data_size = curr_item.buffer_size.i; 
            char buffer[20+item_data_size];

            memset(buffer, 0, sizeof(buffer));
            memcpy(buffer, curr_item.frame_id.b, 4);
            memcpy(&(buffer[4]), message_type.b, 4);
            memcpy(&(buffer[8]), curr_item.previous_service.b, 4);
            memcpy(&(buffer[12]), curr_item.buffer_size.b, 4); 

            // unused logic to account for if payload greater than max size of 65kB
            if (item_data_size > MAX_PACKET_SIZE) {
                int max_packets = ceil(item_data_size / MAX_PACKET_SIZE);
                cout << "[STATUS: " << service <<  "] Packet payload will be greater than " << MAX_PACKET_SIZE <<  "B. ";
                cout << "Therefore the data will be sent in " << max_packets << " parts." << endl;

                // setting index to copy data from for pa
                int initial_index = 0;
                for (int i = 0; i < max_packets; i++) {
                    // setting packet number to be read to account for out-of-order delivery
                    charint curr_packet;
                    curr_packet.i = i;

                    memcpy(&(buffer[16]), curr_packet.b, 4); 
                    memcpy(&(buffer[20]), &(curr_item.buffer)[initial_index], MAX_PACKET_SIZE);
                    
                    int udp_status = sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);

                    // int udp_status = sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);
                    cout << "[STATUS: " << service <<  "] Sent packet #" << i << " of " << max_packets;
                    cout << ". Sender has status " << udp_status << endl;
                    if(udp_status == -1) {
                        printf("Error sending: %i\n",errno);
                    }
                    initial_index = i * MAX_PACKET_SIZE; 
                }
            } else {
                memcpy(&(buffer[20]), curr_item.buffer, curr_item.buffer_size.i);
                    
                int udp_status = sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);
                if(udp_status == -1) {
                    printf("Error sending: %i\n",errno);
                }
            }
            cout << "[STATUS: " << service <<  "] Forwarded Frame " << curr_item.frame_id.i << " to ";
            cout << next_service  << " service for processing ";
            cout << "at " << setprecision(15) << wallclock() << " and payload size is ";
            cout << curr_item.buffer_size.i << endl;
        } 
    }    
}

void *ThreadProcessFunction(void *param) {
    cout << "[STATUS: " << service <<  "] Processing thread created" << endl;
    recognizedMarker marker;
    inter_service_buffer item;
    bool markerDetected = false;
    char tmp[4];

    while (1) {
        if(frames.empty()) {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        frameBuffer curFrame = frames.front();
        frames.pop();

        int frmID = curFrame.frmID;
        int frmDataType = curFrame.dataType;
        int frmSize = curFrame.bufferSize;
        char* frmdata = curFrame.buffer;
        
        if(frmDataType == IMAGE_DETECT) {
            if (service == "main") {
                // perform pre-processing if main service 
                vector<uchar> imgdata(frmdata, frmdata + frmSize);
                Mat img_scene = imdecode(imgdata, CV_LOAD_IMAGE_GRAYSCALE);
                Mat detect = img_scene(Rect(RECO_W_OFFSET, RECO_H_OFFSET, 160, 270));

                // encoding into JPG and copying into buffer 
                vector<uchar> encoding_buffer;
                vector<int> param(2);
                param[0] = IMWRITE_JPEG_QUALITY;
                param[1] = 95; //default(95) 0-100
                imencode(".jpg", detect, encoding_buffer, param);

                int detect_size = encoding_buffer.size();

                cout << "[STATUS: " << service <<  "] Image reduced to a Mat object of size " << detect_size << endl;

                item.frame_id.i = frmID;
                item.previous_service.i = service_value;
                item.buffer_size.i = detect_size;
                item.buffer = new unsigned char[detect_size];
                copy(encoding_buffer.begin(), encoding_buffer.end(), item.buffer);

                inter_service_data.push(item);
            }

            // markerDetected = query(detect, marker);
        } else if (frmDataType == DATA_TRANSMISSION) {
            if (service == "sift") {
                SiftData tData;
                float sift_array[2];
                int sift_result;
                float sift_resg;
                int height, width;
                vector<float> test;

                Mat detect_image = imdecode(Mat(1, frmSize, CV_8UC1, frmdata), CV_LOAD_IMAGE_UNCHANGED);

                auto sift_results = sift_processing(detect_image, tData, test, true, false);

                charint siftresult;
                siftresult.i = get<0>(sift_results);

                char* sift_buffer = get<1>(sift_results);
                int sift_buffer_size = 4 * siftresult.i; // size of char values

                // push data required for next service 
                item.frame_id.i = frmID;
                item.previous_service.i = service_value;
                item.buffer_size.i = 4 + sift_buffer_size;
                item.buffer = new unsigned char[4 + sift_buffer_size];
                memset(item.buffer, 0, 4 + sift_buffer_size);
                memcpy(&(item.buffer[0]), siftresult.b, 4);
                memcpy(&(item.buffer[4]), sift_buffer, sift_buffer_size);

                inter_service_data.push(item);

                // send SIFT data to the matching service
                char* sift_data_buffer = get<2>(sift_results);
                int sift_data_size = 4 * siftresult.i * (15-2+3+128); // taken from export_siftdata
                cout << "[STATUS: " << service <<  "] Expected size of SIFT data buffer to send to matching is " << sift_data_size << " Bytes" << endl;

                if (sift_data_size > MAX_PACKET_SIZE) {
                    int max_packets = ceil(sift_data_size / MAX_PACKET_SIZE);
                    cout << "[STATUS: " << service <<  "] Packet payload will be greater than " << MAX_PACKET_SIZE <<  " Bytes. ";
                    cout << "Therefore the data will be sent in " << max_packets << " parts." << endl;

                    // preparing the buffer of the packets to be sent
                    char buffer[16 + MAX_PACKET_SIZE];
                    memset(buffer, 0, sizeof(buffer));

                    charint curr_frame_no;
                    curr_frame_no.i = frmID;
                    memcpy(&(buffer[0]), curr_frame_no.b, 4);

                    charint total_packets;
                    total_packets.i = max_packets;
                    memcpy(&(buffer[8]), total_packets.b, 4);

                    charint total_size;
                    total_size.i = sift_data_size;
                    memcpy(&(buffer[12]), total_size.b, 4);

                    // setting index to copy data from 
                    int initial_index = 0;
                    for (int i = 0; i < max_packets; i++) {
                        // setting packet number to be read to account for out-of-order delivery
                        charint curr_packet;
                        curr_packet.i = i;

                        memcpy(&(buffer[4]), curr_packet.b, 4); 
                        memcpy(&(buffer[16]), &(sift_data_buffer)[initial_index], MAX_PACKET_SIZE);
                        
                        int sock = socket(AF_INET, SOCK_DGRAM, 0);
                        int udp_status = sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&sift_rec_remote_addr, sizeof(sift_rec_remote_addr));

                        cout << "[STATUS: " << service <<  "] Sent packet #" << i+1 << " of " << max_packets;
                        cout << ". Sender has status " << udp_status << endl;
                        if(udp_status == -1) {
                            cout << "Error sending: " << strerror(errno) << endl;
                        }
                        initial_index = i * MAX_PACKET_SIZE; 
                    }
                }

            } else if (service == "encoding") {
                float* siftres;

                memcpy(tmp, &(frmdata[0]), 4);
                int sift_result = *(int*)tmp;

                char* sift_resg = (char*)calloc(sift_result, 4);
                memcpy(sift_resg, &(frmdata[4]), 4*sift_result);

                siftres = (float *)calloc(sizeof(float), sizeof(float)*128*sift_result);

                // looping through char array to convert data back into floats
                // at i = 0, index should begin at 4
                int data_index = 0;
                for (int i=0; i<sift_result; i++) {
                    memcpy(tmp, &(sift_resg[data_index]), 4);
                    float *curr_float = (float*)tmp;

                    memcpy(siftres, curr_float, (128+1)*sizeof(float));
                    siftres += 128;
                    data_index += 4;
                }

                auto encoding_results = encoding(siftres, sift_result);

                charint encoded_size;
                encoded_size.i = get<0>(encoding_results);
                int encoding_buffer_size = 4 * encoded_size.i; // size of char values

                char* encoded_vector = get<1>(encoding_results);

                item.frame_id.i = frmID;
                item.previous_service.i = service_value;
                item.buffer_size.i = 4 + encoding_buffer_size;
                item.buffer = new unsigned char[4 + encoding_buffer_size];
                memset(item.buffer, 0, 4 + encoding_buffer_size);
                memcpy(&(item.buffer[0]), encoded_size.b, 4);
                memcpy(&(item.buffer[4]), encoded_vector, encoding_buffer_size);

                inter_service_data.push(item);

                cout << "[STATUS: " << service <<  "] Performed encoding on received SIFT data" << endl;
            } else if (service == "lsh") {
                vector<float> enc_vec;

                memcpy(tmp, &(frmdata[0]), 4);
                int enc_size = *(int*)tmp;

                char* enc_vec_char = (char*)calloc(enc_size, 4);
                memcpy(enc_vec_char, &(frmdata[4]), 4*enc_size);

                // looping through char array to convert data back into floats
                // at i = 0, index should begin at 4
                int data_index = 0;
                for (int i=0; i<enc_size; i++) {
                    memcpy(tmp, &(enc_vec_char[data_index]), 4);
                    float *curr_float = (float*)tmp;
                    enc_vec.push_back(*curr_float);

                    data_index += 4;
                }
                auto results_returned = lsh_nn(enc_vec);

                charint results_size;
                results_size.i = get<0>(results_returned);
                int results_buffer_size = 4 * results_size.i; // size of char values

                char* results_vector = get<1>(results_returned);

                item.frame_id.i = frmID;
                item.previous_service.i = service_value;
                item.buffer_size.i = 4 + results_buffer_size;
                item.buffer = new unsigned char[4 + results_buffer_size];
                memset(item.buffer, 0, 4 + results_buffer_size);
                memcpy(&(item.buffer[0]), results_size.b, 4);
                memcpy(&(item.buffer[4]), results_vector, results_buffer_size);

                inter_service_data.push(item);
            } else if (service == "matching") {
                vector<int> result;

                memcpy(tmp, &(frmdata[0]), 4);
                int result_size = *(int*)tmp;

                char* results_char = (char*)calloc(result_size, 4);
                memcpy(results_char, &(frmdata[4]), 4*result_size);

                int data_index = 0;
                for (int i=0; i<result_size; i++) {
                    memcpy(tmp, &(results_char[data_index]), 4);
                    int *curr_int = (int*)tmp;
                    result.push_back(*curr_int);

                    data_index += 4;
                }
                markerDetected = matching(result, reconstructed_data, marker);

                inter_service_buffer curRes;
                if(markerDetected) {
                    charfloat p;
                    curRes.frame_id.i = frmID;
                    curRes.previous_service.i = BOUNDARY;
                    curRes.buffer_size.i = 1;
                    curRes.buffer = new unsigned char[100 * curRes.buffer_size.i];

                    int pointer = 0;
                    memcpy(&(curRes.buffer[pointer]), marker.markerID.b, 4);
                    pointer += 4;
                    memcpy(&(curRes.buffer[pointer]), marker.height.b, 4);
                    pointer += 4;
                    memcpy(&(curRes.buffer[pointer]), marker.width.b, 4);
                    pointer += 4;

                    for(int j = 0; j < 4; j++) {
                        p.f = marker.corners[j].x;
                        memcpy(&(curRes.buffer[pointer]), p.b, 4);
                        pointer+=4;
                        p.f = marker.corners[j].y;
                        memcpy(&(curRes.buffer[pointer]), p.b, 4);        
                        pointer+=4;            
                    }

                    memcpy(&(curRes.buffer[pointer]), marker.markername.data(), marker.markername.length());

                    recognizedMarkerID = marker.markerID.i;
                    // cout << recognizedMarkerID << endl;

                    // if(curRes.markerNum.i > 0)
                    //     addCacheItem(curFrame, curRes);
                    //     cout << "Added item to cache" << endl;
                }
                else {
                    curRes.frame_id.i = frmID;
                    curRes.buffer_size.i = 0;
                }

                inter_service_data.push(curRes);
            }
        }
    }
}

void *ThreadCacheSearchFunction(void *param) {
    cout<<"Cache searcher thread created"<<endl;
    recognizedMarker marker;
    bool markerDetected = false;

    while (1) {
        if(frames.empty()) {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        frameBuffer curFrame = frames.front();
        frames.pop();

        int frmID = curFrame.frmID;
        int frmDataType = curFrame.dataType;
        int frmSize = curFrame.bufferSize;
        char* frmdata = curFrame.buffer;
        
        if(frmDataType == IMAGE_DETECT) {
            cout << "Searching the cache" << endl;
            vector<uchar> imgdata(frmdata, frmdata + frmSize);
            Mat img_scene = imdecode(imgdata, CV_LOAD_IMAGE_GRAYSCALE);
            imwrite("cacheQuery.jpg",img_scene);
            Mat detect = img_scene(Rect(RECO_W_OFFSET, RECO_H_OFFSET, 160, 270));
            markerDetected = cacheQuery(detect, marker);
        }

        if(markerDetected) {
            resBuffer curRes;

            charfloat p;
            curRes.resID.i = frmID;
            curRes.resType.i = BOUNDARY;
            curRes.markerNum.i = 1;
            curRes.buffer = new char[100 * curRes.markerNum.i];

            int pointer = 0;
            memcpy(&(curRes.buffer[pointer]), marker.markerID.b, 4);
            pointer += 4;
            memcpy(&(curRes.buffer[pointer]), marker.height.b, 4);
            pointer += 4;
            memcpy(&(curRes.buffer[pointer]), marker.width.b, 4);
            pointer += 4;

            for(int j = 0; j < 4; j++) {
                p.f = marker.corners[j].x;
                memcpy(&(curRes.buffer[pointer]), p.b, 4);
                pointer+=4;
                p.f = marker.corners[j].y;
                memcpy(&(curRes.buffer[pointer]), p.b, 4);        
                pointer+=4;            
            }

            memcpy(&(curRes.buffer[pointer]), marker.markername.data(), marker.markername.length());

            recognizedMarkerID = marker.markerID.i;
            results.push(curRes);
        } else {
            offloadframes.push(curFrame);
        }
    }
}

void runServer(int port, string service) {
    pthread_t senderThread, receiverThread, imageProcessThread, processThread;
    pthread_t sift_listen_thread;
    char buffer[PACKET_SIZE];
    char fileid[4];
    int status = 0;
    int sockUDP;
    int sl_udp_sock;

    memset((char*)&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = htons(port);

    if((sockUDP = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        cout << "[ERROR] Unable to open UDP socket" << endl;
        exit(1);
    }
    if(bind(sockUDP, (struct sockaddr *)&localAddr, sizeof(localAddr)) < 0) {
        cout << "[ERROR] Unable to bind UDP " << endl;
        exit(1);
    }
    cout << endl << "[STATUS: " << service <<  "] Server UDP port for service " << service <<  " is bound to " << port << endl;

    isClientAlive = true;
    pthread_create(&receiverThread, NULL, ThreadUDPReceiverFunction, (void *)&sockUDP);
    pthread_create(&senderThread, NULL, ThreadUDPSenderFunction, (void *)&sockUDP);
    pthread_create(&imageProcessThread, NULL, ThreadProcessFunction, NULL);
    // pthread_create(&processThread, NULL, ThreadCacheSearchFunction, NULL);

    if (service == "matching") {
        memset((char*)&sift_rec_addr, 0, sizeof(sift_rec_addr));
        sift_rec_addr.sin_family = AF_INET;
        sift_rec_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        sift_rec_addr.sin_port = htons(51005);

        if((sl_udp_sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
            cout << "[ERROR] Unable to open UDP socket" << endl;
            exit(1);
        }
        if(bind(sl_udp_sock, (struct sockaddr *)&sift_rec_addr, sizeof(sift_rec_addr)) < 0) {
            cout << "[ERROR] Unable to bind UDP " << endl;
            exit(1);
        }

        pthread_create(&sift_listen_thread, NULL, udp_sift_data_listener, (void *)&sl_udp_sock);
    }

    pthread_join(receiverThread, NULL);
    pthread_join(senderThread, NULL);
    pthread_join(imageProcessThread, NULL);
    // pthread_join(processThread, NULL);

    if (service == "matching") {
        pthread_join(sift_listen_thread, NULL);
    }

    cout << endl;
}

void loadOnline() 
{
    ifstream file("data/onlineData.dat");
    string line;
    int i = 0;
    while(getline(file, line)) {
        char* fileName = new char[256];
        strcpy(fileName, line.c_str());

        if(i%2 == 0) onlineImages.push_back(fileName);
        else onlineAnnotations.push_back(fileName);
        ++i;
    }
    file.close();
}

inline string getCurrentDateTime( string s ){
        time_t now = time(0);
        struct tm  tstruct;
        char  buf[80];
        tstruct = *localtime(&now);
        if(s=="now")
            strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
        else if(s=="date")
            strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);
        return string(buf);
    };

int main(int argc, char *argv[])
{
    int querysizefactor, nn_num;

    service = string(argv[1]);

    cout << "[STATUS: " << service <<  "] Selected service is: " << argv[1] << endl;
    cout << "[STATUS: " << service <<  "] IP of main module provided is " << argv[2] << endl; 

    service_value = service_map.at(argv[1]);

    int pp_req[3]{3,4,5}; // pre-processing required

    if (find(begin(pp_req), end(pp_req), service_value) != end(pp_req)) {
        // performing initial variable loading and encoding
        loadOnline();
        loadImages(onlineImages);
        loadParams();

        // arbitrarily encoding the above variables
        querysizefactor = 3;
        nn_num = 5;
        if (service_value != 5) {
            encodeDatabase(querysizefactor, nn_num); 
        }
    }

    // cout << service_value << endl;

    // setting the specified host IP address and the hardcoded port
    inet_pton(AF_INET, argv[2], &(main_addr.sin_addr)); 
    main_addr.sin_port = htons(50000+int(service_map.at("main")));

    int port = MAIN_PORT + service_value; // hardcoding the initial port 
    
    runServer(port, service);

    // int querysizefactor, nn_num, port;
    // if(argc < 4) {
    //     cout << "Usage: " << argv[0] << " size[s/m/l] NN#[1/2/3/4/5] port" << endl;
    //     return 1;
    // } else {
    //     if (argv[1][0] == 's') querysizefactor = 4;
    //     else if (argv[2][0] == 'm') querysizefactor = 2;
    //     else querysizefactor = 1;
    //     nn_num = argv[2][0] - '0';
    //     if (nn_num < 1 || nn_num > 5) nn_num = 5;
    //     port = strtol(argv[3], NULL, 10);
    // }

//     //trainCacheParams();
// #ifdef TRAIN
//     trainParams();
// #else
//     loadParams();
// #endif
//     encodeDatabase(querysizefactor, nn_num); 

    // outputting terminal outputs into dated log files
    // using namespace std;
    // string log_file = "logs_server/logs/log_" + getCurrentDateTime("now") + ".txt";
    // string error_file = "logs_server/errors/error_" + getCurrentDateTime("now") + ".txt";

    // freopen( log_file.c_str(), "w", stdout );
    // freopen( error_file.c_str(), "w", stderr );
    //scalabilityTest();    

    freeParams();
    return 0;
}
