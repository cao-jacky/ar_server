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

#include <errno.h>

#define MAIN_PORT 50000

#define MESSAGE_ECHO 0
#define MESSAGE_REGISTER 1
#define MESSAGE_NEXT_SERVICE_IP 2
#define DATA_TRANSMISSION 3

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

socklen_t addrlen = sizeof(remoteAddr);
bool isClientAlive = false;

queue<frameBuffer> frames, offloadframes;
queue<resBuffer> results;
int recognizedMarkerID;

vector<char *> onlineImages;
vector<char *> onlineAnnotations;

// declaring variables needed for distributed operation
string service;
int service_value;
queue<inter_service_buffer> inter_service_data;

// hard coding the maps for each service, nothing clever needed about this
std::map<string, int> service_map = {
    {"main", 1},
    {"sift", 2},
    {"encoding", 3},
    {"lsh", 4},
    {"matching", 5},
    {"main_return", 6}
};

std::map<int, string> service_map_reverse = {
    {1, "main"},
    {2, "sift"},
    {3, "encoding"},
    {4, "lsh"},
    {5, "matching"},
    {6, "main_return"}
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
    cout << "STATUS: Service " << service << " is attempting to register with main module ";
    cout << MAIN_PORT+int(service_map.at("main")) << endl;
}

void *ThreadUDPReceiverFunction(void *socket) {
    cout << "STATUS: UDP receiver thread created" << endl;
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

        // copy client frames into frames buffer if main 
        frameBuffer curFrame;    
        memcpy(tmp, buffer, 4);
        curFrame.frmID = *(int*)tmp;        
        memcpy(tmp, &(buffer[4]), 4);
        curFrame.dataType = *(int*)tmp;

        // cout << "message type is " << curFrame.dataType << endl;

        if (service == "main") {
            if(curFrame.dataType == MESSAGE_ECHO) {
                cout << "STATUS: Received an echo message" << endl;
                charint echoID;
                echoID.i = curFrame.frmID;
                char echo[4];
                memcpy(echo, echoID.b, 4);
                sendto(sock, echo, sizeof(echo), 0, (struct sockaddr *)&remoteAddr, addrlen);
                cout << "STATUS: Sent an echo reply" << endl;
                continue;
            } else if (curFrame.dataType == MESSAGE_REGISTER) {
                string service_to_register = service_map_reverse.at(curFrame.frmID);
                // string service_to_register = distance(service_map.begin(),service_map.find(curFrame.frmID));

                registered_services.insert({service_to_register, device_ip});

                if (service_to_register == "sift") {
                    // main should assign next service IP from current stage
                    inet_pton(AF_INET, device_ip, &(next_service_addr.sin_addr));
                    next_service_addr.sin_port = htons(MAIN_PORT+service_map.at(service_to_register));
                }

                cout << "STATUS: Received a register request from service " << service_to_register;
                cout << " located on IP " << device_ip << endl; 
                cout << "STATUS: Service " << service_to_register << " is now registered" << endl;

                // check whether the service which follows the newly regisetered 
                // is registered
                string next_service = service_map_reverse.at(curFrame.frmID+1);

                if (registered_services.find(next_service) == registered_services.end()) {
                    // not registered, telling the newly registered to wait
                    cout << "STATUS: Next service " << next_service;
                    cout << " is not registered, telling " << service_to_register;
                    cout << " to wait"  << endl;
                } else {
                    // service is registered, providing service with associated IP
                    char *next_ser_ip = &string(registered_services.at(next_service))[0];           

                    charint register_id;
                    register_id.i = service_value; // ID of service registering itself

                    charint message_type;
                    message_type.i = MESSAGE_NEXT_SERVICE_IP;

                    // socklen_t addrlen = sizeof(remoteAddr);

                    charint size_next_ip;
                    size_next_ip.i = strlen(next_ser_ip);

                    char nsi_array[12+strlen(next_ser_ip)];
                    memcpy(nsi_array, register_id.b, 4);
                    memcpy(&(nsi_array[4]), message_type.b, 4);
                    memcpy(&(nsi_array[8]), size_next_ip.b, 4);
                    memcpy(&(nsi_array[12]), next_ser_ip, size_next_ip.i);
                    sendto(sock, nsi_array, sizeof(nsi_array), 0, (struct sockaddr *)&remoteAddr, addrlen);

                    cout << "STATUS: Service " << next_service;
                    cout << " is registered, providing " << service_to_register;
                    cout << " with the IP " << next_ser_ip << endl;   
                }
            } else if (curFrame.dataType == IMAGE_DETECT){
                memcpy(tmp, &(buffer[8]), 4);
                curFrame.bufferSize = *(int*)tmp;
                // cout<<"================================================"<<endl;
                cout << "STATUS: Frame " << curFrame.frmID << " received, filesize: ";
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

                memcpy(tmp, &(buffer[8]), 4);
                int next_ip_data_size = *(int*)tmp;

                char tmp_ip[next_ip_data_size];
                memcpy(tmp_ip, &(buffer[12]), next_ip_data_size);
                char* next_ip = tmp_ip;

                inet_pton(AF_INET, next_ip, &(next_service_addr.sin_addr)); 
                next_service_addr.sin_port = htons(MAIN_PORT+service_value+1);
                cout << "STATUS: Received IP for next service, assigning ";
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

                    cout << "buffer size is " << curFrame.bufferSize << endl;

                    // check if a multipacket request
                    // memcpy(tmp, &(buffer[16]), 4);
                    // int num_packets = *(int*)tmp;

                    // cout << num_packets << endl;

                    curFrame.buffer = new char[curFrame.bufferSize];
                    memset(curFrame.buffer, 0, curFrame.bufferSize);
                    memcpy(curFrame.buffer, &(buffer[20]), curFrame.bufferSize);

                    memcpy(tmp, &(buffer[20]), 4);
                    int sift_result = *(int*)tmp;
                    cout << sift_result << endl;

                    frames.push(curFrame);
                }

                cout << "STATUS: Received data from previous service" << endl;
            }

        } 
        
        // else {
        //     memset(buffer, 0, sizeof(buffer));
        //     memcpy(buffer, curr_item.frame_id.b, 4);
        //     memcpy(&(buffer[4]), curr_item.previous_service.b, 4);
        //     memcpy(&(buffer[8]), curr_item.buffer_size.b, 4);
        //     memcpy(&(buffer[12]), curr_item.buffer, curr_item.buffer_size.i);
        //     sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&remoteAddr, addrlen);
            
        // }
    }
}

void *ThreadUDPSenderFunction(void *socket) {
    cout << "STATUS: UDP sender thread created" << endl;
    // char buffer[RES_SIZE];
    int sock = *((int*)socket);

    socklen_t next_service_addrlen = sizeof(next_service_addr);

    string next_service = service_map_reverse.at(service_value+1);

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

            cout << "STATUS: Frame " << curr_item.frame_id.i << " sent to ";
            cout << next_service  << " service for processing ";
            cout << "at " << setprecision(15) << wallclock() << " and payload size is ";
            cout << curr_item.buffer_size.i << endl;
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
            // int udp_status = sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);
            // cout << ". Sender has status " << udp_status << endl;
            // if(udp_status == -1) {
            //     printf("Error sending: %i\n",errno);
            // }

            // keeping this buffer size the same, so next service can check 
            // whether the item is complete
            memcpy(&(buffer[12]), curr_item.buffer_size.b, 4); 

            // logic to account for if payload greater than max size of 65kB
            if (item_data_size > MAX_PACKET_SIZE) {
                int max_packets = ceil(item_data_size / MAX_PACKET_SIZE);
                cout << "STATUS: Packet payload will be greater than " << MAX_PACKET_SIZE <<  "B. ";
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
                    cout << "STATUS: Sent packet #" << i << " of " << max_packets;
                    cout << ". Sender has status " << udp_status << endl;
                    if(udp_status == -1) {
                        printf("Error sending: %i\n",errno);
                    }
                    initial_index = i * MAX_PACKET_SIZE; 
                }
            } else {
                memcpy(&(buffer[20]), curr_item.buffer, curr_item.buffer_size.i);
                    
                int udp_status = sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);
                // int udp_status = sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&next_service_addr, next_service_addrlen);
                // cout << "STATUS: Sent data " << i << " of " << max_packets;
                // cout << ". Sender has status " << udp_status << endl;
                if(udp_status == -1) {
                    printf("Error sending: %i\n",errno);
                }

            }

            cout << "STATUS: Forwarded Frame " << curr_item.frame_id.i << " to ";
            cout << next_service  << " service for processing ";
            cout << "at " << setprecision(15) << wallclock() << " and payload size is ";
            cout << curr_item.buffer_size.i << endl;
        } 

        // resBuffer curRes = results.front();
        // results.pop();

        // memset(buffer, 0, sizeof(buffer));
        // memcpy(buffer, curRes.resID.b, 4);
        // memcpy(&(buffer[4]), curRes.resType.b, 4);
        // memcpy(&(buffer[8]), curRes.markerNum.b, 4);
        // if(curRes.markerNum.i != 0)
        //     memcpy(&(buffer[12]), curRes.buffer, 100 * curRes.markerNum.i);
        // sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&remoteAddr, addrlen);
        // cout<<"Frame "<<curRes.resID.i<<" res sent, "<<"marker#: "<<curRes.markerNum.i;
        // cout<<" at "<<setprecision(15)<<wallclock()<<endl<<endl;
    }    
}

void *ThreadProcessFunction(void *param) {
    cout << "STATUS: Processing thread created" << endl;
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

                // markerDetected = query(detect, marker);

                // encoding into JPG and copying into buffer 
                vector<uchar> encoding_buffer;
                vector<int> param(2);
                param[0] = IMWRITE_JPEG_QUALITY;
                param[1] = 95; //default(95) 0-100
                imencode(".jpg", detect, encoding_buffer, param);

                int detect_size = encoding_buffer.size();

                cout << "STATUS: Image reduced to a Mat object of size " << detect_size << endl;

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
                // vector<float> sift_array(2);
                float sift_array[2];
                int sift_result;
                float sift_resg;
                int height, width;
                vector<float> test;

                Mat detect_image = imdecode(Mat(1, frmSize, CV_8UC1, frmdata), CV_LOAD_IMAGE_UNCHANGED);
                // imwrite("sift.jpg", detect_image);

                auto sift_results = sift_processing(detect_image, tData, test, true, false);

                charint siftresult;
                siftresult.i = get<0>(sift_results);

                char* sift_buffer = get<1>(sift_results);
                int sift_buffer_size = 4 * siftresult.i; // size of char values

                item.frame_id.i = frmID;
                item.previous_service.i = service_value;
                item.buffer_size.i = 4 + sift_buffer_size;
                item.buffer = new unsigned char[4 + sift_buffer_size];
                memset(item.buffer, 0, 4 + sift_buffer_size);
                memcpy(&(item.buffer[0]), siftresult.b, 4);
                memcpy(&(item.buffer[4]), sift_buffer, sift_buffer_size);

                inter_service_data.push(item);
            } else if (service == "encoding") {
                float* siftres;

                memcpy(tmp, &(frmdata[0]), 4);
                int sift_result = *(int*)tmp;

                char* sift_resg = (char*)calloc(sift_result, 4);
                memcpy(sift_resg, &(frmdata[4]), 4*sift_result);

                float tmp[4];
                memcpy(tmp, &(sift_resg)[2856], 4);
                cout << *(float*)tmp << endl;

                siftres = (float *)calloc(sizeof(float), sizeof(float)*128*sift_result);

                // looping through char array to convert data back into floats
                // at i = 0, index should begin at 4
                int data_index = 0;
                for (int i=0; i<sift_result; i++) {
                    memcpy(tmp, &(sift_resg[data_index]), 4);
                    float *curr_float = (float*)tmp;
                    // cout << i << " " << curr_float << endl;

                    memcpy(siftres, curr_float, (128+1)*sizeof(float));
                    siftres += 128;
                    data_index += 4;

                }

                encoding(siftres, sift_result);

                // cout << sift_result << endl;
                // cout << sift_resg << endl;
                cout << "STATUS: Performed encoding on received SIFT data" << endl;

            }
        }

        

        // resBuffer curRes;
        // if(markerDetected) {
        //     charfloat p;
        //     curRes.resID.i = frmID;
        //     curRes.resType.i = BOUNDARY;
        //     curRes.markerNum.i = 1;
        //     curRes.buffer = new char[100 * curRes.markerNum.i];

        //     int pointer = 0;
        //     memcpy(&(curRes.buffer[pointer]), marker.markerID.b, 4);
        //     pointer += 4;
        //     memcpy(&(curRes.buffer[pointer]), marker.height.b, 4);
        //     pointer += 4;
        //     memcpy(&(curRes.buffer[pointer]), marker.width.b, 4);
        //     pointer += 4;

        //     for(int j = 0; j < 4; j++) {
        //         p.f = marker.corners[j].x;
        //         memcpy(&(curRes.buffer[pointer]), p.b, 4);
        //         pointer+=4;
        //         p.f = marker.corners[j].y;
        //         memcpy(&(curRes.buffer[pointer]), p.b, 4);        
        //         pointer+=4;            
        //     }

        //     memcpy(&(curRes.buffer[pointer]), marker.markername.data(), marker.markername.length());

        //     recognizedMarkerID = marker.markerID.i;

        //     if(curRes.markerNum.i > 0)
        //         addCacheItem(curFrame, curRes);
        //         cout << "Added item to cache" << endl;
        // }
        // else {
        //     curRes.resID.i = frmID;
        //     curRes.markerNum.i = 0;
        // }

        // results.push(curRes);
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
    char buffer[PACKET_SIZE];
    char fileid[4];
    int status = 0;
    int sockUDP;

    memset((char*)&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = htons(port);

    if((sockUDP = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        cout << "ERROR: Unable to open UDP socket" << endl;
        exit(1);
    }
    if(bind(sockUDP, (struct sockaddr *)&localAddr, sizeof(localAddr)) < 0) {
        cout << "ERROR: Unable to bind UDP " << endl;
        exit(1);
    }
    cout << endl << "STATUS: Server UDP port for service " << service <<  " is bound to " << port << endl;

    isClientAlive = true;
    pthread_create(&receiverThread, NULL, ThreadUDPReceiverFunction, (void *)&sockUDP);
    pthread_create(&senderThread, NULL, ThreadUDPSenderFunction, (void *)&sockUDP);
    pthread_create(&imageProcessThread, NULL, ThreadProcessFunction, NULL);
    // pthread_create(&processThread, NULL, ThreadCacheSearchFunction, NULL);

    pthread_join(receiverThread, NULL);
    pthread_join(senderThread, NULL);
    pthread_join(imageProcessThread, NULL);
    // pthread_join(processThread, NULL);

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

    cout << "STATUS: Selected service is: " << argv[1] << endl;
    cout << "STATUS: IP of main module provided is " << argv[2] << endl; 

    service_value = service_map.at(argv[1]);

    if (service_value == 3) {
        // performing initial variable loading and encoding
        loadOnline();
        loadImages(onlineImages);
        loadParams();

        // arbitrarily encoding the above variables
        querysizefactor = 3;
        nn_num = 5;
        
        encodeDatabase(querysizefactor, nn_num); 
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
