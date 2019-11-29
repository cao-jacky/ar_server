#include "reco.hpp"
#include "cuda_files.h"

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

#include <iostream>
#include <sstream>
#include <string>

#define MESSAGE_ECHO 0
#define FEATURES 1
#define IMAGE_DETECT 2
#define BOUNDARY 3
#define PACKET_SIZE 60000
#define RES_SIZE 512
//#define TRAIN
#define UDP

#define MAXCHAR 1000

using namespace std;
using namespace cv;

struct sockaddr_in localAddr;
struct sockaddr_in remoteAddr;
socklen_t addrlen = sizeof(remoteAddr);
bool isClientAlive = false;

queue<frameBuffer> frames, offloadframes;
queue<resBuffer> results;
int recognizedMarkerID;

vector<char *> onlineImages;
vector<char *> onlineAnnotations;

void *ThreadUDPReceiverFunction(void *socket) {
    cout<<"UDP receiver thread created"<<endl;
    char tmp[4];
    char buffer[PACKET_SIZE];
    int sock = *((int*)socket);

    while (1) {
        memset(buffer, 0, sizeof(buffer));
        recvfrom(sock, buffer, PACKET_SIZE, 0, (struct sockaddr *)&remoteAddr, &addrlen);
        char *device_ip = inet_ntoa(remoteAddr.sin_addr);

        frameBuffer curFrame;    
        memcpy(tmp, buffer, 4);
        curFrame.frmID = *(int*)tmp;        
        memcpy(tmp, &(buffer[4]), 4);
        curFrame.dataType = *(int*)tmp;

        if(curFrame.dataType == MESSAGE_ECHO) {
            cout<<"echo message!"<<endl;
            charint echoID;
            echoID.i = curFrame.frmID;
            char echo[4];
            memcpy(echo, echoID.b, 4);
            sendto(sock, echo, sizeof(echo), 0, (struct sockaddr *)&remoteAddr, addrlen);
            cout<<"echo reply sent!"<<endl;
            continue;
        }

        memcpy(tmp, &(buffer[8]), 4);
        curFrame.bufferSize = *(int*)tmp;
        cout<<"================================================"<<endl;
        cout<<"Frame "<<curFrame.frmID<<" received, filesize: "<<curFrame.bufferSize;
        cout<<" at "<<setprecision(15)<<wallclock();
        cout<<" from device with IP "<<device_ip<<endl;
        curFrame.buffer = new char[curFrame.bufferSize];
        memset(curFrame.buffer, 0, curFrame.bufferSize);
        memcpy(curFrame.buffer, &(buffer[12]), curFrame.bufferSize);
        
        frames.push(curFrame);
    }
}

void *ThreadUDPSenderFunction(void *socket) {
    cout << "UDP sender thread created" << endl;
    char buffer[RES_SIZE];
    int sock = *((int*)socket);

    while (1) {
        if(results.empty()) {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        resBuffer curRes = results.front();
        results.pop();

        memset(buffer, 0, sizeof(buffer));
        memcpy(buffer, curRes.resID.b, 4);
        memcpy(&(buffer[4]), curRes.resType.b, 4);
        memcpy(&(buffer[8]), curRes.markerNum.b, 4);
        if(curRes.markerNum.i != 0)
            memcpy(&(buffer[12]), curRes.buffer, 100 * curRes.markerNum.i);
        sendto(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&remoteAddr, addrlen);
        cout<<"Frame "<<curRes.resID.i<<" res sent, "<<"marker#: "<<curRes.markerNum.i;
        cout<<" at "<<setprecision(15)<<wallclock()<<endl<<endl;
    }    
}

void *ThreadProcessFunction(void *param) {
    cout<<"Process thread created"<<endl;
    recognizedMarker marker;
    bool markerDetected = false;

    while (1) {
        if(offloadframes.empty()) {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        frameBuffer curFrame = offloadframes.front();
        offloadframes.pop();

        int frmID = curFrame.frmID;
        int frmDataType = curFrame.dataType;
        int frmSize = curFrame.bufferSize;
        char* frmdata = curFrame.buffer;
        
        if(frmDataType == IMAGE_DETECT) {
            vector<uchar> imgdata(frmdata, frmdata + frmSize);
            Mat img_scene = imdecode(imgdata, CV_LOAD_IMAGE_GRAYSCALE);
            Mat detect = img_scene(Rect(RECO_W_OFFSET, RECO_H_OFFSET, 160, 270));
            markerDetected = query(detect, marker);
        }

        resBuffer curRes;
        if(markerDetected) {
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

            if(curRes.markerNum.i > 0)
                addCacheItem(curFrame, curRes);
                cout << "Added item to cache" << endl;
        }
        else {
            curRes.resID.i = frmID;
            curRes.markerNum.i = 0;
        }

        results.push(curRes);
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

int num_lines_searcher(string file_name) {
    int number_of_lines = 0;
    string line;
    ifstream currfile(file_name);

    while (getline(currfile, line))
        ++number_of_lines;
    return number_of_lines;
}


void *ThreadEdgeServerSearcherFunction(void *param) {
    // on server startup, read through the list of edger servers and attempt to form an initial communication
    // imitating a central server - edge should form initial connection to central cloud, and pull list of 
    // all registered edge servers
    cout<<"Edge server searcher thread created"<<endl;

    // make an initial connection to central server and returns:
    // - current machine IP 
    // - current machine socket/port
    // - list of all other edge servers which have registered with the 
    //      central server
    
    // cout<<inet_ntoa(localAddr.sin_addr)<<endl;
    // cout<<ntohs(localAddr.sin_port)<<endl;

    // manually setting machine IP until "central server" is implemented
    string current_ip = "10.42.0.1";
    int current_port = (int) ntohs(localAddr.sin_port);
    cout<<"Current machine IP is "<<current_ip<<" and port is "<<current_port<<endl;

    string ef_name = "edge_server_details.txt";
    int array_len = num_lines_searcher(ef_name);
    // array: rows = num of servers, columns = IP and port, also whitespace
    string server_details[array_len][3]; 

    ifstream edge_server_file(ef_name);
    if (edge_server_file.is_open()) {
        string line;
        int curr_row_count = 0;
        while (getline(edge_server_file, line)) {
            // read through edge server details file, and attempt to make
            // initial connection with each server 
            
            // split string into IP and port
            istringstream esf(line);
            int curr_col_count = 0;
            string indiv_word;
            int comp_track = 0; // integer tracker for if same machine
            while (esf >> indiv_word) {
                // compare IP and port from list to current machine 
                if ((indiv_word.compare(current_ip)) == 0)
                    ++ comp_track; 
                if ((indiv_word.compare(to_string(current_port))) == 0)
                    ++ comp_track; 
                server_details[curr_row_count][curr_col_count] = {(string)indiv_word};
                ++ curr_col_count;
            }
            string final_comp_track = to_string(comp_track);
            server_details[curr_row_count][3] = {final_comp_track};
            cout<<server_details[curr_row_count][3]<<endl;
            ++ curr_row_count;
        }
        edge_server_file.close();
    } else {
        cout<<"Edge server details cannot be found"<<endl;
    }
 
}

void runServer(int port) {
    pthread_t senderThread, receiverThread, imageProcessThread, processThread, edgeSearchThread;
    char buffer[PACKET_SIZE];
    char fileid[4];
    int status = 0;
    int sockUDP;

    memset((char*)&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = htons(port);

    if((sockUDP = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        cout<<"ERROR opening UDP socket"<<endl;
        exit(1);
    }
    if(bind(sockUDP, (struct sockaddr *)&localAddr, sizeof(localAddr)) < 0) {
        cout<<"ERROR on UDP binding"<<endl;
        exit(1);
    }
    cout << endl << "======== Server started, waiting for clients to connect ==========" << endl;

    isClientAlive = true;
    pthread_create(&receiverThread, NULL, ThreadUDPReceiverFunction, (void *)&sockUDP);
    pthread_create(&senderThread, NULL, ThreadUDPSenderFunction, (void *)&sockUDP);
    pthread_create(&imageProcessThread, NULL, ThreadProcessFunction, NULL);
    pthread_create(&processThread, NULL, ThreadCacheSearchFunction, NULL);
    pthread_create(&edgeSearchThread, NULL, ThreadEdgeServerSearcherFunction, NULL);

    pthread_join(receiverThread, NULL);
    pthread_join(senderThread, NULL);
    pthread_join(imageProcessThread, NULL);
    pthread_join(processThread, NULL);
    pthread_join(edgeSearchThread, NULL);

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
            strftime(buf, sizeof(buf), "%s", &tstruct);
        return string(buf);
    };

int main(int argc, char *argv[])
{
    int querysizefactor, nn_num, port;
    if(argc < 4) {
        cout << "Usage: " << argv[0] << " size[s/m/l] NN#[1/2/3/4/5] port" << endl;
        return 1;
    } else {
        if (argv[1][0] == 's') querysizefactor = 4;
        else if (argv[2][0] == 'm') querysizefactor = 2;
        else querysizefactor = 1;
        nn_num = argv[2][0] - '0';
        if (nn_num < 1 || nn_num > 5) nn_num = 5;
        port = strtol(argv[3], NULL, 10);
    }

    loadOnline();
    loadImages(onlineImages);
    //trainCacheParams();
#ifdef TRAIN
    trainParams();
#else
    loadParams();
#endif
    encodeDatabase(querysizefactor, nn_num); 

    // outputting terminal outputs into dated log files
    using namespace std;
    string log_file = "logs_server/logs/log_" + getCurrentDateTime("now") + ".txt";
    string error_file = "logs_server/errors/error_" + getCurrentDateTime("now") + ".txt";

    // freopen( log_file.c_str(), "w", stdout );
    // freopen( error_file.c_str(), "w", stderr );

    runServer(port);
    //scalabilityTest();    

    freeParams();
    return 0;
}
