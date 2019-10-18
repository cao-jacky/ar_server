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
#include <cstdio>
#include <string>  

#define MESSAGE_ECHO 0
#define FEATURES 1
#define IMAGE_DETECT 2
#define BOUNDARY 3
#define PACKET_SIZE 60000
#define RES_SIZE 512
//#define TRAIN
#define UDP

using namespace std;
using namespace cv;

struct sockaddr_in localAddr;
struct sockaddr_in remoteAddr;
struct sockaddr_in offloadAddr;
socklen_t addrlen = sizeof(remoteAddr);
bool isClientAlive = false;

queue<frameBuffer> frames, offloadframes;
queue<resBuffer> results;
int recognizedMarkerID;

vector<char *> onlineImages;
vector<char *> onlineAnnotations;

void *ThreadUDPReceiverFunction(void *socket) {
    // Keep function 
    cout<<"Receiver Thread Created!"<<endl;
    char tmp[4];
    char buffer[PACKET_SIZE];
    int sock = *((int*)socket);

    while (1) {
        memset(buffer, 0, sizeof(buffer));
        recvfrom(sock, buffer, PACKET_SIZE, 0, (struct sockaddr *)&remoteAddr, &addrlen);

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
        cout<<"frame "<<curFrame.frmID<<" received, filesize: "<<curFrame.bufferSize;
        cout<<" at "<<setprecision(15)<<wallclock()<<endl;
        curFrame.buffer = new char[curFrame.bufferSize];
        memset(curFrame.buffer, 0, curFrame.bufferSize);
        memcpy(curFrame.buffer, &(buffer[12]), curFrame.bufferSize);
        
        frames.push(curFrame);
    }
}

void *ThreadUDPSenderFunction(void *socket) {
    // Keep function
    cout << "Sender Thread Created!" << endl;
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
        cout<<"frame "<<curRes.resID.i<<" res sent, "<<"marker#: "<<curRes.markerNum.i;
        cout<<" at "<<setprecision(15)<<wallclock()<<endl<<endl;
    }    
}

void *ThreadTCPReceiverFunction(void *socket) {
    // Don't need - join with UDP receiver - in factr, it is more or less the same as the UDP function
    cout<<"Receiver Thread Created!"<<endl;
    char tmp[4];
    char header[12];
    char buffer[PACKET_SIZE];
    int sock = *((int*)socket);

    while (isClientAlive) {
        memset(buffer, 0, sizeof(buffer));
        if(read(sock, header, 12) <= 0) {
	    cout<<"client disconnects"<<endl;
	    isClientAlive = false;
	    continue;
	}

        frameBuffer curFrame;    
        memcpy(tmp, header, 4);
        curFrame.frmID = *(int*)tmp;        
        memcpy(tmp, &(header[4]), 4);
        curFrame.dataType = *(int*)tmp;
        memcpy(tmp, &(header[8]), 4);
        curFrame.bufferSize = *(int*)tmp;
	
	int size = 0;
	while(size < curFrame.bufferSize) {
	    size += read(sock, &(buffer[size]), curFrame.bufferSize-size);
	}
        cout<<"frame "<<curFrame.frmID<<" received, filesize: "<<curFrame.bufferSize;
        cout<<" at "<<setprecision(15)<<wallclock()<<endl<<endl;
        curFrame.buffer = new char[curFrame.bufferSize];
        memset(curFrame.buffer, 0, curFrame.bufferSize);
        memcpy(curFrame.buffer, buffer, curFrame.bufferSize);
        
        frames.push(curFrame);
    }

    cout<<"Receiver Thread finished!"<<endl;
}

void *ThreadTCPSenderFunction(void *socket) {
    // Don't need - join with UDP sender - again, same as UDP function, just turn off
    cout << "Sender Thread Created!" << endl;
    char buffer[RES_SIZE];
    int sock = *((int*)socket);

    while (isClientAlive) {
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
        write(sock, buffer, sizeof(buffer));
        cout<<"frame "<<curRes.resID.i<<" res sent, "<<"marker#: "<<curRes.markerNum.i;
        cout<<" at "<<setprecision(15)<<wallclock()<<endl<<endl;
    }    

    cout<<"Sender Thread finished!"<<endl;
}

void *ThreadProcessFunction(void *param) {
    // Keep function - this is only called/created on the recognition server
    cout<<"Process Thread Created!"<<endl;
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
                cout << "Cache item added" << endl;
        }
        else {
            curRes.resID.i = frmID;
            curRes.markerNum.i = 0;
        }

        results.push(curRes);
    }
}

void *ThreadTCPOffloaderFunction(void *socket) {
    // Don't need to offload - work under the same single program
    cout << "Offloader Thread Created!" << endl;
    char tmp[4];
    char *requestbuffer;
    char *resultbuffer;
    int sock = *((int*)socket);

    while (isClientAlive) {
        if(offloadframes.empty()) {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        frameBuffer curFrame = offloadframes.front();
        offloadframes.pop();

        resBuffer offloadRequest; //it's not result, just utilize the structure
        offloadRequest.resID.i = curFrame.frmID;
        offloadRequest.resType.i = curFrame.dataType;
        offloadRequest.markerNum.i = curFrame.bufferSize;
        offloadRequest.buffer = curFrame.buffer;

        requestbuffer = (char *)malloc(curFrame.bufferSize+12);
        memset(requestbuffer, 0, curFrame.bufferSize+12);
        memcpy(requestbuffer, offloadRequest.resID.b, 4);
        memcpy(&(requestbuffer[4]), offloadRequest.resType.b, 4);
        memcpy(&(requestbuffer[8]), offloadRequest.markerNum.b, 4);
        memcpy(&(requestbuffer[12]), offloadRequest.buffer, offloadRequest.markerNum.i);
        cout<<"frame "<<curFrame.frmID<<" offloaded to server at "<<wallclock()<<endl;
        write(sock, requestbuffer, curFrame.bufferSize+12);
        free(requestbuffer);

        resultbuffer = (char *)malloc(RES_SIZE);
        memset(resultbuffer, 0, RES_SIZE);
        if(read(sock, resultbuffer, RES_SIZE) <= 0) {
            cout<<"recognition server disconnects"<<endl;
            isClientAlive = false;
            continue;
        }
        cout<<"frame "<<curFrame.frmID<<" res received from server at "<<wallclock()<<endl;

        resBuffer curRes;    
        memcpy(tmp, resultbuffer, 4);
        curRes.resID.i = *(int*)tmp;        
        memcpy(tmp, &(resultbuffer[4]), 4);
        curRes.resType.i = *(int*)tmp;
        memcpy(tmp, &(resultbuffer[8]), 4);
        curRes.markerNum.i = *(int*)tmp;
        //curRes.buffer = (char *)malloc(RES_SIZE-12);
        curRes.buffer = (char *)calloc(sizeof(char), RES_SIZE-12);
        memcpy(curRes.buffer, &(resultbuffer[12]), RES_SIZE-12);
        free(resultbuffer);

        results.push(curRes);
        
        // disabling adding cache items
        if(curRes.markerNum.i > 0)
            addCacheItem(curFrame, curRes);
    }    

    cout<<"Offloader Thread finished!"<<endl;
}

void *ThreadCacheSearchFunction(void *param) {
    // Keep function - make it work under one program
    cout<<"Cache Search Thread Created!"<<endl;
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
            cout << "Searching cache" << endl;
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


void runServer(int port) {
    pthread_t senderThread, receiverThread, imageProcessThread, processThread, offloadThread, testThread, annotationThread;
    char buffer[PACKET_SIZE];
    char fileid[4];
    int status = 0;
    int sockTCP, sockUDP, sockTCPCli;

    memset((char*)&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = htons(port);

    if((sockUDP = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        cout<<"ERROR opening udp socket"<<endl;
        exit(1);
    }
    if(bind(sockUDP, (struct sockaddr *)&localAddr, sizeof(localAddr)) < 0) {
        cout<<"ERROR on udp binding"<<endl;
        exit(1);
    }
    cout << endl << "========server started, waiting for clients==========" << endl;

    isClientAlive = true;
    pthread_create(&receiverThread, NULL, ThreadUDPReceiverFunction, (void *)&sockUDP);
    pthread_create(&senderThread, NULL, ThreadUDPSenderFunction, (void *)&sockUDP);
    pthread_create(&imageProcessThread, NULL, ThreadProcessFunction, NULL);
    pthread_create(&processThread, NULL, ThreadCacheSearchFunction, NULL);
    //pthread_create(&offloadThread, NULL, ThreadTCPOffloaderFunction, (void *)&sockTCPCli);

    pthread_join(receiverThread, NULL);
    pthread_join(senderThread, NULL);
    pthread_join(imageProcessThread, NULL);
    pthread_join(processThread, NULL);
    //pthread_join(offloadThread, NULL);
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
    //test();


    // outputting terminal outputs into dated log files
    // using namespace std;
    // string log_file = "logs/log_" + to_string(mode) + "_" + getCurrentDateTime("now") + ".txt";
    // string error_file = "logs/error_" + to_string(mode) + "_" + getCurrentDateTime("now") + ".txt";

    // freopen( log_file.c_str(), "w", stdout );
    // freopen( error_file.c_str(), "w", stderr );

    runServer(port);

    freeParams();
    return 0;
}
