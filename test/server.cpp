#include <iostream>
#include <vector>
#include <cstring>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <iomanip>
#include <queue>
#include <fstream>

#define PACKET_SIZE 50000
#define RES_SIZE 512

using namespace std;

double ts[5], te[5], tt[5];

double wallclock (void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

void ThreadReceiverFunction(int sock, int i) {
    cout<<"Receiver Thread "<<i<<" Created!"<<endl;
    char buffer[RES_SIZE];
    int resCount = 0;
    int markerCount = 0; 

    while (1) {
            memset(buffer, 0, sizeof(buffer));
            recvfrom(sock, buffer, PACKET_SIZE, 0, NULL, NULL);
            int markerNum = *(int*)&(buffer[8]);
            markerCount += markerNum;
            resCount++; 
            te[i] = wallclock();
            tt[i] += te[i] - ts[i];
            cout<<"stream "<<i<<" received with "<<markerNum<<" markers, in total "<<markerCount<<"/"<<resCount<<" "<<tt[i]<<endl;
    }
}

void ThreadSenderFunction(int *sock, struct sockaddr_in *remoteAddr, int i) {
    cout<<"Sender Thread "<<i<<" Created!"<<endl;
    char buffer[PACKET_SIZE];

    ifstream in("request", ios::in | ios::binary);
    in.read(buffer, 48713);
    in.close();

    for(int test = 0; test < 1000; test++) {
        this_thread::sleep_for(chrono::milliseconds(200));

        sendto(sock[i], buffer, 48713, 0, (struct sockaddr *)&remoteAddr[i], sizeof(remoteAddr[i]));
        cout<<"request sent to container "<<i<<endl;
        ts[i] = wallclock();
    }    
}

int main(int argc, char *argv[]) {
    struct sockaddr_in localAddr[5];
    struct sockaddr_in remoteAddr[5];
    int sockUDP[5];
    thread senderThread[5]; 
    thread receiverThread[5];

    for(int i = 0; i < 5; i++) {
        memset((char*)&localAddr[i], 0, sizeof(localAddr[i]));
        localAddr[i].sin_family = AF_INET;
        localAddr[i].sin_addr.s_addr = htonl(INADDR_ANY);
        int port = 51710+i;
        localAddr[i].sin_port = htons(port);

        if((sockUDP[i] = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
            cout<<"ERROR opening udp socket"<<endl;
            exit(1);
        }
        if(bind(sockUDP[i], (struct sockaddr *)&localAddr[i], sizeof(localAddr[i])) < 0) {
            cout<<"ERROR on udp binding"<<endl;
            exit(1);
        }
    }

    struct hostent *hp;
    hp = gethostbyname("192.168.50.223");
    for(int i = 0; i < 5; i++) {
        memset((char*)&remoteAddr[i], 0, sizeof(remoteAddr[i]));
        remoteAddr[i].sin_family = AF_INET;
        int port = 51717+i;
        remoteAddr[i].sin_port = htons(port);
        memcpy((void*)&remoteAddr[i].sin_addr, hp->h_addr_list[0], hp->h_length);
    }

    for(int i = 0; i < 5; i++) { 
        receiverThread[i] = thread(ThreadReceiverFunction, sockUDP[i], i);
        senderThread[i] = thread(ThreadSenderFunction, sockUDP, remoteAddr, i);
    }

    for(int i = 0; i < 5; i++) {
        receiverThread[i].join();
        senderThread[i].join();
    }
}
