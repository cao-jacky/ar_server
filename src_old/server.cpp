#include "reco.hpp"
#include "services/services.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <stdlib.h>
#include <thread>
#include <queue>

#define PACKET_SIZE 60000

using namespace std;
using json = nlohmann::json;

struct sockaddr_in local_addr;

string service;
int service_value;

vector<char *> online_images;
vector<char *> online_annotations;

queue<frame_buffer> frames;
queue<inter_service_buffer> inter_service_data;

void thread_udp_receiver(service_data *service_context)
{
    string curr_service = service_context->name;
    print_log(curr_service, "0", "0", "Thread created to receive data sent with UDP");

    char tmp[4];
    char buffer[60 + PACKET_SIZE];

    struct sockaddr_in remoteAddr;
    socklen_t addrlen = sizeof(remoteAddr);

    int udp_socket = service_context->udp_socket;

    while (1)
    {
        frame_buffer curr_frame;

        char client_id[4];

        memset(buffer, 0, sizeof(buffer));
        recvfrom(udp_socket, buffer, PACKET_SIZE, 0, (struct sockaddr *)&remoteAddr, &addrlen);

        char *device_ip = inet_ntoa(remoteAddr.sin_addr);
        int device_port = htons(remoteAddr.sin_port);

        // select out data from pre-determined format
        memcpy(client_id, &(buffer[0]), 4);
        client_id[4] = '\0';

        char *curr_client = (char *)client_id;
        curr_frame.client_id = curr_client;

        memcpy(tmp, &(buffer[4]), 4);
        tmp[4] = '\0';
        curr_frame.frame_no = *(int *)tmp;

        memcpy(tmp, &(buffer[8]), 4);
        tmp[4] = '\0';
        curr_frame.data_type = *(int *)tmp;
        int curr_data_type = curr_frame.data_type;

        memcpy(tmp, &(buffer[12]), 4);
        curr_frame.buffer_size = *(int *)tmp;

        // store client IP and port details
        curr_frame.client_ip = device_ip;
        curr_frame.client_port = device_port;

        // pass frame data to appropiate function depending in curr_data_type
        if (curr_data_type >= 0)
        {
            // client_echo(udp_socket, curr_service, curr_frame);
            if (curr_data_type == 0)
            {
                client_echo(curr_service, curr_frame);
            }
            else if (curr_data_type == 1)
            {
                client_preprocessing_request(curr_service, curr_frame, buffer);
            }
        }
    }
}

void thread_udp_sender(service_data *service_context)
{
}

void thread_processor(service_data *service_context)
{
    string curr_service = service_context->name;
    int curr_service_order = service_context->order;

    print_log(curr_service, "0", "0", "Thread created to process data that has been pushed to frames buffer");

    // void array of functions relating to data types
    void (*processing_functions[1])(string, int, frame_buffer) = {primary_processing};

    while (1)
    {
        if (frames.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        frame_buffer curr_frame = frames.front();
        frames.pop();

        // call appropiate function
        (*processing_functions[curr_service_order])(curr_service, curr_service_order, curr_frame);
    }
}

void thread_grpc_sender()
{
}

void thread_grpc_receiver()
{
}

bool in_array(const string &value, const vector<string> &array)
{
    return find(array.begin(), array.end(), value) != array.end();
}

void load_online()
{
    ifstream file("data/onlineData.dat");
    string line;
    int i = 0;
    while (getline(file, line))
    {
        char *file_name = new char[256];
        strcpy(file_name, line.c_str());

        if (i % 2 == 0)
            online_images.push_back(file_name);
        else
            online_annotations.push_back(file_name);
        ++i;
    }
    file.close();
}

void run_server(string service_name, int service_order, string service_ip, int service_port)
{
    // pthread_t receiver_udp_thread, sender_thread, processor_thread;

    // Storing service data into struct to pass to relevant thread
    service_data curr_service;
    curr_service.name = service_name;
    curr_service.order = service_order;
    curr_service.ip = service_ip;
    curr_service.port = service_port;

    thread processor_thread(thread_processor, &curr_service);

    // Start UDP listener only for primary service
    if (service_name == "primary")
    {
        int udp_socket;

        memset((char *)&local_addr, 0, sizeof(local_addr));
        local_addr.sin_family = AF_INET;
        local_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        local_addr.sin_port = htons(service_port);

        if ((udp_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        {
            print_log(service, "0", "0", "[ERROR] Unable to open UDP socket");
            exit(1);
        }
        if (bind(udp_socket, (struct sockaddr *)&local_addr, sizeof(local_addr)) < 0)
        {
            print_log(service, "0", "0", "[ERROR] Unable to bind UDP");
            exit(1);
        }
        curr_service.udp_socket = udp_socket;

        thread receiver_udp_thread(thread_udp_receiver, &curr_service);
        receiver_udp_thread.join();

        // pthread_create(&receiver_thread, NULL, thread_udp_receiver, &curr_service);
        // pthread_join(receiver_thread, NULL);
    }

    processor_thread.join();
}

int main(int argc, char *argv[])
{
    string service;
    int service_order;

    int query_size_factor = 3;
    int nn_num = 5;

    // Load service details from the JSON
    string service_details_path = "data/service_details_oakestra.json";
    ifstream sdd(service_details_path);
    json sdd_json = json::parse(sdd);
    json service_details = sdd_json[0]["services"];

    // Check if help called as an argument
    vector<string> help_strings{"help", "h"};
    if (in_array(argv[1], help_strings))
    {
        cout << "Distributed Augmented Reality Server Application\n"
             << endl;
    }

    if (argc == 1)
    {
        print_log("", "0", "0", "No arguments were passed");
    }
    else
    {
        for (auto &[key, val] : service_details.items())
        {
            string curr_service_name = val["service_name"];
            int curr_service_order = val["order"];
            if (argv[1] == curr_service_name)
            {
                service = curr_service_name;
                service_order = curr_service_order;
                string service_ip = val["server"]["ip"];
                int service_port = val["server"]["port"];

                print_log(service, "0", "0", "Selected service is " + service + " and the IP of it is " + service_ip);
                print_log(service, "0", "0", "The provided IP of the primary module is " + string(argv[2]));

                // Perform service preprocessing if required, i.e., initial variable loading and encoding
                bool preprocesisng = val["preprocessing"];
                if (preprocesisng)
                {
                    load_online();
                    load_images(online_images);
                    load_params();

                    encodeDatabase(query_size_factor, nn_num);
                }

                // Start server code for service and the specified port
                run_server(service, service_order, service_ip, service_port);

                free_params();
            }
        }
    }
    return 0;
}