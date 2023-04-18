#include "reco.hpp"
#include "services/services.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdlib.h>
#include <thread>

using namespace std;
using json = nlohmann::json;

struct sockaddr_in local_addr;

string service;
int service_value;

vector<char *> online_images;
vector<char *> online_annotations;

void thread_udp_receiver(service_data *service_context)
{
    // struct service_data *curr_service = (struct service_data *)service_context;
    cout << service_context->name << endl;
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

void run_server(string service_name, string service_ip, int service_port)
{
    pthread_t receiver_thread, sender_thread;

    // Storing service data into struct to pass to relevant thread
    service_data curr_service;
    curr_service.name = service_name;
    curr_service.ip = service_ip;
    curr_service.port = service_port;

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

        thread receiver_thread(thread_udp_receiver, &curr_service);
        receiver_thread.join();

        // pthread_create(&receiver_thread, NULL, thread_udp_receiver, &curr_service);
        // pthread_join(receiver_thread, NULL);
    }
}

int main(int argc, char *argv[])
{
    string service;

    int query_size_factor = 3;
    int nn_num = 5;

    // Load service details from the JSON
    string service_details_path = "src/data/service_details_oakestra.json";
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
            if (argv[1] == curr_service_name)
            {
                service = curr_service_name;
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
                run_server(service, service_ip, service_port);

                free_params();
            }
        }
    }
    return 0;
}