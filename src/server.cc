#include "reco.hpp"
#include "services/services.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <algorithm>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <stdlib.h>
#include <thread>
#include <queue>
#include <filesystem>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#ifdef BAZEL_BUILD
#include "frame.grpc.pb.h"
#else
#include "frame.grpc.pb.h"
#endif

using grpc::Channel;
using grpc::ClientContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

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
queue<inter_service_buffer> results_frames;

Frame MakeFrame(string client, string id, string qos, char *data, int data_size)
{
    Frame f;
    f.set_client(client);
    f.set_id(id);
    f.set_qos(qos);
    f.set_data(data, data_size);
    return f;
}

// Implementation of the classes for gRPC server and client behavior.
class QueueImpl final : public QueueService::Service
{
    Status NextFrame(ServerContext *context, const Frame *request,
                     Frame *reply) override
    {
        string curr_client = request->client();
        string curr_id = request->id();
        string curr_qos = request->qos();

        // received data from gRPC server, will store into relevant data structure
        string curr_data = request->data();

        char tmp[4];
        char tmp_ip[16];

        frame_buffer curr_frame;

        memcpy(tmp, curr_data.c_str(), 4);
        tmp[4] = '\0';
        curr_frame.client_id = tmp;

        memcpy(tmp, &(curr_data.c_str()[4]), 4);
        curr_frame.frame_no = *(int *)tmp;

        memcpy(tmp, &(curr_data.c_str()[8]), 4);
        curr_frame.data_type = *(int *)tmp;

        memcpy(tmp, &(curr_data.c_str()[12]), 4);
        curr_frame.buffer_size = *(int *)tmp;
        int buffer_size = curr_frame.buffer_size;

        memcpy(tmp_ip, &(curr_data.c_str()[16]), 16);
        tmp_ip[16] = '\0';
        curr_frame.client_ip = tmp_ip;

        memcpy(tmp, &(curr_data.c_str()[32]), 4);
        curr_frame.client_port = *(int *)tmp;

        // selecting out sift buffer size, and sift data is buffer size > 0
        memcpy(tmp, &(curr_data.c_str()[40]), 4);
        int sift_buffer_size = *(int *)tmp;
        curr_frame.sift_buffer_size = sift_buffer_size;
        if (sift_buffer_size > 0)
        {
            curr_frame.sift_buffer = (char *)malloc(sift_buffer_size);
            memset(curr_frame.sift_buffer, 0, sift_buffer_size);
            memcpy(curr_frame.sift_buffer, &(curr_data.c_str()[44 + buffer_size]), sift_buffer_size);
        }

        print_log(service, curr_frame.client_id, to_string(curr_frame.frame_no), "Frame " + to_string(curr_frame.frame_no) + " received and has a service buffer size of " + to_string(buffer_size) + " Bytes and a sift buffer size of " + to_string(sift_buffer_size) + " for client with IP " + curr_frame.client_ip + " and port " + to_string(curr_frame.client_port));

        // copy frame image data into buffer
        curr_frame.buffer = (char *)malloc(buffer_size);
        memset(curr_frame.buffer, 0, buffer_size);
        memcpy(curr_frame.buffer, &(curr_data.c_str()[44]), buffer_size);

        frames.push(curr_frame);

        return Status::OK;
    }
};

class QueueClient
{
public:
    QueueClient(shared_ptr<Channel> channel)
        : stub_(QueueService::NewStub(channel)) {}

    void NextFrame(string client, string id, string qos, char *data, int data_size)
    {
        // Prepare data to be sent to next service
        Frame request;
        request = MakeFrame(client, id, qos, data, data_size);

        Frame reply;
        ClientContext context;
        Status status = stub_->NextFrame(&context, request, &reply);

        if (status.ok())
        {
            print_log("", "0", "0", "Receiving gRPC server sent an ok status, data transmitted succesfully");
        }
        else
        {
            cout << status.error_code() << ": " << status.error_message()
                      << endl;
            // print_log(curr_service, "0", "0", "[ERROR]" + status.error_code() + ": " + status.error_message());
            // return "RPC failed";
        }
    }

private:
    unique_ptr<QueueService::Stub> stub_;
};

void thread_udp_receiver(service_data *service_context)
{
    string curr_service = service_context->name;
    print_log(curr_service, "0", "0", "Thread created to receive data sent with UDP");

    char tmp[4];
    char buffer[44 + PACKET_SIZE];

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
    string curr_service = service_context->name;
    int curr_service_order = service_context->order;
    int curr_service_port = service_context->port;

    int next_service_port = curr_service_port + 1;

    print_log(curr_service, "0", "0", "Thread created to send results data to the client with UDP");

    while (1)
    {
        if (results_frames.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        inter_service_buffer curr_res = results_frames.front();
        results_frames.pop();

        char tmp[4];
        int buffer_size = curr_res.buffer_size.i;

        int next_service_socket;
        struct sockaddr_in next_service_sock;
        struct sockaddr_in client_addr;
        if ((next_service_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        {
            perror("socket creation failed");
            exit(EXIT_FAILURE);
        }
        memset((char *)&next_service_sock, 0, sizeof(next_service_sock));

        // Filling server information
        next_service_sock.sin_family = AF_INET;
        next_service_sock.sin_port = htons(0);
        next_service_sock.sin_addr.s_addr = htonl(INADDR_ANY);

        // Forcefully attaching socket to the port 8080
        if (bind(next_service_socket, (struct sockaddr *)&next_service_sock,
                 sizeof(next_service_sock)) < 0)
        {
            print_log(service, "0", "0", "ERROR: Unable to bind UDP");
            exit(1);
        }

        char buffer[16 + (100 * buffer_size)];
        memset(buffer, 0, sizeof(buffer));

        memcpy(buffer, curr_res.client_id.c_str(), 4);
        memcpy(&(buffer[4]), curr_res.frame_no.b, 4);
        memcpy(&(buffer[12]), curr_res.buffer_size.b, 4);
        if (buffer_size != 0)
        {
            memcpy(&(buffer[16]), curr_res.results_buffer, 100 * buffer_size);
        }

        memcpy(tmp, curr_res.frame_no.b, 4);
        int frame_no = *(int *)tmp;

        string client_return_ip = curr_res.client_ip;

        memcpy(tmp, curr_res.client_port.b, 4);
        tmp[4] = '\0';
        int client_return_port = *(int *)tmp;

        client_addr.sin_family = AF_INET;
        client_addr.sin_addr.s_addr = inet_addr(client_return_ip.c_str());
        client_addr.sin_port = htons(client_return_port);

        inet_pton(AF_INET, client_return_ip.c_str(), &(client_addr.sin_addr));

        int udp_status = sendto(next_service_socket, buffer, sizeof(buffer), 0, (struct sockaddr *)&client_addr, sizeof(client_addr));
        print_log(service, curr_res.client_id, to_string(frame_no), "Results for Frame " + to_string(frame_no) + " sent to client with number of markers of " + to_string(buffer_size) + " where client has details of IP " + client_return_ip + " and port " + to_string(client_return_port));

        close(next_service_socket);
        if (udp_status == -1)
        {
            printf("Error sending: %i\n", errno);
        }
    }
}

void thread_processor(service_data *service_context)
{
    string curr_service = service_context->name;
    int curr_service_order = service_context->order;

    print_log(curr_service, "0", "0", "Thread created to process data that has been pushed to frames buffer");

    // void array of functions relating to data types
    void (*processing_functions[5])(string, int, frame_buffer) = {primary_processing, sift_processing, encoding_processing, lsh_processing, matching_processing};

    while (1)
    {
        if (frames.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        frame_buffer curr_frame = frames.front();
        frames.pop();

        // call appropiate function with 0-indexed selection
        (*processing_functions[curr_service_order - 1])(curr_service, curr_service_order, curr_frame);
    }
}

void thread_sender(service_data *service_context)
{
    string curr_service = service_context->name;
    int curr_service_order = service_context->order;
    int curr_service_port = service_context->port;

    int next_service_port = curr_service_port + 1;

    print_log(curr_service, "0", "0", "Thread created to use gRPC to send data that has been pushed to inter-frames buffer");

    while (1)
    {
        if (inter_service_data.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        inter_service_buffer curr_item = inter_service_data.front();
        inter_service_data.pop();

        int data_buffer_size = curr_item.buffer_size.i;
        int buffer_size = 60 + data_buffer_size;

        int sift_buffer_size = 0;
        if (curr_service != "primary")
        {
            // setting buffer size according to the SIFT data required to carry throughout the services
            sift_buffer_size = curr_item.sift_buffer_size.i;
            buffer_size += sift_buffer_size;
        }

        char buffer[buffer_size];
        memset(buffer, 0, sizeof(buffer));

        if (curr_service == "primary")
        {
            memcpy(&(buffer[44]), &(curr_item.image_buffer)[0], data_buffer_size);
        }
        else
        {
            // store sift buffer size and then the sift data itself
            memcpy(&(buffer[40]), curr_item.sift_buffer_size.b, 4);
            memcpy(&(buffer[44 + data_buffer_size]), curr_item.sift_buffer, sift_buffer_size);

            // store main buffer data
            memcpy(&(buffer[44]), &(curr_item.buffer)[0], data_buffer_size);
        }

        memcpy(buffer, curr_item.client_id.c_str(), 4);
        memcpy(&(buffer[4]), curr_item.frame_no.b, 4);
        memcpy(&(buffer[8]), curr_item.data_type.b, 4);
        memcpy(&(buffer[12]), curr_item.buffer_size.b, 4);
        memcpy(&(buffer[16]), curr_item.client_ip.c_str(), 16);
        memcpy(&(buffer[32]), curr_item.client_port.b, 4);
        memcpy(&(buffer[36]), curr_item.previous_service.b, 4);

        string next_service_grpc_str = "localhost:" + to_string(next_service_port);

        QueueClient QueueService(
            grpc::CreateChannel(next_service_grpc_str, grpc::InsecureChannelCredentials()));
        QueueService.NextFrame(curr_item.client_id, "1", "1", buffer, buffer_size);
        print_log(curr_service, curr_item.client_id, to_string(curr_item.frame_no.i), "Frame " + to_string(curr_item.frame_no.i) + " offloaded to gRPC for transmission to the next service for later processing - the frame has a total payload size of " + to_string(buffer_size) + " which includes next service buffer size of " + to_string(data_buffer_size) + " Bytes and sift buffer size of " + to_string(sift_buffer_size) + " Bytes");
    }
}

void RunServer(service_data *service_context)
{
    string curr_service = service_context->name;
    // int curr_service_port = service_context->port;

    // hardcoding port to be 5000 to be able to communicate with the Oakestra queue
    int curr_service_port = 5000; 

    string server_address = absl::StrFormat("0.0.0.0:%d", curr_service_port);
    QueueImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    builder.RegisterService(&service);

    unique_ptr<Server> server(builder.BuildAndStart());
    print_log(curr_service, "0", "0", "Thread created to run gRPC server listening locally on port " + to_string(curr_service_port));
    // cout << "Server listening on " << server_address << endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

bool in_array(const string &value, const vector<string> &array)
{
    return find(array.begin(), array.end(), value) != array.end();
}

void load_online(string binary_directory)
{
    ifstream file(binary_directory + "/../../data/onlineData.dat");
    string line;
    int i = 0;
    while (getline(file, line))
    {
        string corr_path = binary_directory + "/../../";
        line = corr_path.append(line);

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
    // Storing service data into struct to pass to relevant thread
    service_data curr_service;
    curr_service.name = service_name;
    curr_service.order = service_order;
    curr_service.ip = service_ip;
    curr_service.port = service_port;

    thread processor_thread(thread_processor, &curr_service);
    thread sender_thread(thread_sender, &curr_service);
    thread grpc_thread(RunServer, &curr_service);

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
    else if (service_name == "matching")
    {
        thread sender_udp_thread(thread_udp_sender, &curr_service);
        sender_udp_thread.join();
    }

    processor_thread.join();
    grpc_thread.join();
    sender_thread.join();

    // start the gRPC listening server
    // RunServer(service_name, service_order, service_ip, service_port);
    // pthread_t receiver_udp_thread, sender_thread, processor_thread;
}

int main(int argc, char **argv)
{
    string service;
    int service_order;

    int query_size_factor = 3;
    int nn_num = 5;

    filesystem::path binary_path = filesystem::absolute(argv[0]);
    filesystem::path binary_directory = binary_path.parent_path();
    string binary_directory_string = binary_directory.string();

    // Load service details from the JSON
    string service_details_path =  binary_directory_string + "/../../data/service_details_oakestra.json";
    
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
                // print_log(service, "0", "0", "The provided IP of the primary module is " + string(argv[2]));

                // Perform service preprocessing if required, i.e., initial variable loading and encoding
                bool preprocesisng = val["preprocessing"];
                if (preprocesisng)
                {
                    load_online(binary_directory_string);
                    load_images(binary_directory_string, online_images);
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
