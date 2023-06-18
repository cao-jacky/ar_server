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

string curr_service;
int curr_service_order;
int curr_service_port;

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

class QueueImpl final : public QueueService::Service
{
    Status NextFrame(ServerContext *context, const Frame *request,
                     Frame *reply) override
    {
        string curr_client = request->client();
        string curr_id = request->id();
        string curr_qos = request->qos();

        int next_service_port = curr_service_port + 1;

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

        print_log(curr_service, curr_frame.client_id, to_string(curr_frame.frame_no), "Frame " + to_string(curr_frame.frame_no) + " received and has a service buffer size of " + to_string(buffer_size) + " Bytes and a sift buffer size of " + to_string(sift_buffer_size) + " for client with IP " + curr_frame.client_ip + " and port " + to_string(curr_frame.client_port));

        // copy frame image data into buffer
        curr_frame.buffer = (char *)malloc(buffer_size);
        memset(curr_frame.buffer, 0, buffer_size);
        memcpy(curr_frame.buffer, &(curr_data.c_str()[44]), buffer_size);

        // frames.push(curr_frame);

        // void (*processing_functions[5])(string, int, frame_buffer, inter_service_buffer&) = {primary_processing, sift_processing, encoding_processing, lsh_processing, matching_processing};

        // call appropiate function with 0-indexed selection
        inter_service_buffer results_frame;

        if (curr_service == "primary")
        {
            primary_processing(curr_service, curr_service_order, curr_frame, results_frame);
        }
        else if (curr_service == "sift")
        {
            sift_processing(curr_service, curr_service_order, curr_frame, results_frame);
        }
        else if (curr_service == "encoding")
        {
            encoding_processing(curr_service, curr_service_order, curr_frame, results_frame);
        }
        else if (curr_service == "lsh")
        {
            lsh_processing(curr_service, curr_service_order, curr_frame, results_frame);
        }
        else if (curr_service == "matching")
        {
            matching_processing(curr_service, curr_service_order, curr_frame, results_frame);
        }
        // (*processing_functions[curr_service_order - 1])(curr_service, curr_service_order, curr_frame, results_frame);

        int to_send_data_buffer_size = results_frame.buffer_size.i;
        int to_send_buffer_size = 60 + to_send_data_buffer_size;

        int to_send_sift_buffer_size = 0;
        if (curr_service != "primary" && curr_service != "matching")
        {
            // setting buffer size according to the SIFT data required to carry throughout the services
            to_send_sift_buffer_size = results_frame.sift_buffer_size.i;
            to_send_buffer_size += to_send_sift_buffer_size;
        }
        else if (curr_service == "matching")
        {
            to_send_buffer_size = 16 + (100 * to_send_data_buffer_size);
        }

        char buffer[to_send_buffer_size];
        memset(buffer, 0, sizeof(buffer));

        if (curr_service == "primary")
        {
            memcpy(&(buffer[44]), &(results_frame.image_buffer)[0], to_send_data_buffer_size);
        }
        else if (curr_service == "matching")
        {
            memcpy(buffer, results_frame.client_id.c_str(), 4);
            memcpy(&(buffer[4]), results_frame.frame_no.b, 4);
            memcpy(&(buffer[12]), results_frame.buffer_size.b, 4);
            if (to_send_data_buffer_size != 0)
            {
                memcpy(&(buffer[16]), &(results_frame.results_buffer)[0], 100 * to_send_data_buffer_size);
            }
        }
        else if (curr_service != "matching" and curr_service != "primary")
        {
            // store sift buffer size and then the sift data itself
            memcpy(&(buffer[40]), results_frame.sift_buffer_size.b, 4);
            memcpy(&(buffer[44 + to_send_data_buffer_size]), results_frame.sift_buffer, to_send_sift_buffer_size);

            // store main buffer data
            memcpy(&(buffer[44]), &(results_frame.buffer)[0], to_send_data_buffer_size);
        }

        if (curr_service != "matching")
        {
            memcpy(buffer, results_frame.client_id.c_str(), 4);
            memcpy(&(buffer[4]), results_frame.frame_no.b, 4);
            memcpy(&(buffer[8]), results_frame.data_type.b, 4);
            memcpy(&(buffer[12]), results_frame.buffer_size.b, 4);
            memcpy(&(buffer[16]), results_frame.client_ip.c_str(), 16);
            memcpy(&(buffer[32]), results_frame.client_port.b, 4);
            memcpy(&(buffer[36]), results_frame.previous_service.b, 4);
        }

        char *buffer_pointer = buffer;
        // reply->set_data(buffer_pointer, to_send_buffer_size);

        // Comment this code if needed to test locally, else, use "reply->set_data"
        // cout << next_service_grpc_str << endl;
        // QueueClient QueueService(
        //     grpc::CreateChannel(next_service_grpc_str, grpc::InsecureChannelCredentials()));
        // QueueService.NextFrame(results_frame.client_id, "1", "1", buffer, to_send_buffer_size);

        print_log(curr_service, results_frame.client_id, to_string(results_frame.frame_no.i), "Frame " + to_string(results_frame.frame_no.i) + " offloaded to gRPC for transmission to the next service for later processing - the frame has a total payload size of " + to_string(to_send_buffer_size) + " which includes next service buffer size of " + to_string(to_send_data_buffer_size) + " Bytes and sift buffer size of " + to_string(to_send_sift_buffer_size) + " Bytes");

        return Status::OK;
    }
};

void RunServer(service_data *service_context)
{
    string curr_service = service_context->name;

    // Comment when developing locally
    // int curr_service_port = service_context->port;

    // Hardcoding port 5000 to communicate with the Oakestra queue
    int curr_service_port = 5000;

    string server_address = absl::StrFormat("0.0.0.0:%d", curr_service_port);
    QueueImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    builder.RegisterService(&service);

    unique_ptr<Server> server(builder.BuildAndStart());
    print_log(curr_service, "0", "0", "Thread created to run gRPC server listening locally on port " + to_string(curr_service_port));

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

    thread grpc_thread(RunServer, &curr_service);
    grpc_thread.join();
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
    string service_details_path = binary_directory_string + "/../../data/service_details_oakestra.json";

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
                curr_service = service;

                service_order = curr_service_order;
                curr_service_order = service_order;

                // string service_ip = val["server"]["ip"];
                string service_ip = "0.0.0.0";
                int service_port = val["server"]["port"];
                curr_service_port = service_port;

                print_log(service, "0", "0", "Selected service is " + service + " and the IP of it is " + service_ip);

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
