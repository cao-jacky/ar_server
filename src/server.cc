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

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "helloworld.grpc.pb.h"
#endif

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;

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

ABSL_FLAG(uint16_t, port, 50051, "Server port for the service");

// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public Greeter::Service
{
    Status SayHello(ServerContext *context, const HelloRequest *request,
                    HelloReply *reply) override
    {
        std::string prefix("Hello ");
        reply->set_message(prefix + request->name());
        return Status::OK;
    }
};

void RunServer(uint16_t port)
{
    std::string server_address = absl::StrFormat("0.0.0.0:%d", port);
    GreeterServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

bool in_array(const string &value, const vector<string> &array)
{
    return find(array.begin(), array.end(), value) != array.end();
}

void load_online()
{
    ifstream file("../../data/onlineData.dat");
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

}


int main(int argc, char **argv)
{
    string service;
    int service_order;

    int query_size_factor = 3;
    int nn_num = 5;

    // Load service details from the JSON
    string service_details_path = "../../data/service_details_oakestra.json";
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
                    cout << "1" << endl;
                    load_online();
                    cout << "2" << endl;
                    load_images(online_images);
                    cout << "3" << endl;
                    load_params();

                    encodeDatabase(query_size_factor, nn_num);
                }

                // Start server code for service and the specified port
                run_server(service, service_order, service_ip, service_port);

                free_params();
            }
        }
    }

    // absl::ParseCommandLine(argc, argv);
    // RunServer(absl::GetFlag(FLAGS_port));
    return 0;
}
