#include <CL/sycl.hpp>
#include <iostream>

int main(){
    // Construct a queue
    auto q = sycl::queue();

    // Get the device associated with the queue
    auto device_associated = q.get_device();

    // Print the device name
    std::cout << "Device name is: " << device_associated.get_info<sycl::info::device::name>() << std::endl;

    return 0;

}