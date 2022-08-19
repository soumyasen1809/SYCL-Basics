#include<iostream>
#include<CL/sycl.hpp>

int main(){
    for (const sycl::platform& platform : sycl::platform::get_platforms())
    {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;

        for (const sycl::device& device : platform.get_devices())
        {
            std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
        }
        
    }

    sycl::queue q{sycl::host_selector{}};
    std::cout << "Device selected now: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    return 0;
}