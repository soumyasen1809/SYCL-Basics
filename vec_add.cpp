#include<iostream>
#include<vector>
#include<CL/sycl.hpp>
#include<dpc_common.hpp>

int main(){
    constexpr size_t dataSize = 1024*1024;
    constexpr size_t wgSize = 64;

    std::vector<int> a, b, c;
    for (size_t i = 0; i < dataSize; i++)
    {
        a.push_back(i);
        b.push_back(2*i);
        c.push_back(0);
    }

    // auto t1 = std::chrono::steady_clock::now();  // Can use chrono for time measurement
    dpc_common::TimeInterval t_offload;     // Can use dpc_common for time measurement

    sycl::queue q{};

    auto *a_dev = sycl::malloc_device<int>(dataSize, q);
    auto *b_dev = sycl::malloc_device<int>(dataSize, q);
    auto *c_dev = sycl::malloc_device<int>(dataSize, q);

    auto e1 = q.memcpy(a_dev, a.data(), sizeof(int)*dataSize);
    auto e2 = q.memcpy(b_dev, b.data(), sizeof(int)*dataSize);

    auto nd_range = sycl::nd_range(sycl::range(dataSize), sycl::range(wgSize));

    q.submit([&](sycl::handler &h){
        sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> local_a(sycl::range<1>(wgSize), h);
        sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> local_b(sycl::range<1>(wgSize), h);
        sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> local_c(sycl::range<1>(wgSize), h);
        h.parallel_for(nd_range, [=](sycl::nd_item<1> item){
            auto localId = item.get_local_linear_id();
            auto globalId = item.get_global_linear_id();
            auto groupId = item.get_group_linear_id();
            auto num_groups = item.get_group_range(0);

            local_a[localId] = a_dev[globalId];
            local_b[localId] = b_dev[globalId];

            local_c[localId] = local_a[localId] + local_b[localId];

            c_dev[globalId] = local_c[localId];
        });
    }).wait();

    
    q.memcpy(c.data(), c_dev, sizeof(int)*dataSize).wait();
    q.wait_and_throw();

    // auto t2 = std::chrono::steady_clock::now();   // Stop timing for chrono
    // std::cout << "Time elapsed:" << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microsecs" << std::endl;
    auto device_time = t_offload.Elapsed();     // Stop timing for dpc_common
    std::cout << "Device Offload time: " << device_time << " sec" << std::endl;

    // for(const auto& i:c){std::cout << i << std::endl;}
    return 0;
}