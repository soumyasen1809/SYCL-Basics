#include<CL/sycl.hpp>
#include<iostream>

int main(){
    constexpr size_t dataSize = 16;
    constexpr size_t wgSize = 2;

    int a[dataSize], b[dataSize];
    for (size_t i = 0; i < dataSize; i++){
        a[i] = 2*i;
        b[i] = 0;
    }

    auto q = sycl::queue();
    
    auto *a_dev = sycl::malloc_device<int>(dataSize, q);
    auto e1 = q.memcpy(a_dev, a, sizeof(int)*dataSize);

    auto *b_dev = sycl::malloc_device<int>(dataSize, q);

    auto nd_range = sycl::nd_range(sycl::range(dataSize), sycl::range(wgSize));
    sycl::buffer<int, 1> buff(a, sycl::range<1>(wgSize));
    q.submit([&](sycl::handler &cgh){
        // sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>(wgSize), cgh);
        // Above line is 1 line alterative implmentation of the use of buffer (2 lines)
        auto local_mem = buff.get_access<sycl::access::mode::read_write>(cgh);
        // In the cgh.parallel_for dont write the dependency on {e1}
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item){
            auto localId = item.get_local_linear_id();
            auto globalId = item.get_global_linear_id();
            auto groupId = item.get_group_linear_id();
            auto num_groups = item.get_group_range(0);

            local_mem[localId] = a_dev[globalId];
            item.barrier();
            b_dev[(num_groups-groupId-1)*wgSize+localId] = local_mem[wgSize-localId-1];
        });
    }).wait(); 
    
    q.memcpy(b, b_dev, sizeof(int)*dataSize).wait();

    for (const auto& i:b){std::cout << i << std::endl;}
    

    return 0;
}