#include<iostream>
#include<vector>
#include<CL/sycl.hpp>
#include<oneapi/dpl/algorithm>
#include<oneapi/dpl/execution>
#include<dpc_common.hpp>

int main(){
    constexpr size_t dataSize = 512;
    constexpr size_t num_iters = 50000;
    constexpr float C = 0.4;
    std::vector<float> u(dataSize, 100.0);
    std::vector<float> u_new(dataSize, 0.0);
    
    dpc_common::TimeInterval t_offload;
    sycl::queue q{};
    float *u_dev = sycl::malloc_device<float>(dataSize, q);
    auto e1 = q.memcpy(u_dev, u.data(), dataSize*sizeof(float));
    float *u_new_dev = sycl::malloc_device<float>(dataSize, q);
    auto e2 = q.memcpy(u_new_dev, u_new.data(), dataSize*sizeof(float));

    // FD scheme
    for (size_t it = 0; it < num_iters; it++)
    {    
        q.submit([&](sycl::handler& cgh){
            cgh.parallel_for(sycl::range<1>{dataSize}, [=](sycl::id<1> index){
                auto i = index[0];
                // u_new_acc[i] = u_acc[i] + C*(u_acc[i+1] - 2*u_acc[i] + u_acc[i-1]); 
                u_new_dev[i] = u_dev[i] + C*(u_dev[i+1] - 2*u_dev[i] + u_dev[i-1]);    

            });

        });
        q.wait_and_throw();
        // Boundary conditions: Dirichlet: one end at 100 other at 0
        u_new_dev[0] = 100.0;
        u_new_dev[dataSize-1] = 0.0;
        q.submit([&](sycl::handler& cgh){
            cgh.parallel_for(sycl::range<1>{dataSize}, [=](sycl::id<1> index){
                auto i = index[0];
                u_dev[i] = u_new_dev[i];   

            });
        });
        q.wait_and_throw();
    }

    q.memcpy(u_new.data(), u_new_dev, dataSize*sizeof(float));
    std::cout << "Time elapsed: " << t_offload.Elapsed() << " secs" << std::endl;

    for (const auto& i:u_new){std::cout << i << std::endl;}
    

    return 0;
}