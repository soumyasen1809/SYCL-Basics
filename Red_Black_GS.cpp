// https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec14.pdf

#include<iostream>
#include<vector>
#include<CL/sycl.hpp>
#include<dpc_common.hpp>
#include<oneapi/dpl/algorithm>
#include<oneapi/dpl/execution>

int main(){
    constexpr size_t dataSize = 1024*8;

    sycl::queue q{};
    auto *u_old = sycl::malloc_shared<float>(dataSize*dataSize, q);
    auto *u_new = sycl::malloc_shared<float>(dataSize*dataSize, q);
    auto *b_vec = sycl::malloc_shared<float>(dataSize*dataSize, q);

    // parallel initialization of vectors
    sycl::buffer<float, 1> u_old_buff(u_old, sycl::range<1>{dataSize*dataSize}, sycl::property::buffer::use_host_ptr());
    sycl::buffer<float, 1> u_new_buff(u_new, sycl::range<1>{dataSize*dataSize}, sycl::property::buffer::use_host_ptr());
    sycl::buffer<float, 1> b_vec_buff(b_vec, sycl::range<1>{dataSize*dataSize}, sycl::property::buffer::use_host_ptr());
    q.submit([&](sycl::handler& cgh){
        auto u_old_acc = u_old_buff.get_access<sycl::access::mode::write>(cgh);
        auto u_new_acc = u_new_buff.get_access<sycl::access::mode::write>(cgh);
        auto b_vec_acc = b_vec_buff.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::range<2>{dataSize, dataSize}, [=](sycl::id<2> index){
            auto i = index[0];
            auto j = index[1];
            u_old_acc[i*dataSize+j] = (i*i) + (2*j);
            u_new_acc[i*dataSize+j] = 0.0;
            b_vec_acc[i*dataSize+j] = i+j;
        });
    });
    q.wait();

    auto time_elapsed = dpc_common::TimeInterval();
    // Parallel Red Black GS method
    q.submit([&](sycl::handler& cgh){
        auto u_old_acc = u_old_buff.get_access<sycl::access::mode::read>(cgh);
        auto u_new_acc = u_new_buff.get_access<sycl::access::mode::write>(cgh);
        auto b_vec_acc = b_vec_buff.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<2>{dataSize, dataSize}, [=](sycl::id<2> index){
            auto i = index[0];
            auto j = index[1];
            // Red color
            if ((i+j)%2 == 0)
            {
                u_new_acc[i*dataSize+j] = 0.25*(u_old_acc[i*dataSize+(j+1)] + u_old_acc[i*dataSize+(j-1)] + u_old_acc[(i+1)*dataSize+(j)] + u_old_acc[(i-1)*dataSize+(j)]) - 0.25*(b_vec_acc[i*dataSize+j]);
            }
            
        });
    });
    q.wait();
    q.submit([&](sycl::handler& cgh){
        auto u_old_acc = u_old_buff.get_access<sycl::access::mode::read>(cgh);
        auto u_new_acc = u_new_buff.get_access<sycl::access::mode::write>(cgh);
        auto b_vec_acc = b_vec_buff.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<2>{dataSize, dataSize}, [=](sycl::id<2> index){
            auto i = index[0];
            auto j = index[1];
            // Black color
            if ((i+j)%2 == 1)
            {
                u_new_acc[i*dataSize+j] = 0.25*(u_old_acc[i*dataSize+(j+1)] + u_old_acc[i*dataSize+(j-1)] + u_old_acc[(i+1)*dataSize+(j)] + u_old_acc[(i-1)*dataSize+(j)]) - 0.25*(b_vec_acc[i*dataSize+j]);
            }
        });
    });
    q.wait_and_throw();
    std::cout << "Time elapsed (sec): "<< time_elapsed.Elapsed() << std::endl;

    // Printing of results
    // for (size_t i = 0; i < dataSize*dataSize; i++)
    // {
    //     std::cout << u_new[i] << std::endl;
    // }
    


    return 0;
}