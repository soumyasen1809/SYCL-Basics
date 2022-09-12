/*
==========================================
 Title:  1D Convolution
 Author: Sen
 Date:   10 Sep 2022
 Problem: https://www.geeksforgeeks.org/program-to-generate-an-array-having-convolution-of-two-given-arrays/
==========================================
*/

#include<iostream>
#include<vector>
#include<CL/sycl.hpp>
#include<dpc_common.hpp>
#include<oneapi/dpl/algorithm>
#include<oneapi/dpl/execution>

int main(){
    constexpr size_t dataSize = 1024*1024;
    constexpr size_t maskSize = 4;
    constexpr size_t convolSize = dataSize+maskSize-1;
    
    std::vector<int> image_vec(dataSize);
    std::vector<int> mask(maskSize);
    std::vector<int> conv_vec(convolSize, 0);
    std::for_each(oneapi::dpl::execution::par_unseq, image_vec.begin(), image_vec.end(), [&](auto& i){i = (&i-&image_vec[0])+1;});
    std::for_each(oneapi::dpl::execution::par_unseq, mask.begin(), mask.end(), [&](auto& i){i = (&i-&mask[0])+5;});
    
    // std::cout << "Image vec: " << std::endl;
    // for (const auto& i:image_vec){std::cout << i << std::endl;}
    // std::cout << "Mask: " << std::endl;
    // for (const auto& i:mask){std::cout << i << std::endl;}

    sycl::queue q{};
    sycl::buffer<int, 1> image_vec_buff(image_vec.data(), image_vec.size(), sycl::property::buffer::use_host_ptr());
    sycl::buffer<int, 1> mask_buff(mask.data(), mask.size(), sycl::property::buffer::use_host_ptr());
    sycl::buffer<int, 1> conv_vec_buff(conv_vec.data(), conv_vec.size(), sycl::property::buffer::use_host_ptr());

    auto time_elapsed = dpc_common::TimeInterval();
    q.submit([&](sycl::handler& cgh){
        auto image_vec_acc = image_vec_buff.get_access<sycl::access::mode::read>(cgh);
        auto mask_acc = mask_buff.get_access<sycl::access::mode::read>(cgh);
        auto conv_vec_acc = conv_vec_buff.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::range<2>{dataSize, maskSize}, [=](sycl::id<2> index){
            auto i = index[0];
            auto j = index[1];

            conv_vec_acc[i+j] += image_vec_acc[i] * mask_acc[j];
        });
    });
    q.wait_and_throw();

    std::cout << "Time elapsed: " << time_elapsed.Elapsed() << " sec" << std::endl;
    // for (const auto& i:conv_vec){std::cout << i << std::endl;}

    return 0;
}