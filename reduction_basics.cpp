#include<iostream>
#include<vector>
#include<CL/sycl.hpp>

int main(){
    constexpr size_t dataSize = 32;

    sycl::queue q{};
    auto *a_usm = malloc_shared<int>(dataSize, q);    // Using unified shared memory
    for (size_t i = 0; i < dataSize; i++){a_usm[i] = i;}
    
    auto *sum_usm = malloc_shared<int>(1, q);
    *sum_usm = 0;

    sycl::buffer<int, 1> a_buff(a_usm, sycl::range<1>(dataSize));
    sycl::buffer<int, 1> sum_buff(sum_usm, 1);
    
    q.submit([&](sycl::handler& cgh){
        auto a_acc = a_buff.get_access<sycl::access::mode::read>(cgh);
        auto sum_reduction = sycl::reduction(sum_buff, cgh, sycl::plus<>());  // Create a reduction object

        cgh.parallel_for(sycl::range<1>(dataSize), sum_reduction, [=](sycl::id<1> index, auto& temp_sum){
            auto i = index[0];
            temp_sum += a_acc[i];
        });
    }).wait();

    // Alternative: Use OneAPI DPC++ : https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/kernels/reduction.html
    // q.submit([&](auto &h) {
    //   sycl::accessor buf_acc(buf, h, sycl::read_only);
    //   sycl::accessor sum_acc(sum_buf, h, sycl::read_write);
    //   auto sumr =
    //       sycl::ext::oneapi::reduction(sum_acc, sycl::ext::oneapi::plus<>());
    //   h.parallel_for(sycl::nd_range<1>{data_size, 256}, sumr,
    //                  [=](sycl::nd_item<1> item, auto &sumr_arg) {
    //                    int glob_id = item.get_global_id(0);
    //                    sumr_arg += buf_acc[glob_id];
    //                  });
    // });


    // *sum_usm = sum_buff.get_host_access()[0];    // Not required since value stored directly to sum_usm
    std::cout << "Sum is: " << *sum_usm << std::endl;

    sycl::free(a_usm, q);       // USM allocations must be deallocated
    sycl::free(sum_usm, q);
    return 0;
}