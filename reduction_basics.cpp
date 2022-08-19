#include<iostream>
#include<vector>
#include<CL/sycl.hpp>

int main(){
    constexpr size_t dataSize = 16;

    sycl::queue q{};
    auto *a_usm = malloc_shared<int>(dataSize, q);    // Using unified shared memory
    auto *sum_usm = malloc_shared<int>(1, q);

    sycl::buffer<int, 1> a_buff(a_usm, sycl::range<1>(dataSize));
    sycl::buffer<int, 1> sum_buff(sum_usm, 1);
    
    q.submit([&](sycl::handler& cgh){
        auto a_acc = a_buff.get_access<sycl::access::mode::read>(cgh);
        auto sum_reduction = sycl::reduction(sum_buff, cgh, sycl::plus<>());  // Create a reduction object

        cgh.parallel_for(sycl::range<1>(dataSize), sum_reduction, [=](sycl::id<1> index, auto& sum){
            sum += a_acc[index];
        });
    }).wait();

    *sum_usm = sum_buff.get_host_access()[0];
    std::cout << "Sum is: " << *sum_usm << std::endl;

    sycl::free(a_usm, q);       // USM allocations must be deallocated
    sycl::free(sum_usm, q);
    return 0;
}