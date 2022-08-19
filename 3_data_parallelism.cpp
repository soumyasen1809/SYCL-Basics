#include <CL/sycl.hpp>
#include <iostream>

int main(){
    constexpr size_t dataSize = 1024;
    float a[dataSize], b[dataSize], c[dataSize];
    for (size_t i = 0; i < dataSize; i++)
    {
        a[i] = 1.0*i;
        b[i] = 2.0*i;
        c[i] = 0.0;
    }
    

    // Define a queue
    auto q = sycl::queue();

    // Allocate device memory and copy contents
    auto *a_dev = sycl::malloc_device<float>(dataSize, q);
    auto *b_dev = sycl::malloc_device<float>(dataSize, q);
    auto *c_dev = sycl::malloc_device<float>(dataSize, q);

    auto e1 = q.memcpy(a_dev, a, sizeof(float)*dataSize);
    auto e2 = q.memcpy(b_dev, b, sizeof(float)*dataSize);

    // Kernel operation
    q.parallel_for(sycl::range{dataSize}, {e1, e2}, [=](sycl::id<1> idx){
        auto globalId = idx[0];
        c_dev[globalId] = a_dev[globalId] + b_dev[globalId];
    }).wait();

    // Copy contents back to host
    q.memcpy(c, c_dev, sizeof(float)*dataSize).wait();

    for (const auto& i:c)
    {
        std::cout << i << std::endl;
    }
    

    return 0;
}