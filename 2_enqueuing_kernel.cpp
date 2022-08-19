#include <CL/sycl.hpp>
#include <iostream>

int main(){
    int a = 10, b = 20;

    // Construct a queue
    auto q = sycl::queue();

    // Allocate device memory and copy contents
    auto a_dev = sycl::malloc_device<int>(1, q);
    auto e1 = q.memcpy(a_dev, &a, sizeof(int));

    auto b_dev = sycl::malloc_device<int>(1, q);
    auto e2 = q.memcpy(b_dev, &b, sizeof(int));

    // Kernel Operation
    auto e3 = q.single_task(e1, [=] {*a_dev *= 2;});
    auto e4 = q.single_task(e2, [=] {*b_dev *= 10;});
    q.single_task({e3, e4}, [=] {*a_dev += *b_dev;}).wait();        // Since we are using wait(), we dont return an event (return is void)
    // If we don't use wait(), we can return an event as auto e5 = q.single_task(...). 
    // But then we need to use the e5 event in the next line where it is a a dependency, else there is race condition

    // Copy contents back
    q.memcpy(&a, a_dev, sizeof(int));

    std::cout << "a is: " << a << std::endl;


    return 0;
}