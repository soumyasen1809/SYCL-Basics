#include<iostream>
#include<vector>
#include<random>
#include<CL/sycl.hpp>
#include<dpc_common.hpp>
#include<oneapi/dpl/algorithm>
#include<oneapi/dpl/execution>

void matrix_vector_product(sycl::queue& q, const size_t dataSize, std::vector<float> &x, std::vector<float> &A, std::vector<float> &result){
    sycl::buffer<float, 1> x_buff(x.data(), sycl::range<1>(dataSize), sycl::property::buffer::use_host_ptr());
    // sycl::buffer x_buff(x.data(), sycl::range<1>(dataSize)); // Simplified in oneapi
    // https://spec.oneapi.io/versions/1.0-rev-3/architecture.html
    sycl::buffer<float, 1> A_buff(A.data(), sycl::range<1>{dataSize*dataSize}, sycl::property::buffer::use_host_ptr());
    sycl::buffer<float, 1> result_buff(result.data(), sycl::range<1>(dataSize), sycl::property::buffer::use_host_ptr());

    q.submit([&](sycl::handler &cgh){
        auto x_acc = x_buff.get_access<sycl::access::mode::read_write>(cgh);
        auto A_acc = A_buff.get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buff.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::range<1>{dataSize}, [=](sycl::id<1> index){
            auto i = index[0];
            // auto j = index[1];
            auto temp = 0.0;
            for (size_t j = 0; j < dataSize; j++)
            {
                temp += A_acc[i*dataSize+j] * x_acc[j];
            }
            
            result_acc[i] = temp;
        });
    });
    q.wait();
}

void vector_vector_product(sycl::queue& q, const size_t dataSize, std::vector<float> &a, std::vector<float> &b, float &result){
    sycl::buffer<float, 1> a_buff(a.data(), sycl::range<1>(dataSize), sycl::property::buffer::use_host_ptr());
    sycl::buffer<float, 1> b_buff(b.data(), sycl::range<1>(dataSize), sycl::property::buffer::use_host_ptr());
    sycl::buffer<float, 1> result_buff(&result, 1, sycl::property::buffer::use_host_ptr());

    q.submit([&](sycl::handler &cgh){
        auto a_acc = a_buff.get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b_buff.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<1>{dataSize}, sycl::reduction(result_buff, cgh, std::plus<float>(), sycl::property::reduction::initialize_to_identity()), 
            [=](sycl::id<1> index, auto& sum){
                auto i = index[0];
                sum += a_acc[i]*b_acc[i];
            });

    });
    q.wait();
}

void steepest_descent(sycl::queue& q, const size_t dataSize, std::vector<float> &x, std::vector<float> &A, std::vector<float> &b){

    // Steepest descent agorithm here
    std::vector<float> r(dataSize, 0.0);
    std::vector<float> Ax(dataSize, 0.0);
    std::vector<float> Ar(dataSize, 0.0);

    matrix_vector_product(q, dataSize, x, A, Ax);
    for (size_t i = 0; i < r.size(); i++)
    {
        r[i] = b[i] - Ax[i];
    }

    float temp1, temp2;
    matrix_vector_product(q, dataSize, r, A, Ar);
    vector_vector_product(q, dataSize, r, r, temp1);
    vector_vector_product(q, dataSize, r, Ar, temp2);
    float alpha = temp1/temp2;

    for (size_t i = 0; i < x.size(); i++)
    {
        x[i] += alpha*r[i];
    }

}


int main(){
    // Algo: Gradient descent: https://en.wikipedia.org/wiki/Gradient_descent

    constexpr size_t dataSize = 1024*8;
    std::vector<float> x(dataSize, 1.0);
    std::vector<float> A(dataSize*dataSize);
    std::for_each(oneapi::dpl::execution::par_unseq, A.begin(), A.end(), [](auto& i){i = rand()%10;});
    std::vector<float> b(dataSize);
    std::for_each(oneapi::dpl::execution::par_unseq, b.begin(), b.end(), [](auto& i){i = rand()%10;});

    sycl::queue q{};
    
    float tol = 1.0;
    std::vector<float> x_old(x.size(), 0.0);
    std::vector<float> diff(x.size(), 0.0);
    size_t counter = 0;

    auto time_elapsed = dpc_common::TimeInterval();
    while (tol > 1e-6)
    {
        // Gradient descent method
        steepest_descent(q, dataSize, x, A, b);
        for (size_t i = 0; i < x.size(); i++)
        {
            diff[i] = x[i] - x_old[i];
        }
        tol = *std::max_element(oneapi::dpl::execution::par_unseq, diff.begin(), diff.end());
        std::copy(oneapi::dpl::execution::par_unseq, x.begin(), x.end(), x_old.begin());

        counter++;
    }
    q.wait_and_throw();
    std::cout << "Time elapsed is: " << time_elapsed.Elapsed() << " secs with number of iterations: " << counter << std::endl;

    // Print solution vector
    // for (const auto& i:x)
    // {
    //     std::cout << i << std::endl;
    // }
    
    return 0;
}