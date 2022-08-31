#include<iostream>
#include<random>
#include<oneapi/dpl/execution>
#include<CL/sycl.hpp>
#include<dpc_common.hpp>

class Complex_Num
{
private:
    float real_part;
    float imaginary_part;
public:
    Complex_Num();
    Complex_Num(float, float);
    ~Complex_Num();
    Complex_Num complex_mult(const Complex_Num&) const;
    float get_real_part();
    float get_imaginary_part();
    void set_real_part();
    void set_imaginary_part();
};

Complex_Num::Complex_Num(){
    real_part = 0.0;
    imaginary_part = 0.0;
}

Complex_Num::Complex_Num(float x, float y):real_part(x), imaginary_part(y){};

Complex_Num::~Complex_Num(){};

// Note the declaration of the method
Complex_Num Complex_Num::complex_mult(const Complex_Num& sec_num) const{
    return Complex_Num(real_part*sec_num.real_part - imaginary_part*sec_num.imaginary_part, real_part*sec_num.imaginary_part + imaginary_part*sec_num.real_part);
}

float Complex_Num::get_real_part(){return real_part;}
float Complex_Num::get_imaginary_part(){return imaginary_part;}

int main(){
    constexpr size_t dataSize = 1024*1024;
    
    sycl::queue q{};
    Complex_Num* vec_1 = sycl::malloc_shared<Complex_Num>(dataSize, q);
    Complex_Num* vec_2 = sycl::malloc_host<Complex_Num>(dataSize, q);
    Complex_Num* vec_out = sycl::malloc_host<Complex_Num>(dataSize, q);

    for (size_t i = 0; i < dataSize; i++)
    {
        Complex_Num cnum_obj1(rand(), rand());
        Complex_Num cnum_obj2(cnum_obj1.get_real_part()*2, cnum_obj1.get_imaginary_part()*3);
        vec_1[i] = cnum_obj1;
        vec_2[i] = cnum_obj2;
    }
    

    auto timeElapsed = dpc_common::TimeInterval();
    sycl::buffer<Complex_Num, 1> vec_1_buff(vec_1, sycl::range<1>(dataSize));
    sycl::buffer<Complex_Num, 1> vec_2_buff(vec_2, sycl::range<1>(dataSize));
    sycl::buffer<Complex_Num, 1> vec_out_buff(vec_out, sycl::range<1>(dataSize));
    q.submit([&](sycl::handler& cgh){
        auto vec_1_acc = vec_1_buff.get_access<sycl::access::mode::read>(cgh);
        auto vec_2_acc = vec_2_buff.get_access<sycl::access::mode::read>(cgh);
        auto vec_out_acc = vec_out_buff.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::range<1>(dataSize), [=](sycl::id<1> idx){
            auto i = idx[0];
            vec_out_acc[i] = vec_1_acc[i].complex_mult(vec_2_acc[i]);

        });
    });
    q.wait_and_throw();
    std::cout << "Time taken (secs) : " << timeElapsed.Elapsed() << std::endl;

    // for (size_t i = 0; i < dataSize; i++)
    // {
    //     std::cout << vec_1[i].get_real_part() << " + i" << vec_1[i].get_imaginary_part() << " * " << vec_2[i].get_real_part() << " + i" << vec_2[i].get_imaginary_part() << " = " << vec_out[i].get_real_part() << " + i" << vec_out[i].get_imaginary_part() << std::endl;
    // }
    


    return 0;
}