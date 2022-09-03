// https://www.it.uu.se/education/phd_studies/phd_courses/pasc/lecture-1
#include<iostream>
#include<vector>
#include<random>
#include<CL/sycl.hpp>
#include<oneapi/dpl/algorithm>
#include<oneapi/dpl/execution>
#include<dpc_common.hpp>

int main(){
    constexpr size_t dataSize = 1024*16;
    std::vector<double> vec(dataSize, 0.0);
    std::vector<std::vector<double>> mat(dataSize, std::vector<double>(dataSize, 0.0));
    std::vector<double> result(dataSize, 0.0);

    std::for_each(oneapi::dpl::execution::par_unseq, vec.begin(), vec.end(), [&](double& i){i = &i-&vec[0];}); // https://www.techiedelight.com/find-index-of-each-value-in-a-range-based-for-loop-in-cpp/
    // Creating a sparse matrix
    size_t nnz = 0;
    for (size_t i = 0; i < dataSize; i++)
    {
        for (size_t j = 0; j < dataSize; j++)
        {
            bool toss = rand()%2;
            if (toss == 0)
            {
                mat[i][j] = (i+j)*10;
                nnz++;
            }
        }
        
    }
    std::vector<int> row_offset;
    std::vector<int> column_index;
    std::vector<double> values;

    bool same_row_flag = false;
    int col_position = 0;
    for (size_t i = 0; i < dataSize; i++)
    {
        for (size_t j = 0; j < dataSize; j++)
        {
            if (mat[i][j] != 0)
            {
                values.push_back(mat[i][j]);
                column_index.push_back(j);

                if (same_row_flag == false)
                {
                    row_offset.push_back(col_position);
                }
                col_position++;
                same_row_flag = true;
            }
        }
        same_row_flag = false; 
    }
    row_offset.push_back(nnz);

    
    auto time_elapsed = dpc_common::TimeInterval();
    sycl::queue q{};
    
    // Store in CSR format
    sycl::buffer<double, 1> vec_buff(vec.data(), vec.size(), sycl::property::buffer::use_host_ptr());
    sycl::buffer<double, 1> values_buff(values.data(), values.size(), sycl::property::buffer::use_host_ptr());
    sycl::buffer<int, 1> row_offset_buff(row_offset.data(), row_offset.size(), sycl::property::buffer::use_host_ptr());
    sycl::buffer<int, 1> column_index_buff(column_index.data(), column_index.size(), sycl::property::buffer::use_host_ptr());
    sycl::buffer<double, 1> result_buff(result.data(), result.size(), sycl::property::buffer::use_host_ptr());
    // result_buff.set_final_data(result);

    // Algo:
    // for (int i=0; i<n; ++i) {
    // y[i] = 0.0;
    // for (int j=row_off[i]; j<row_off[i+1]; ++j)
    // y[i] += val[j]*x[col[j]];
    // }

    q.submit([&](sycl::handler& cgh){
        auto vec_acc = vec_buff.get_access<sycl::access::mode::read>(cgh);
        auto values_acc = values_buff.get_access<sycl::access::mode::read>(cgh);
        auto row_offset_acc = row_offset_buff.get_access<sycl::access::mode::read>(cgh);
        auto column_index_acc = column_index_buff.get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::range<1>(dataSize), [=](sycl::id<1> index){
            auto i = index[0];
            for (int j=row_offset_acc[i]; j<row_offset_acc[i+1]; j++){
                result_acc[i] += values_acc[j] * vec_acc[column_index_acc[j]];
            }

        });
    });
    q.wait_and_throw();
    std::cout << "Time taken (sec) : " << time_elapsed.Elapsed() << std::endl;

    // Printing results for debugging
    // std::cout << "-- Row-offset vector --" << std::endl;
    // for (const auto& i:row_offset)
    // {
    //     std::cout << i << std::endl;
    // }
    // std::cout << "------------" << std::endl;
    // std::cout << "-- Column index vector --" << std::endl;
    // for (const auto& i:column_index)
    // {
    //     std::cout << i << std::endl;
    // }
    // std::cout << "------------" << std::endl;
    // std::cout << "-- Sparse matrix (A) --" << std::endl;
    // for (const auto& i:mat)
    // {
    //     for (const auto& j:i)
    //     {
    //         std::cout << j << "  ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "------------" << std::endl;
    // std::cout << "-- Vector (b) --" << std::endl;
    // for (const auto& i:vec)
    // {
    //     std::cout << i << std::endl;
    // }
    // std::cout << "------------" << std::endl;
    // std::cout << "-- Result (Ax*b) --" << std::endl;
    // for (const auto& i:result)
    // {
    //     std::cout << i << std::endl;
    // }
    // std::cout << "------------" << std::endl;

    return 0;
}