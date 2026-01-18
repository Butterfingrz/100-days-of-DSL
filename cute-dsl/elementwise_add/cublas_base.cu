// https://mp.weixin.qq.com/s/yiCP4sCU_kb_jM4UHgQ-vg
//pixi run nvcc cute-dsl/elementwise_add/cublas_base.cu  -o cute-dsl/elementwise_add/cublas_base -lcublas -arch=sm_90 -O2

// --------------------------------------------------------
// Average execution time: 0.0949907 ms
// Performance (GFLOPS): 353.239 GFLOPS
// Effective Memory Bandwidth: 4238.87 GB/s
// --------------------------------------------------------
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA 和 CUBLAS API 调用的错误检查宏
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


int main() {
    // 1. 定义矩阵维度
    int M = 8192;
    int N = 4096;
    long long num_elements = (long long)M * N;

    std::cout << "Testing cublasSgeam (element-wise add) with matrix size" << M << "x" << N << std::endl;

    // C = alpha * A + beta * B. 对于简单的加法, alpha=1.0, beta=1.0
    float alpha = 1.0f;
    float beta = 1.0f;

    // 2. 在主机端分配和初始化数据
    std::vector<float> h_A(num_elements);
    std::vector<float> h_B(num_elements);
    std::vector<float> h_C_gpu(num_elements);
    std::vector<float> h_C_cpu(num_elements);

    // 使用随机数填充矩阵
    for (long long i = 0 ; i < num_elements; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 3. 在设备端分配内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, num_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, num_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, num_elements * sizeof(float)));

    // 4. 创建 CUBLAS 句柄
    cublasHandle_t handler;
    CHECK_CUBLAS(cublasCreate(&handler));

    // 5. 将数据从主机拷贝到设备
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    // 6. 性能测试
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    CHECK_CUBLAS(cublasSgeam(handler,
                CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置 A 和 B
                M, N,
                &alpha,
                d_A, M,                    // A 和它的 leading dimension
                &beta,
                d_B, M,                    // B 和它的 leading dimension
                d_C, M));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 正式计时
    int iteration = 100;
    float total_time = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iteration; ++i) {
        CHECK_CUBLAS(cublasSgeam(handler,
                CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置 A 和 B
                M, N,
                &alpha,
                d_A, M,                    // A 和它的 leading dimension
                &beta,
                d_B, M,                    // B 和它的 leading dimension
                d_C, M));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));  // 等待所有 GPU 操作完成
    CHECK_CUDA(cudaEventElapsedTime(&total_time, start, stop));

    // 7. 计算并打印性能结果
    float avg_time_ms = total_time / iteration;
    float avg_time_s = avg_time_ms / 1000.f;

    // 计算 GFLOPS
    // 每个元素有 1 次浮点加法
    double flops = (double)num_elements;
    double gflops = (flops / 1e9) / avg_time_s;

    // 计算有效内存带宽
    // 每次操作需要: 读取 A (M*N floats), 读取 B (M*N floats), 写入 C (M*N floats)
    // 总共传输 3 * M * N * sizeof(float) 字节
    double bytes_transferred = 3.0 * num_elements * sizeof(float);
    double bandwidth_gb_s = (bytes_transferred / 1e9) / avg_time_s;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Average execution time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance (GFLOPS): " << gflops << " GFLOPS" << std::endl;
    std::cout << "Effective Memory Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    // 将 GPU 计算结果拷贝回主机
    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // 9. 清理资源
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handler));

    return 0;
}
