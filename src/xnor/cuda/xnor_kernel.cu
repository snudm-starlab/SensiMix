/*
SensiMix: Sensitivity-Aware 8-bit Index & 1-bit Value Mixed Precision Quantization for BERT Compression
Authors:
- Tairen Piao (piaotairen@snu.ac.kr), Seoul National University
- Ikhyun Cho (ikhyuncho@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>



namespace{


__device__ uint32_t concatenate(float* array)
{
    unsigned int rvalue=0;
    unsigned int sign;
    
    for (int i = 0; i < 32; i++) {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<<i);
    }
    return rvalue;
}

// __global__ void encode_rows_kernel(float* input, uint32_t* output, int size)
// { 
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i<size) output[i] = concatenate(&input[i*32]);
// }

__global__ void encode_rows_kernel(float* input, uint32_t* output, int size)
{ 
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rvalue=0;
    uint32_t sign;
    float* array = &input[i*32];

    if(i<size) {
        for(int j = 0;j < 32;j++) {
            sign = (array[j]>=0);
            rvalue = rvalue | (sign<<j);
        }
        output[i] = rvalue;
    }
}


__global__ void encode_cols_kernel(float *a, uint32_t *b, int m, int n)
{   

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(j<n) {
        float * array = new float[32];
        for(int i=0; i<m; i+=32) {
            for(int k=0; k<32;k++) {
                array[k] = a[j + n*(i+k)];
            } 
            b[j+n*i/32]=concatenate(array); 
        } 
        delete[] array;
    }
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_gemm_kernel(uint32_t* A, uint32_t* B, float* C, int m, int n, int k) {
    // Block row and column
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    // Thread row and column within Csub
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    const int BLOCK_SIZE = 16;
    // Each thread block computes one sub-matrix Csub of C
    const int c = blockIdx.x * blockDim.x + threadIdx.x; //piao added
    const int r = blockIdx.y * blockDim.y + threadIdx.y; //piao added

    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];
    // Shared memory used to store Asub and Bsub respectively
    __shared__ uint32_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ uint32_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    float Cvalue = 0.0;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results

    for (int i = 0; i < (n-1) / BLOCK_SIZE + 1; ++i) 
    {
        uint32_t* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        uint32_t* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        if ((BLOCK_SIZE*i+col)<n && r<m)
            As[row][col] = Asub[row*n+col];// * ((BLOCK_SIZE*i+col)<n && r<m); //optimized 
        else
            As[row][col] = 0;
        if ((BLOCK_SIZE*i+row)<n && c<k)
            Bs[row][col] = Bsub[row*k+col];// *((BLOCK_SIZE*i+row)<n && c<k); //optimized
        else
            Bs[row][col] = 0;
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; ++j)
            Cvalue += __popc(As[row][j]^Bs[j][col]);

        __syncthreads();
    }


    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) {
        Csub[row*k+col] = -(2*(float)Cvalue-32*n);
        // Csub[row*k+col] = 99;
    }

}




} //namespace







torch::Tensor encode_rows_cuda(torch::Tensor input)
{
    const int m = input.size(0);
    const int n = input.size(1);
    const int l = 1+(n-1)/32;
    const int size = m*l;

    torch::Tensor output = torch::zeros({m,l},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    float* a = (float*)input.data<float>();
    uint32_t* b = (uint32_t*)output.data<float>();

    const int threadsPerBlock = 64;
    const int blocksPerGrid = m * n / 32  + 1;

    encode_rows_kernel<<<blocksPerGrid, threadsPerBlock>>> (a, b, size);

    return output;
}

torch::Tensor encode_cols_cuda(torch::Tensor input)
{
    const int n = input.size(0);
    const int k = input.size(1);
    const int l = 1+(n-1)/32;
    torch::Tensor output = torch::zeros({l,k},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    float* a = (float*)input.data<float>();
    uint32_t* b = (uint32_t*)output.data<float>();
    
    const int threadsPerBlock = 64;
    const int blocksPerGrid = k / threadsPerBlock + 1;
    encode_cols_kernel<<<blocksPerGrid, threadsPerBlock>>> (a, b, n, k);     

    return output;
}

torch::Tensor test_gemm_cuda(torch::Tensor input_a, torch::Tensor input_b)
{
    
    const int m = input_a.size(0);
    const int n = input_a.size(1);
    const int k = input_b.size(1);
    const int l = 1+(n-1)/32;
    const int bin_a_size = m * l;
    torch::Tensor bin_input_a = torch::zeros({m,l},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));


    float* a1 = (float*)input_a.data<float>();
    uint32_t* b1 = (uint32_t*)bin_input_a.data<float>();

    const int threadsPerBlock = 64;
    const int blocksPerGrid = m * n / 32  + 1;

    encode_rows_kernel<<<blocksPerGrid, threadsPerBlock>>> (a1, b1, bin_a_size);

    torch::Tensor output = torch::zeros({m,k},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    uint32_t* b2 = (uint32_t*)input_b.data<float>();
    float* c2 = (float*)output.data<float>();

    const dim3 blockDim(16, 16);
	const dim3 gridDim((k-1) / 16 + 1, (m-1) / 16 + 1);
    xnor_gemm_kernel <<<gridDim, blockDim>>> (b1, b2, c2, m, l, k);

    return output;

}
