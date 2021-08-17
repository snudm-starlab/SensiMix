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
#include <iostream>
#include <torch/types.h>


torch::Tensor encode_rows_cuda(torch::Tensor);
torch::Tensor encode_cols_cuda(torch::Tensor);
torch::Tensor test_gemm_cuda(torch::Tensor, torch::Tensor);


torch::Tensor encode_rows(torch::Tensor input) {
  return encode_rows_cuda(input);
}

torch::Tensor encode_cols(torch::Tensor input) {
  return encode_cols_cuda(input);
}

torch::Tensor test_gemm(torch::Tensor input_a, torch::Tensor intput_b) {
  return test_gemm_cuda(input_a,intput_b);
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_rows",&encode_rows,"encode_rows");
    m.def("encode_cols",&encode_cols,"encode_cols");
    m.def("test_gemm",&test_gemm,"test_gemm");
  }
