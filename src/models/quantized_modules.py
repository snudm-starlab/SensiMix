"""
SensiMix: Sensitivity-Aware 8-bit Index & 1-bit Value Mixed Precision Quantization for BERT Compression
Authors:
- Tairen Piao (piaotairen@snu.ac.kr), Seoul National University
- Ikhyun Cho (ikhyuncho@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import torch
import torch.nn as nn
import math
import xnor_cuda


# Binarize function: binarize input tensors
def Binarize(tensor):
    binarized = torch.where(tensor>0, torch.ones_like(tensor,dtype=torch.float32, device='cuda'), torch.full((tensor.shape),-1, dtype=torch.float32, device='cuda'))
    return binarized

def xnor_linear(input, weight,bias=True):

    weight_col = Binarize(weight.t())
    bin_weight = xnor_cuda.encode_cols(weight_col)
 
    output1 = input.matmul(weight.t())
    output2 = xnor_cuda.test_gemm(input,bin_weight)
    print(torch.equal(output1, output2))

    if bias is not None:
        output1 += bias
    ret = output1

    return ret

def xnor_linear_inference(input, weight, bias=True):
    output = xnor_cuda.test_gemm(input, weight)

    if bias is not None:
        output += bias
    ret = output

    return ret



class BinarizeLinear_inference(nn.Module):

    def __init__(self, in_features, out_features, bias = True):
        super(BinarizeLinear_inference, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantized_weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # Binarize the input while conver FP32 to FP16
        input.data = Binarize(input.data)
        # self.quantized_weight.data = Binarize(self.quantized_weight.data)
        out = xnor_linear_inference(input, self.quantized_weight, self.bias)
        # out = nn.functional.linear(input, self.quantized_weight, self.bias)
        return out


# class BinarizeLinear(nn.Linear):

#     def __init__(self, *kargs, **kwargs):
#         super(BinarizeLinear, self).__init__(*kargs, **kwargs)

#     def forward(self, input):

#         input.data = Binarize(input.data)
#         if not hasattr(self.weight,'org'):
#             self.weight.org=self.weight.data.clone()
#            # self.weight.org.requires_grad = True

#         if self.weight.data.dtype == torch.float:
#             self.weight.data = Binarize(self.weight.org)
#             # out = xnor_linear(input, self.weight, self.bias)
#             out = nn.functional.linear(input, self.weight, self.bias)

#         return out



def quantization(input, bits):
    quantized_max = 2**(bits-1)-1 # e.g., 127 when appying 8-bit index quantization
    quantized_min = -(2**(bits-1)) # e.g., -128 when applying 8-bit index quantization

    pmax = input.max() # pmax: maximun weight or activation
    pmin = input.min() # pmin: minimum weight or activation
    
    int_scale = quantized_max - quantized_min
    fp_scale = pmax - pmin

    quantized = torch.round((input - pmin)*(int_scale / fp_scale)) + quantized_min

    dequantized = (quantized - quantized_min)*(fp_scale / int_scale) + pmin

    return dequantized

class q_Linear(nn.Linear):
    """
    q_Linear layer: provide the 8-bit index quantization
    """

    def __init__(self, *kargs, **kwargs):
        super(q_Linear, self).__init__(*kargs, **kwargs)
        self.q_min = torch.nn.Parameter(torch.Tensor(1), requires_grad = False)
        self.q_max = torch.nn.Parameter(torch.Tensor(1), requires_grad = False)

    def forward(self, input):
        a = float(torch.min(self.weight.data))
        b = float(torch.max(self.weight.data))
        self.q_min.data = torch.Tensor([a]).to(device = 'cuda:0')
        self.q_max.data = torch.Tensor([b]).to(device = 'cuda:0')

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data = quantization(self.weight.data,8)

        if input.data.dtype == torch.half:
            input.data = input.data.half()
            self.weight.data = self.weight.data.to(dtype=torch.half)
            self.bias.data = self.bias.data.to(dtype=torch.half)

        out = nn.functional.linear(input, self.weight, self.bias)

        return out


class mix_Linear(nn.Module):

    def __init__(self, in_features, out_features, bias = True):
        super(mix_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.quantized_weight = nn.Parameter(torch.Tensor(out_features,in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, bit_1_quantize = False, layer_number = 0, bit_32_quantize = False):

        # if layers 4,5,6 in SensiMix (3+3), binarize intermediate and output layers' input
        if layer_number > 2:
            if input.dtype == torch.half:
                input.data = Binarize(input.data).half()
            else:
                input.data = Binarize(input.data)

        # (bit_1_quantize, bit_32_quantize) can be one of (False, False) / (False, True) / (True, False)
        # bit_32_quantize is for doing the full-precision learning (Prioritized training)
        if bit_32_quantize == True:
            if input.data.dtype == torch.half:
                out = nn.functional.linear(input, self.weight.half(), self.bias)
            else:
                out = nn.functional.linear(input, self.weight, self.bias)


        # bit_1_quantize: for 1-bit quantization.
        elif bit_1_quantize == True:
            input.data = Binarize(input.data)
            # if not hasattr(self.weight, 'org'):
            #     self.weight.org = self.weight.data.clone()
            with torch.no_grad():
                if input.data.dtype == torch.float:
                    self.quantized_weight.data = Binarize(self.weight)
                else:
                    input.data = input.data.half()
                    self.quantized_weight.data = Binarize(self.weight)
                    self.quantized_weight.data = self.quantized_weight.data.half()

            out = nn.functional.linear(input, self.quantized_weight, self.bias)

            # if self.input.data.dtype == torch.float:
            #     out = nn.functional.linear(input, self.quantized_weight, self.bias)
            # else:
            #     input.data = input.data.half()
            #     self.quantized_weight.data = self.quantized_weight.data.to(dtype=torch.half)
            #     self.bias.data = self.bias.data.half()
            #     out = nn.functional.linear(input, self.quantized_weight, self.bias)

            # elif self.quantized_weight.data.dtype == torch.half:
            #     input.data = input.data.half()
            #     self.quantized_weight.data = self.quantized_weight.data.to(dtype=torch.half)
            #     self.bias.data = self.bias.data.half()
            #     out = nn.functional.linear(input, self.quantized_weight, self.bias)
        else:
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.data = quantization(self.weight.org,8)
            if input.dtype == torch.half:
                self.weight.data = self.weight.data.to(dtype=torch.half)
            out = nn.functional.linear(input, self.weight, self.bias)
        return out



# 8-bit index quantized embedding class
class qEmbedding(nn.Embedding):

    def __init__(self, *kargs, **kwargs):
        super(qEmbedding, self).__init__(*kargs, **kwargs)

        # q_min and q_max store the minimum and the values maximum full-precision weight
        self.q_min = torch.nn.Parameter(torch.Tensor(1), requires_grad = False)
        self.q_max = torch.nn.Parameter(torch.Tensor(1), requires_grad = False)

    def forward(self, input):
        a = float(torch.min(self.weight.data))
        b = float(torch.max(self.weight.data))
        self.q_min.data = torch.Tensor([a]).to(device = 'cuda:0')
        self.q_max.data = torch.Tensor([b]).to(device = 'cuda:0')

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        # self.weight.data = torch.zeros_like(self.weight.data)
        self.weight.data=quantization(self.weight.data,8)

        return torch.nn.functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
