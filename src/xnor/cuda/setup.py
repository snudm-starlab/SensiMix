"""
SensiMix: Sensitivity-Aware 8-bit Index & 1-bit Value Mixed Precision Quantization for BERT Compression
Authors:
- Tairen Piao (piaotairen@snu.ac.kr), Seoul National University
- Ikhyun Cho (ikhyuncho@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
      name='xnor_cuda',
      ext_modules=[
            CUDAExtension('xnor_cuda', [
                  'xnor_cuda.cpp',
                  'xnor_kernel.cu',])],
      cmdclass={
            'build_ext':BuildExtension
      })
