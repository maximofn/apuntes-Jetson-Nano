# import torch
# import torchvision
# import cv2
import numpy as np
# import time
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import matplotlib.pyplot as plt
import os
# import skimage.transform as skt

BATCH_SIZE=1

# Export onnx to engine
USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32

tensorrt_file = f"resnet50_engine_pytorch_BS{BATCH_SIZE}_docker.engine"
if not os.path.exists(tensorrt_file):
    if USE_FP16:
        !/usr/src/tensorrt/bin/trtexec --onnx=resnet50_pytorch_BS{BATCH_SIZE}.onnx --saveEngine={tensorrt_file}  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
    else:
        !/usr/src/tensorrt/bin/trtexec --onnx=resnet50_pytorch_BS{BATCH_SIZE}.onnx --saveEngine={tensorrt_file}  --explicitBatch
else:
    print(f"{tensorrt_file} engine already exists")