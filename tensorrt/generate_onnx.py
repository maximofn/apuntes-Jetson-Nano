import torchvision
import torch
from torch.onnx import OperatorExportTypes

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

# Export to ONNX
BATCH_SIZE=1
onnx_file = f"resnet50_pytorch_BS{BATCH_SIZE}.onnx"
dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)
torch.onnx.export(model, dummy_input, onnx_file, verbose=False, input_names=["input"], output_names=["output"], operator_export_type=OperatorExportTypes.ONNX_EXPLICIT_BATCH)
print("Done exporting to ONNX")

# Export to TensorRT
# !/usr/src/tensorrt/bin/trtexec --onnx=../../tensorrt/resnet50_pytorch_BS1.onnx --saveEngine=resnet50_engine_pytorch_BS1.engine --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16