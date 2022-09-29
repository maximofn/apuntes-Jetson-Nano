#import shutil

#print("Copy resnet18.caffemodel_b16_gpu0_fp16.engine"
# shutil.copy2('/opt/nvidia/deepstream/deepstream-6.0/samples/models/Secondary_CarMake/resnet18.caffemodel_b16_gpu0_fp16.engine', '/dli/task/deepstream_apps/resnet18.caffemodel_b16_gpu0_fp16.engine')
#shutil.copy2('/dli/task/deepstream_apps/sample_720p.h264', '/dli/task/deepstream_apps/sample_720p2.h264')


import subprocess


# resnet18.caffemodel_b16_gpu0_fp16.engine
bashCommand = "cp /dli/task/deepstream_apps/Secondary_CarMake/resnet18.caffemodel_b16_gpu0_fp16.engine /opt/nvidia/deepstream/deepstream-6.0/samples/models/Secondary_CarMake/resnet18.caffemodel_b16_gpu0_fp16.engine"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# resnet18.caffemodel_b16_gpu0_fp16.engine
bashCommand = "cp /dli/task/deepstream_apps/Secondary_CarColor/resnet18.caffemodel_b16_gpu0_fp16.engine /opt/nvidia/deepstream/deepstream-6.0/samples/models/Secondary_CarColor/resnet18.caffemodel_b16_gpu0_fp16.engine"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# resnet10.caffemodel_b1_gpu0_fp16.engine
bashCommand = "cp /dli/task/deepstream_apps/Primary_Detector/resnet10.caffemodel_b1_gpu0_fp16.engine /opt/nvidia/deepstream/deepstream-6.0/samples/models/Primary_Detector/resnet10.caffemodel_b1_gpu0_fp16.engine"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()