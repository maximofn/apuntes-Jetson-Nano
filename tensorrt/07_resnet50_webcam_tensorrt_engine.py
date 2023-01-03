import torch
import torchvision
import cv2
import numpy as np
import time
from thread import InputThread
from udp_socket import udp_socket
from video import video
import pycuda.driver as cuda
import pycuda.autoinit
import skimage.transform as skt
from torch.onnx import OperatorExportTypes
import tensorrt as trt

# Create input thread
input_thread = InputThread()
input_thread.start()

# Create UDP socket
sock = udp_socket('localhost', 8554, send=True)

# Open webcam
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 30
video = video(resize=False, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=CAPTURE_FPS, name="frame", display=False)
video.open(device=0)

# dict with ImageNet labels
with open('imagenet_labels.txt') as f:
    labels = eval(f.read())

# Configuration of text on the screen
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (0, 0, 255)#(10,10,10)
lineThickness= 1
lineType = cv2.LINE_AA

# Time variables
T0 = time.time()
t_camera = 0
t_preprocess = 0
t_inference = 0
t_postprocess = 0
t_bucle = 0
FPS = 0

# Media variables
iteracctions = -1
t_read_frame_list = []
t_preprocess_list = []
t_inference_list = []
t_postprocess_list = []
t_bucle_list = []
FPS_list = []

# Read engine model
f = open(f"resnet50_pytorch_BS1.engine", "rb")

# Create runtime
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

# Deserialize engine
engine = runtime.deserialize_cuda_engine(f.read())

# Create context
context = engine.create_execution_context()

# Resize image
BATCH_SIZE=1
ret, frame = video.read()
print(frame[0,0,:])
# img = torch.from_numpy(frame)[:,:,[2,1,0]].numpy() # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = torch.flip(torch.from_numpy(frame), dims=[2]).numpy() # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(img[0,0,:], frame[0,0,:])
img = (img/255).astype(np.float64)
# img = skt.resize(img, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
input_batch = np.array(np.repeat(img, BATCH_SIZE, axis=0), dtype=np.float32)

def preprocess(frame):
    print(f"\nframe.shape: {frame.shape}, frame.dtype: {frame.dtype}, frame.min(): {frame.min()}, frame.max(): {frame.max()}, frame type: {type(frame)}, frame[0,0,:]: {frame[0,0,:]}")
    img = torch.flip(torch.from_numpy(frame), dims=[2]).numpy() # Convert BGR to RGB
    print(f"BGR to RGB img.shape: {img.shape}, img.dtype: {img.dtype}, img.min(): {img.min()}, img.max(): {img.max()}, img type: {type(img)}, img[0,0,:]: {img[0,0,:]}")
    img = (img/255).astype(np.float64) # Normalize
    print(f"Normalize img.shape: {img.shape}, img.dtype: {img.dtype}, img.min(): {img.min()}, img.max(): {img.max()}, img type: {type(img)}, img[0,0,:]: {img[0,0,:]}")
    img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    print(f"Expand dims img.shape: {img.shape}, img.dtype: {img.dtype}, img.min(): {img.min()}, img.max(): {img.max()}, img type: {type(img)}, img[0,0,0,:]: {img[0,0,0,:]}")
    procesed_image = np.array([preprocess_image(image) for image in img])
    print(f"Preprocess image procesed_image.shape: {procesed_image.shape}, procesed_image.dtype: {procesed_image.dtype}, procesed_image.min(): {procesed_image.min()}, procesed_image.max(): {procesed_image.max()}, procesed_image type: {type(procesed_image)}, procesed_image[0,0,0,:]: {procesed_image[0,0,0,:]}")
    norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    nomalized_image = np.array(norm(torch.from_numpy(img).transpose(1,3).transpose(2,3)), dtype=np.float16)
    print(f"Normalize image nomalized_image.shape: {nomalized_image.shape}, nomalized_image.dtype: {nomalized_image.dtype}, nomalized_image.min(): {nomalized_image.min()}, nomalized_image.max(): {nomalized_image.max()}, nomalized_image type: {type(nomalized_image)}, nomalized_image[0,0,0,:]: {nomalized_image[0,0,0,:]}")
    return nomalized_image

def preprocess_image(img):
    norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    result = norm(torch.from_numpy(img).transpose(0,2).transpose(1,2))
    result = np.array(result, dtype=np.float16)
    return result

# need to set input and output precisions to FP16 to fully enable it
USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 

# allocate device memory
d_input = cuda.mem_alloc(1 * np.array(input_batch, dtype=np.float16).nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    
    return output


while True:
    t0 = time.time()
    t_start = time.time()
    ret, frame = video.read()
    t_camera = time.time() - t0
    if not ret:
        continue

    # Preprocess image
    t0 = time.time()
    preprocessed_image = preprocess(frame)
    t_preprocess = time.time() - t0

    # Inference
    t0 = time.time()
    outputs = predict(preprocessed_image)
    t_inference = time.time() - t0

    # Postprocess
    t0 = time.time()
    idx = outputs[0].argmax()
    t_postprocess = time.time() - t0

    # Bucle time
    t_bucle = time.time() - t_start

    # FPS
    FPS = 1 / t_bucle

    # Put text
    y = 30
    cv2.putText(frame, f"Todo en GPU:", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"FPS: {FPS:.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"Image shape: {img.shape}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t camera: {t_camera*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t preprocess: {t_preprocess*1000:.2f} ms, preprocessed_image type {type(preprocessed_image)}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    # cv2.putText(frame, f"t model to gpu: {t_model_gpu*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    # cv2.putText(frame, f"t image to gpu: {t_img_gpu*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t inference: {t_inference*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t postprocess: {t_postprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t bucle: {t_bucle*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"Predicted: {idx}-{labels[idx]}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30

    # Media variables
    iteracctions += 1
    if iteracctions >= 0:
        t_read_frame_list.append(t_camera)
        t_preprocess_list.append(t_preprocess)
        t_inference_list.append(t_inference)
        t_postprocess_list.append(t_postprocess)
        t_bucle_list.append(t_bucle)
        FPS_list.append(FPS)
        cv2.putText(frame, f"Media: {iteracctions} iteracctions, t read frame {np.mean(t_read_frame_list)*1000:.2f} ms, t preprocess {np.mean(t_preprocess_list)*1000:.2f} \
ms, t inference {np.mean(t_inference_list)*1000:.2f} ms, t postprocess {np.mean(t_postprocess_list)*1000:.2f} ms, t bucle {np.mean(t_bucle_list)*1000:.2f} ms, FPS {np.mean(FPS_list):.2f}",
(10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30

    # Mandamos el frame por el socket
    success, encoded_frame = video.encode_frame(frame)
    if success:
        message = encoded_frame.tobytes(order='C')
        sock.send(message)

    # If user press type 'q' into the console in non blocking mode, exit
    if input_thread.get_data() is not None and input_thread.get_data().strip() == 'q':
        print("Se ha ha parado por el usuario")
        input_thread.clear_data()
        break


# Cerramos el socket y la cámara
sock.close()
video.close()