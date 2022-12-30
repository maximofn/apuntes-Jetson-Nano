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

# Create input thread
input_thread = InputThread()
input_thread.start()

# Create UDP socket
sock = udp_socket('localhost', 8554, send=True)

# Open webcam
video = video(resize=False, width=1920, height=1080, fps=30, name="frame", display=False)
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
engine_model = "resnet50_pytorch_BS1.engine"
with open(engine_model, "rb") as f:
    serialized_engine = f.read()

# Create runtime
runtime = trt.Runtime(logger)

# Deserialize engine
engine = runtime.deserialize_cuda_engine(serialized_engine)

# Create context
context = engine.create_execution_context()

# need to set input and output precisions to FP16 to fully enable it
BATCH_SIZE=1
USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 

# allocate device memory
d_input = cuda.mem_alloc(1 * np.array(input_batch, dtype=np.float16).nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

def preprocess_image(img):
    norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    result = norm(torch.from_numpy(img).transpose(0,2).transpose(1,2))
    result = np.array(result, dtype=np.float16)
    return result

while True:
    t0 = time.time()
    t_start = time.time()
    ret, frame = video.read()
    t_camera = time.time() - t0
    if not ret:
        continue

    # Send image to GPU
    t0 = time.time()
    img = frame.copy()
    img = torch.from_numpy(img)
    img = img.cuda()
    t_img_gpu = time.time() - t0

    # Preprocess image
    t0 = time.time()
    img = img[:, :, [2, 1, 0]]  # Convert to RGB
    img = img.permute(2, 0, 1)  # Convert to CHW
    img = (img/255).float()     # Normalize
    img = img.unsqueeze(0)      # Add batch dimension
    t_preprocess = time.time() - t0

    # Send model to GPU
    t0 = time.time()
    model = model.cuda()
    t_model_gpu = time.time() - t0

    # Inference
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        end = time.time()
    t_inference = time.time() - t0

    # Postprocess
    t0 = time.time()
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    outputs = outputs.squeeze(0)
    outputs = outputs.tolist()
    idx = outputs.index(max(outputs))
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
    cv2.putText(frame, f"t preprocess: {t_preprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t model to gpu: {t_model_gpu*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t image to gpu: {t_img_gpu*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
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