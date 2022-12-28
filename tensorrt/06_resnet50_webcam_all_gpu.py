import torch
import torchvision
import cv2
import numpy as np
import time
from thread import InputThread
from udp_socket import udp_socket
from video import video

# Create input thread
input_thread = InputThread()
input_thread.start()

# Create UDP socket
sock = udp_socket('localhost', 8554, send=True)

# Open webcam
video = video(resize=False, width=1920, height=1080, fps=30, name="frame", display=False)
video.open(device=0)

# Download model
print("Creating model...")
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

# dict with ImageNet labels
with open('imagenet_labels.txt') as f:
    labels = eval(f.read())

T0 = time.time()
t_camera = 0
t_preprocess = 0
t_inference = 0
t_postprocess = 0
t_bucle = 0
FPS = 0
# iteracctions = -1
# t_read_frame = []
# t_preprocess = []
# t_inference = []
# t_postprocess = []
# FPS_list = []

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
    cv2.putText(frame, f"Todo en GPU:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30
    cv2.putText(frame, f"FPS: {FPS:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30
    cv2.putText(frame, f"Image shape: {img.shape}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30
    cv2.putText(frame, f"t camera: {t_camera*1000:.2f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30
    cv2.putText(frame, f"t preprocess: {t_preprocess*1000:.2f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30
    cv2.putText(frame, f"t model to gpu: {t_model_gpu*1000:.2f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30
    cv2.putText(frame, f"t image to gpu: {t_img_gpu*1000:.2f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30
    cv2.putText(frame, f"t inference: {t_inference*1000:.2f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30
    cv2.putText(frame, f"t postprocess: {t_postprocess*1000:.2f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30
    cv2.putText(frame, f"Predicted: {idx}-{labels[idx]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); y += 30

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