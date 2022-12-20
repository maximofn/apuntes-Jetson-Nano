import torch
import torchvision
import cv2
import numpy as np
import time
import socket
import struct

def send_message(message, chunk_size=1024):
    # Fragment message into chunks
    N = 10
    chunk_size = int(len(message)/N)
    chunks = [message[i:i+chunk_size] for i in range(0, len(message), chunk_size)]
    # Send chunks
    for chunk in chunks:
        udp_socket.sendto(chunk, target_addr)



# Download model
print("Creating model...")
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

# dict with ImageNet labels
with open('imagenet_labels.txt') as f:
    labels = eval(f.read())

# Open webcam and start inference
print("Starting webcam...")
cap = cv2.VideoCapture(0)
# Full HD
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initialize UDP socket
print("Initializing UDP socket...")
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
target_addr = ('localhost', 8554)

# Set maximum packet size to be sent through the socket
N = 100000
size = N*1024
size_bytes = struct.pack("I", size)
udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, size_bytes)

T0 = time.time()

iteracctions = -1
t_read_frame = []
t_preprocess = []
t_inference = []
t_postprocess = []
FPS_list = []

while True:
    t0 = time.time()
    t_start = time.time()
    ret, frame = cap.read()
    t = time.time() - t0
    if iteracctions >= 0: t_read_frame.append(t)
    print(f"\nTime to read frame: {t*1000:.2f} ms, frame shape: {frame.shape}")
    if not ret:
        continue

    # Preprocess image
    t0 = time.time()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    t = time.time() - t0
    if iteracctions >= 0: t_preprocess.append(t)
    print(f"Time to preprocess image: {t*1000:.2f} ms")

    # Inference
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        start = time.time()
        outputs = model(img)
        end = time.time()
        cv2.putText(frame, f"Inference time: {((end - start)*1000):.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    t = time.time() - t0
    if iteracctions >= 0: t_inference.append(t)
    print(f"Time to inference: {t*1000:.2f} ms")

    # Postprocess
    t0 = time.time()
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    outputs = outputs.squeeze(0)
    outputs = outputs.tolist()
    idx = outputs.index(max(outputs))
    cv2.putText(frame, f"Predicted: {idx}-{labels[idx]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    t = time.time() - t0
    if iteracctions >= 0: t_postprocess.append(t)
    print(f"Time to postprocess: {t*1000:.2f} ms, predicted: {idx}-{labels[idx]}")

    # FPS
    t = time.time() - t_start
    cv2.putText(frame, f"FPS: {1/t:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(f"FPS: {1/t:.2f}")
    if iteracctions >= 0: FPS_list.append(1/t)

    # Image shape
    cv2.putText(frame, f"Image shape: {img.shape}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(f"Image shape: {img.shape}")

    # Display
    # cv2.imshow("frame", frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break

    # Resize image
    #frame = torchvision.transforms.Resize((224, 224))(frame)
    frame = cv2.resize(frame, (224, 224))

    # Get frame size as bytes
    frame_size = frame.shape[0]*frame.shape[1]*frame.shape[2]
    frame_size_bytes = struct.pack('i', frame_size)
    print(f"Frame size: {frame_size} bytes")

    # Send image
    t0 = time.time()
    # udp_socket.sendto(frame, target_addr)
    send_message(frame)
    t = time.time() - t0

    # Iteracctions
    iteracctions += 1

    # Infererence for a fixed amount of time
    if time.time() - T0 > 60:
        break


cap.release()
cv2.destroyAllWindows()
udp_socket.close()


# Print stats
print(f"\nTotal iteracctions: {iteracctions+1}")
print(f"time to read frame: {np.mean(t_read_frame)*1000:.2f} ms")
print(f"time to preprocess: {np.mean(t_preprocess)*1000:.2f} ms")
print(f"time to inference: {np.mean(t_inference)*1000:.2f} ms")
print(f"time to postprocess: {np.mean(t_postprocess)*1000:.2f} ms")
print(f"FPS: {np.mean(FPS_list):.2f}")