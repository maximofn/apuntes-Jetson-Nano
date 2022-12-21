import cv2
import socket
import numpy as np

# Creamos el socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Abrimos la cámara
cap = cv2.VideoCapture(0)
# Full HD
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

len_frame = 0
len_frame_old = 0
len_resized_frame = 0
len_resized_frame_old = 0

running = True

# Creamos un bucle infinito para enviar el vídeo por el socket
while True:
    # Leemos el frame de la cámara
    ret, frame = cap.read()

    # Si no hay frame, salimos del bucle
    if not ret:
        break

    # Imprimimos el tamaño del frame
    len_frame = len(frame.tobytes())
    if len_frame != len_frame_old:
        print(f"len_frame: {len_frame}, shape: {frame.shape}")
        len_frame_old = len_frame

    # Redimensionamos el array NumPy a un tamaño de 127x170
    frame = cv2.resize(frame, (170, 127), interpolation=cv2.INTER_NEAREST)

    # Imprimimos el tamaño del frame redimensionado
    len_resized_frame = len(frame.tobytes())
    if len_resized_frame != len_resized_frame_old:
        print(f"len_resized_frame: {len_resized_frame}, shape: {frame.shape}")
        len_resized_frame_old = len_resized_frame

    # Display
    # cv2.imshow("frame", frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break

    # Enviamos el frame por el socket en bloques de tamaño máximo permitido
    message = frame.tobytes(order='C')
    block_size = 65000  # Tamaño máximo de cada bloque
    for i in range(0, len(message), block_size):
        sock.sendto(message[i:i + block_size], ('127.0.0.1', 8554))

# Cerramos el socket y la cámara
sock.close()
cap.release()
