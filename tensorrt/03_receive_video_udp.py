import cv2
import socket
import numpy as np

# Creamos el socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Asignamos la dirección y puerto del socket
sock.bind(('127.0.0.1', 8554))

len_message = 0
len_message_old = 0

frame = []

# Creamos un bucle infinito para recibir y mostrar el vídeo por el socket
while True:
    # Recibimos el mensaje del socket
    message, addr = sock.recvfrom(64770)#65507)

    len_message = len(message)
    if len_message != len_message_old:
        print(f"len_message: {len_message}, type: {type(len_message)}")
        len_message_old = len_message

    # Añadimos bytes al final del mensaje hasta que su tamaño sea divisible por 3
    # while len(message) % 65151 != 0:
    #     message += b'\x00'

    # Creamos un array NumPy a partir de la secuencia de bytes
    frame = np.frombuffer(message, dtype=np.uint8)

    # Redimensionamos el array NumPy a una imagen de 3 canales
    frame = frame.reshape(127, 170, 3)
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)

    # Mostramos la imagen en una ventana
    cv2.imshow('Video', frame)

    # Si se presiona la tecla 'q', salimos del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerramos la ventana y el socket
# cv2.destroyAllWindows()
sock.close()
