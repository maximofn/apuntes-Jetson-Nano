import cv2
import socket
import numpy as np

# Creamos el socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Asignamos la dirección y puerto del socket
sock.bind(('localhost', 8554))

len_message = 0
len_message_old = 0

frame = []
start_frame = False
end_frame = False
num_end_end_frame = 0
resize = False
image = False

# Creamos un bucle infinito para recibir y mostrar el vídeo por el socket
while True:
    # Recibimos el mensaje del socket
    message, addr = sock.recvfrom(65000)
    # print(f"len_message: {len(message)}")

    # len_message = len(message)
    # if len_message != len_message_old:
    #     print(f"len_message: {len_message}, type: {type(len_message)}")
    #     len_message_old = len_message

    if len(message) == 10:
        print("******************************************start frame")
        start_frame = True
        continue

    # Creamos un array NumPy a partir de la secuencia de bytes
    if len(message) == 20:
        print(f"--------------------------------------------end frame, len frame: {len(frame)}")
        end_frame = True
        if num_end_end_frame == 0:
            num_end_end_frame += 1
        elif num_end_end_frame == 1:
            num_end_end_frame += 1
    else:
        if start_frame:
            frame = np.frombuffer(message, dtype=np.uint8)
            start_frame = False
        else:
            frame = np.append(frame, np.frombuffer(message, dtype=np.uint8))
        print(f"len message: {len(message)}, len frame: {len(frame)}")

    # Si hemos recibido el frame completo, lo mostramos
    if end_frame and num_end_end_frame > 1:
        # Redimensionamos el array NumPy a una imagen de 3 canales
        print(f"len frame: {len(frame)}")
        frame = frame.reshape(720, 1280, 3)

        # # Redimensionamos el frame a un tamaño de 640x480
        # if resize: frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)

        # # Mostramos la imagen en una ventana
        cv2.imshow('Video', frame)
        
        end_frame = False

    # Si se presiona la tecla 'q', salimos del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerramos la ventana y el socket
cv2.destroyAllWindows()
sock.close()
