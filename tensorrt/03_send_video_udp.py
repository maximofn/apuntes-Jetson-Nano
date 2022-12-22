import cv2
import socket
import threading
import sys
import select
import time

# Creamos una clase que hereda de threading.Thread y sobrescribe el método
# run(). Este método será el encargado de verificar la entrada de la terminal.
class InputThread(threading.Thread):
    data = None

    def run(self):
        # Configuramos el descriptor de archivo de la entrada de la terminal como stdin.
        input_fd = sys.stdin.fileno()

        # Usamos el método select.select() para ver si hay algún dato disponible
        # para leer en la entrada de la terminal.
        read_fds, _, _ = select.select([input_fd], [], [])

        # Si read_fds contiene el descriptor de archivo de la entrada de la terminal,
        # significa que hay algún dato disponible para leer. En ese caso, leemos el
        # dato utilizando el método sys.stdin.readline().
        if input_fd in read_fds:
            self.data = sys.stdin.readline()
    
    def get_data(self):
        return self.data
    
    def clear_data(self):
        self.data = None


# Creamos una instancia de la clase InputThread y la iniciamos.
input_thread = InputThread()
input_thread.start()

# Creamos el socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Abrimos la cámara en full HD
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

len_frame = 0
len_frame_old = 0
len_resized_frame = 0
len_resized_frame_old = 0

display = False
resize = False

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
    if resize: frame = cv2.resize(frame, (170, 127), interpolation=cv2.INTER_NEAREST)

    # Imprimimos el tamaño del frame redimensionado
    len_resized_frame = len(frame.tobytes())
    if len_resized_frame != len_resized_frame_old:
        print(f"len_resized_frame: {len_resized_frame}, shape: {frame.shape}")
        len_resized_frame_old = len_resized_frame

    # Display
    if display:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Enviamos el frame por el socket en bloques de tamaño máximo permitido
    message = frame.tobytes(order='C')
    block_size = 65000  # Tamaño máximo de cada bloque
    sock.sendto(message[:10], ('localhost', 8554))
    print("Send start")
    send = 0
    for i in range(0, len(message), block_size):
        sock.sendto(message[i:i + block_size], ('localhost', 8554))
        send += len(message[i:i + block_size])
        print(f"Send: {send} / {len(message)}")
        time.sleep(0.01)
    print(send)
    sock.sendto(message[:20], ('localhost', 8554))
    print("Send end")
    
    # If user press type 'q' into the console in non blocking mode, exit
    if input_thread.get_data() is not None and input_thread.get_data().strip() == 'q':
        print("Se ha ha parado por el usuario")
        input_thread.clear_data()
        break

# Cerramos el socket y la cámara
sock.close()
cap.release()
