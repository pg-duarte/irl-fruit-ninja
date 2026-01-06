import cv2
import numpy as np
import pyopencl as cl
from stabilization.face_detection import detectar_e_preparar_rosto
from stabilization.stabilizer import GPUImageStabilizer
import os
# Força a escolha da primeira plataforma e dispositivo sem perguntar
os.environ['PYOPENCL_CTX'] = '0' 
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1' # Útil para ver erros de kernel

video_path = 'C:\\Users\\marci\\Desktop\\Mega\\TAPDI\\irl-fruit-ninja\\assets\\videos\\FirstVideo.mp4'  # Substitua pelo caminho do seu arquivo


def main():

    cap = cv2.VideoCapture(0) # Use a webcam como fonte de vídeo
    # cap = cv2.VideoCapture(video_path) # Use um arquivo de vídeo como fonte
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        exit()

    # Ler o primeiro frame para inicialização
    ret, first_frame = cap.read()
    cv2.waitKey(3000)  # Esperar a câmera estabilizar
    ret, first_frame = cap.read()
    if not ret:
        print("Erro ao ler o primeiro frame")
        return
    
    # Loop para capturar frame inicial
    while True:
        cv2.imshow('First Frame', first_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):  # 'r' para renovar frame
            ret, first_frame = cap.read()
            
            if ret:
                print("Frame renovado")
        elif key == ord('m'):  # 'm' para seguir em frente
            print("Iniciando estabilização...")
            break
        else:  # Atualizar câmera continuamente
            ret, first_frame = cap.read()
            if not ret:
                print("Erro ao ler frame")
                return
    
    cv2.destroyWindow('First Frame')

    # 1. Converter para cinzento e normalizar (OpenCL kernel espera float)
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) / 255.0
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray = clahe.apply((gray * 255).astype(np.uint8)) / 255.0
    
    # 2. Detetar face inicial 
    # Sugestão: Use o classificador Haar do OpenCV aqui para pegar o template real

    template, x0, y0, frame_processado = detectar_e_preparar_rosto(first_frame)
    template_normalizado = template/255.0
    # template_normalizado = template
    cv2.imshow('Deteção e Gravação', template)
    cv2.imshow('Frame processado', frame_processado)
    print(f"Template capturado em: x0={x0}, y0={y0}")

    # 3. Inicializar Estabilizador
    stabilizer = GPUImageStabilizer(template_normalizado, x0, y0, gray.shape, faceMargin=100, numAngles=4)

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        current_gray = clahe.apply((current_gray * 255).astype(np.uint8)) / 255.0
        
        # 4. CHAMADA DO ESTABILIZADOR
        stabilized, x1_global, y1_global, angle_degrees = stabilizer.process(current_gray)

        #Visualizar o mapa de custos
        debug_map = stabilizer.get_cost_map_visual()
        # debug_map = cv2.applyColorMap(255 - debug_map, cv2.COLORMAP_HOT)

        # # Desenhar círculo preto no centro da detecção
        # center_x = stabilizer.w_match 
        # center_y = stabilizer.h_match
        cv2.circle(current_gray, (int(x0), int(y0)), 30, (0, 0, 0), 4)
        cv2.circle(current_gray, (int(x1_global), int(y1_global)), 30, (0, 0, 0), 4)
        cv2.putText(current_gray, f"Angle: {angle_degrees:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        # Mostrar resultados
        cv2.imshow("Mapa de Custo (Onde a GPU procura)", debug_map)
        cv2.imshow("Original", current_gray)
        cv2.imshow("Estabilizada", stabilized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # cv2.waitKey(30)  # Delay in milliseconds (adjust as needed)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()