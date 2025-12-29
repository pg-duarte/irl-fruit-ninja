import cv2
import os

def detectar_e_preparar_rosto(frame, target_size=(100, 100)):
    # 1. Converter para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Carregar o classificador
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 3. Detetar rostos
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50),maxSize=(300,300))
    
    if len(faces) > 0:
        # Pegamos apenas o primeiro rosto detetado
        (x, y, w, h) = faces[0]
        
        # Extrair a ROI (Region of Interest)
        centroY = y + h // 2
        centroX = x + w // 2
        rosto_recortado = gray[centroY-target_size[1]//2:centroY+target_size[1]//2, centroX-target_size[0]//2:centroX+target_size[0]//2]
        
        # Redimensionar para o tamanho fixo (interpolação linear para qualidade)
        # rosto_redimensionado = cv2.resize(rosto_recortado, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Guardar o rosto (opcional, para debug)
        path = 'assets/images/face'
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(f"{path}/rosto.jpg", (rosto_recortado * 255).astype('uint8'))
        
        # Retornamos: Template, X inicial, Y inicial e o Frame com feedback
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return rosto_recortado, x, y, frame

    return None, None, None, frame