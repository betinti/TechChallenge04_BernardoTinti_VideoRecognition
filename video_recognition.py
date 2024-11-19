import cv2
import mediapipe as mp
from deepface import DeepFace
import os
from tqdm import tqdm

# Globals
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection

#  Escreve o relatorio em um arquivo .txt
def escrever_relatorio(filePath, report):
    report_file = open(filePath, "w")
    report_file.write(report)
    report_file.close()

# Monta o texto a ser escrito no relatorio 
def montar_relatorio(file_path, result_dict):
    # Converte um dicionário de resultados em um formato textual organizado para o relatório.
    reportStr = ""
    
    for key, value in result_dict.items():
        if isinstance(value, str):
            reportStr += f"{key}: {value}\n"
        elif isinstance(value, list) and len(value)>0:
            reportStr += f"{key}: {", ".join(value)}\n"
        elif  (isinstance(value, int) or isinstance(value, float)) and value > 0:
            reportStr += f"{key}: {str(value)}\n"
            
    escrever_relatorio(file_path, reportStr)

# Função para reconhecer gestos com base nos pontos de referência das mãos
def recognize_gesture(landmarks):
    """
    Analisa os pontos de referência das mãos para identificar gestos específicos.
    Retorna o nome do gesto detectado.
    """
    # Dedão para cima
    if (landmarks[mp_holistic.HandLandmark.THUMB_TIP].y < landmarks[mp_holistic.HandLandmark.WRIST].y and
        landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_holistic.HandLandmark.THUMB_TIP].y):
        return "Dedao para cima"
    # Aceno de mão (movimento da mão)
    elif landmarks[mp_holistic.HandLandmark.WRIST].visibility < 0.9:
        return "Aceno de mao"
    # Sinal de paz (dedo indicador e médio afastados)
    elif (landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y and
          landmarks[mp_holistic.HandLandmark.THUMB_TIP].y > landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y):
        return "Sinal de paz"
    # Apontando (dedo indicador estendido)
    elif landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].visibility > 0.9:
        return "Apontando"
    # Contando dedos
    else:
        # Conta o número de dedos visíveis
        visible_fingers = sum(1 for lm in landmarks if lm.visibility > 0.9)
        return f"{visible_fingers} Dedo(s)"

# Função para reconhecer posturas corporais
def recognize_posture(pose_landmarks):
    """
    Analisa os pontos de referência do corpo para identificar a postura (sentado ou em pé).
    Baseia-se na posição dos quadris.
    """
    hip_landmarks = [pose_landmarks.landmark[i] for i in [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]]
    if all(hip.y > 0.7 for hip in hip_landmarks):
        return "Sentado"
    else:
        return "Em pe"

# Função para verificar se o braço está levantado
def is_arm_up(landmarks):
    """
    Verifica se um dos braços está levantado com base na posição do cotovelo em relação aos olhos.
    """
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    left_arm_up = left_elbow.y < left_eye.y
    right_arm_up = right_elbow.y < right_eye.y

    return left_arm_up or right_arm_up

# Função para verificar a postura de "T" (braços em posição de T)
def recognize_T_posture(landmarks):
    """
    Detecta se os braços estão em posição de T, ou seja, entre os olhos e os ombros.
    """
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    left_t = left_wrist.y < left_shoulder.y and left_wrist.y > left_eye.y
    right_t = right_wrist.y < right_shoulder.y  and right_wrist.y > right_eye.y
    
    return left_t and right_t

# Cria pastas de saída com base no nome do vídeo de entrada
def handle_videos_paths(input_video_full_path):
    """
    Cria pastas de saída e retorna os caminhos para o relatório e o vídeo processado.
    """
    
    # Obtenha apenas o nome do arquivo sem a extensão
    videoName = os.path.splitext(os.path.basename(input_video_full_path))[0]
    
    # Caminho para a mesma pasta do script
    outpuPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'output/{videoName}')
    os.makedirs(outpuPath, exist_ok=True)
    
    ouputPath_Report = os.path.join(outpuPath, f'{videoName}_report.txt')
    ouputPath_Video = os.path.join(outpuPath, f'{videoName}_analyzed_video.mp4')
    return ouputPath_Report, ouputPath_Video
    

def detect_emotions_and_pose(video_path, optimized = False):
    """
    Processa o vídeo de entrada para detectar emoções, gestos, posturas corporais e movimentos bruscos.
    Gera um relatório com os resultados e salva um vídeo processado.
    """
    
    outputReport_path, outputVideo_path = handle_videos_paths(video_path)
    
    # Carrega o classificador de rosto pré-treinado do OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Iniciando os contadores para o relatório final
    counter_sudden_movements_detected = 0
    arm_movements_count = 0
    t_posture_count = 0
    detectes_emotions_list = []
    detectes_hand_gestures = []
    detectes_postures = []
    t_posture  = False
    arm_up  = False
    sudden_movements_detected = False
    total_frame_counter = 0
    total_frame_analyzed = 0
    intervalo_entre_frames = 15
    
    # Parametros para definir a detecção de movimentos brusos
    threshold = 100
    area_threshold = 1000
    
    # Inicializar o MediaPipe Pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(outputVideo_path, fourcc, fps, (width, height))

    # Obtenha o primeiro quadro para analisar
    _, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (5, 5), 0)

    # Loop para processar cada frame do vídeo
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break
        
        total_frame_counter += 1
        
        if (not optimized) or (total_frame_counter % intervalo_entre_frames == 0):
            
            total_frame_analyzed += 1
            
            # Detecta rostos no frame
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Iterar sobre cada face detectada
            for x, y, w, h in faces:
                # Obter a emoção dominante
                face = frame[y:y+h, x:x+w]
                
                # Analisar o frame para detectar faces e expressões
                result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                
                if dominant_emotion not in detectes_emotions_list:
                    detectes_emotions_list.append(dominant_emotion)
                
                # Desenhar um retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Escrever a emoção dominante acima da face
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Converter o frame para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processar o frame para detectar a pose
            frame.flags.writeable = False
            pose_results = pose.process(rgb_frame)
            holistic_results = holistic.process(rgb_frame)

            # Desenhe a pose e os pontos de referência das mãos na imagem
            frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Verificar se o braço está levantado
                if is_arm_up(pose_results.pose_landmarks.landmark):
                    arm_movements_count += 1
                    if not arm_up:
                        arm_up = True
                else:
                    arm_up = False
                    
                # Verificar se existe posição dos braços em T
                if recognize_T_posture(pose_results.pose_landmarks.landmark):
                    if not t_posture:
                        t_posture = True
                        t_posture_count += 1
                else:
                    t_posture = False
                
                posture = recognize_posture(pose_results.pose_landmarks)
                if posture not in detectes_postures:
                    detectes_postures.append(posture)
                    
            if holistic_results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
                
            if holistic_results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
                    
            # Reconhecer gestos
            if holistic_results.right_hand_landmarks:
                gesture = recognize_gesture(holistic_results.right_hand_landmarks.landmark)
                if gesture not in detectes_hand_gestures:
                    detectes_hand_gestures.append(gesture)
                    
            # Converte o frame para escala de cinza para detecção de rosto
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate absolute difference between current frame and previous frame
            frame_diff = cv2.absdiff(prev_frame, gray)
            
            # Thresholding to highlight significant differences
            _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Finding contours to detect regions with large movement
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Identifica os momentos de movimentos anomalos
            for contour in contours:
                if cv2.contourArea(contour) > area_threshold:
                    if not sudden_movements_detected:
                        sudden_movements_detected = True
                        counter_sudden_movements_detected += 1
                        (xT, yT, wT, hT) = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (xT, yT), (xT + wT, yT + hT), (0, 255, 0), 2)
                else:
                    sudden_movements_detected = False
                    
            # Update previous frame
            prev_frame = gray

            # Escrever o frame processado no vídeo de saída
        out.write(frame)

    # Criando um dicionario para imprimir os resultados
    resultsToReport = {
        "Total de frames analisados": total_frame_analyzed,
        "Total de maos erguidas contabilizadas": arm_movements_count,
        "Principais emocoes detectadas": detectes_emotions_list,
        "Quantidade de emocoes detectadas": len(detectes_emotions_list),
        "Gestos de mao detectadas": detectes_hand_gestures,
        "Posicoes corporais detectadas": detectes_postures,
        "Posicao dos bracos em T": t_posture_count,
        "Movimentos anomalos": counter_sudden_movements_detected
    }
    
    # Escrever relatorio com as informações coletadas no video
    montar_relatorio(outputReport_path, resultsToReport)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Chamar a função para detectar emoções no vídeo e salvar o vídeo processado
# detect_emotions_and_pose(os.path.join(script_dir, 'videos_input/cadeira_teste.mp4'))
detect_emotions_and_pose(os.path.join(script_dir, 'videos_input/teste_movimentos.mp4'))
# detect_emotions_and_pose(os.path.join(script_dir, 'videos_input/big_t_body_test.mp4'))
# detect_emotions_and_pose(os.path.join(script_dir, 'videos_input/video.mp4'), False)