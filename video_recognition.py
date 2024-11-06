import cv2
import mediapipe as mp
from deepface import DeepFace
import os
from tqdm import tqdm

def escrever_relatorio(filePath, report):
    report_file = open(filePath, "w")
    report_file.write(report)
    report_file.close()

def montar_relatorio(filePath, frameCount, anomalyCount, emotionsList, counterSuddenMovementsDetected):
    report_str = ""
    report_str += "Total de frames analisados: {}\n".format(frameCount)
    report_str += "Total de mãos erguidas contabilizadas: {}\n".format(anomalyCount)
    report_str += "Principais emoções detectadas: {}\n".format(", ".join(emotionsList))
    report_str += "Total de movimentos bruscos detectados: {}\n".format(counterSuddenMovementsDetected)
    escrever_relatorio(filePath, report_str)

def detect_emotions_and_pose(video_path, output_path, report_path):
    # Carrega o classificador de rosto pré-treinado do OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Parametros para definir a detecção de movimentos brusos
    threshold = 50
    area_threshold = 500
    
    # Iniciando os contadores para o relatório final
    arm_movements_count = 0
    detectes_emotions_list = []
    counter_sudden_movements_detected = 0
    
    # Inicializar o MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
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
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get the first frame to analyse
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

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar o frame para detectar a pose
        results_pose = pose.process(rgb_frame)

        # Função para verificar se o braço está levantado
        def is_arm_up(landmarks):
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

            left_arm_up = left_elbow.y < left_eye.y
            right_arm_up = right_elbow.y < right_eye.y

            return left_arm_up or right_arm_up
        
        # Desenhar as anotações da pose no frame
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Verificar se o braço está levantado
            if is_arm_up(results_pose.pose_landmarks.landmark):
                if not arm_up:
                    arm_up = True
                    arm_movements_count += 1
            else:
                arm_up = False

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
        
        for contour in contours:
            if cv2.contourArea(contour) > area_threshold:
                counter_sudden_movements_detected += 1
        
        # Detecta rostos no frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
        
        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    # Escrever relatorio com as informações coletadas no video
    montar_relatorio(report_path, total_frames, arm_movements_count, detectes_emotions_list, counter_sudden_movements_detected)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))


input_video_path = os.path.join(script_dir, 'videos_input/top_model_video.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
# input_video_path = os.path.join(script_dir, 'videos_input/video.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
output_video_path = os.path.join(script_dir, 'output/detect_expressions_output_video.mp4')  # Nome do vídeo de saída
output_report_path = os.path.join(script_dir, 'output/detect_expressions_output_video_report.txt')  # Nome do arquivo de analise do video de saída

# Chamar a função para detectar emoções no vídeo e salvar o vídeo processado
detect_emotions_and_pose(input_video_path, output_video_path, output_report_path)