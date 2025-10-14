"""
Servidor Flask con WebSocket para Sistema de Detección
Procesa imágenes y videos subidos por el usuario
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import threading
import time
import os
from werkzeug.utils import secure_filename
from detector import MultiObjectDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'securevision-ai-pro-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=100 * 1024 * 1024)

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variables globales
detector = None
current_video = None
detection_active = False
processing_thread = None
current_file_path = None
is_image = False

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'}


def allowed_file(filename, file_type='video'):
    """Verifica si el archivo es válido"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    elif file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    else:
        return ext in ALLOWED_VIDEO_EXTENSIONS or ext in ALLOWED_IMAGE_EXTENSIONS


def initialize_detector():
    """Inicializa el detector YOLO"""
    global detector
    try:
        print("🚀 Inicializando detector YOLO...")
        detector = MultiObjectDetector(confidence_threshold=0.4)
        print("✅ Detector YOLO inicializado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error al inicializar detector: {e}")
        return False


def process_image(image_path):
    """Procesa una imagen estática"""
    global detector
    
    try:
        print(f"📸 Procesando imagen: {image_path}")
        
        # Leer imagen
        frame = cv2.imread(image_path)
        if frame is None:
            raise Exception("No se pudo leer la imagen")
        
        # Detectar objetos
        annotated_frame, detections, stats = detector.detect(frame)
        
        # Convertir a base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Obtener estadísticas
        full_stats = detector.get_statistics()
        
        # Enviar resultado
        socketio.emit('image_processed', {
            'frame': frame_base64,
            'detections': detections,
            'stats': stats,
            'full_stats': full_stats,
            'is_image': True
        })
        
        print(f"✅ Imagen procesada: {len(detections)} objetos detectados")
        
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")
        socketio.emit('error', {'message': f'Error al procesar imagen: {str(e)}'})


def process_video_loop():
    """Loop de procesamiento de video"""
    global detection_active, current_video, detector
    
    print("🎬 Iniciando procesamiento de video...")
    frame_count = 0
    
    while detection_active and current_video and current_video.isOpened():
        try:
            ret, frame = current_video.read()
            
            if not ret:
                # Video terminado, reiniciar
                current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                socketio.emit('video_loop', {'message': 'Video reiniciado'})
                continue
            
            frame_count += 1
            
            # Detectar objetos
            annotated_frame, detections, stats = detector.detect(frame)
            
            # Convertir a base64
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Obtener estadísticas completas
            full_stats = detector.get_statistics()
            
            # Calcular progreso del video
            total_frames = int(current_video.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            
            # Enviar datos por WebSocket
            socketio.emit('detection_update', {
                'frame': frame_base64,
                'detections': detections,
                'stats': stats,
                'full_stats': full_stats,
                'frame_count': frame_count,
                'total_frames': total_frames,
                'progress': round(progress, 1),
                'is_image': False
            })
            
            # Control de velocidad (30 FPS aprox)
            time.sleep(0.033)
            
        except Exception as e:
            print(f"❌ Error en procesamiento: {e}")
            socketio.emit('error', {'message': f'Error: {str(e)}'})
            break
    
    print("🛑 Procesamiento de video detenido")
    detection_active = False


@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Estado del sistema"""
    return jsonify({
        'detector_ready': detector is not None,
        'video_loaded': current_video is not None and current_video.isOpened() if current_video else False,
        'detection_active': detection_active,
        'is_image': is_image,
        'timestamp': time.time()
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Maneja la subida de archivos (imagen o video)"""
    global current_video, current_file_path, is_image, detector
    
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió archivo'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400
    
    # Verificar tipo de archivo
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    is_video = file_ext in ALLOWED_VIDEO_EXTENSIONS
    is_img = file_ext in ALLOWED_IMAGE_EXTENSIONS
    
    if not is_video and not is_img:
        return jsonify({
            'error': f'Formato no soportado. Use: {", ".join(ALLOWED_VIDEO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS)}'
        }), 400
    
    try:
        # Guardar archivo
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{int(time.time())}_{filename}")
        file.save(filepath)
        current_file_path = filepath
        
        print(f"📁 Archivo guardado: {filepath}")
        
        # Resetear detector
        if detector:
            detector.reset()
        
        if is_img:
            # Es una imagen
            is_image = True
            print("📸 Archivo detectado: IMAGEN")
            
            # Procesar imagen inmediatamente
            process_image(filepath)
            
            return jsonify({
                'message': 'Imagen cargada y procesada',
                'type': 'image',
                'filename': filename
            })
            
        else:
            # Es un video
            is_image = False
            print("🎬 Archivo detectado: VIDEO")
            
            # Cerrar video anterior si existe
            if current_video:
                current_video.release()
            
            # Abrir nuevo video
            current_video = cv2.VideoCapture(filepath)
            
            if not current_video.isOpened():
                return jsonify({'error': 'No se pudo abrir el video'}), 500
            
            # Obtener información del video
            fps = current_video.get(cv2.CAP_PROP_FPS)
            frame_count = int(current_video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(current_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(current_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"📊 Video: {width}x{height}, {fps} FPS, {frame_count} frames, {duration:.1f}s")
            
            return jsonify({
                'message': 'Video cargado correctamente',
                'type': 'video',
                'filename': filename,
                'info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': round(duration, 2)
                }
            })
            
    except Exception as e:
        print(f"❌ Error al procesar archivo: {e}")
        return jsonify({'error': f'Error al procesar: {str(e)}'}), 500


@app.route('/api/start', methods=['POST'])
def start_detection():
    """Inicia el procesamiento"""
    global detection_active, processing_thread, current_video, is_image, current_file_path
    
    if not detector:
        return jsonify({'error': 'Detector no inicializado'}), 500
    
    if is_image:
        # Si es imagen, solo re-procesar
        if current_file_path and os.path.exists(current_file_path):
            process_image(current_file_path)
            return jsonify({'message': 'Imagen re-procesada'})
        else:
            return jsonify({'error': 'No hay imagen cargada'}), 400
    
    # Si es video
    if not current_video or not current_video.isOpened():
        return jsonify({'error': 'No hay video cargado'}), 400
    
    if detection_active:
        return jsonify({'message': 'Procesamiento ya activo'})
    
    # Reiniciar video al inicio
    current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    detection_active = True
    processing_thread = threading.Thread(target=process_video_loop, daemon=True)
    processing_thread.start()
    
    return jsonify({'message': 'Procesamiento iniciado'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Detiene el procesamiento"""
    global detection_active
    
    detection_active = False
    
    if processing_thread:
        processing_thread.join(timeout=2)
    
    return jsonify({'message': 'Procesamiento detenido'})


@app.route('/api/reset', methods=['POST'])
def reset_counters():
    """Reinicia los contadores"""
    global detector, current_video, current_file_path, is_image
    
    if detector:
        detector.reset()
    
    # Reiniciar video si existe
    if current_video and current_video.isOpened() and not is_image:
        current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    return jsonify({'message': 'Contadores reiniciados'})


@app.route('/api/statistics')
def statistics():
    """Obtiene estadísticas del sistema"""
    if detector:
        return jsonify(detector.get_statistics())
    return jsonify({'error': 'Detector no inicializado'}), 500


@app.route('/api/history')
def history():
    """Obtiene el historial de detecciones"""
    if detector:
        return jsonify({
            'history': detector.detection_history,
            'total': len(detector.detection_history)
        })
    return jsonify({'error': 'Detector no inicializado'}), 500


@socketio.on('connect')
def handle_connect():
    """Cliente conectado"""
    print('✅ Cliente web conectado')
    emit('connection_response', {
        'status': 'connected',
        'detector_ready': detector is not None
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Cliente desconectado"""
    print('❌ Cliente web desconectado')


@socketio.on('request_reprocess')
def handle_reprocess():
    """Re-procesar archivo actual"""
    global current_file_path, is_image
    
    if is_image and current_file_path:
        process_image(current_file_path)
    else:
        emit('error', {'message': 'No hay imagen para re-procesar'})


def cleanup():
    """Limpieza al cerrar"""
    global detection_active, current_video
    
    print("\n🧹 Limpiando recursos...")
    
    detection_active = False
    
    if current_video:
        current_video.release()
    
    print("✅ Recursos liberados")


if __name__ == '__main__':
    print("=" * 70)
    print("   SISTEMA DE DETECCIÓN MULTICLASE - SERVIDOR WEB")
    print("=" * 70)
    print()
    
    # Inicializar detector
    if not initialize_detector():
        print("❌ No se pudo inicializar el detector. Saliendo...")
        exit(1)
    
    print()
    print("🌐 Servidor Web: http://localhost:5000")
    print("📡 WebSocket: ws://localhost:5000")
    print()
    print("📋 Funcionalidades:")
    print("  ✅ Sube IMÁGENES → Detección instantánea")
    print("  ✅ Sube VIDEOS → Detección frame por frame")
    print("  ✅ Tracking automático de objetos")
    print("  ✅ Exportación de datos a CSV")
    print()
    print("📁 Formatos soportados:")
    print(f"  Imágenes: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}")
    print(f"  Videos: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}")
    print()
    print("=" * 70)
    print()
    
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Interrupción recibida")
    finally:
        cleanup()
        print("\n✅ Servidor cerrado correctamente")