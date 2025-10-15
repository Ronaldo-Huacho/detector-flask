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
import mysql.connector
from mysql.connector import Error
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'securevision-ai-pro-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=100 * 1024 * 1024)

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuración MySQL
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'root',
    'password': 'Ronaldo123',
    'database': 'detector_db'
}

# Variables globales
detector = None
current_video = None
detection_active = False
processing_thread = None
current_file_path = None
is_image = False
db_connection = None

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'}


def get_db_connection():
    """Obtiene conexión a MySQL"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"❌ Error conectando a MySQL: {e}")
    return None


def save_detection_to_db(detection_data):
    """Guarda una detección en la base de datos"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        query = """
        INSERT INTO detecciones (
            tracking_id, category, class_name,
            color_dominante, color_rgb, forma,
            confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            width, height, area, aspect_ratio,
            fecha_deteccion, primera_aparicion
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        values = (
            detection_data.get('tracking_id', 'UNKNOWN'),
            detection_data.get('category', 'unknown'),
            detection_data.get('class_name', 'unknown'),
            detection_data.get('color_dominante', 'N/A'),
            detection_data.get('color_rgb', 'N/A'),
            detection_data.get('forma', 'Rectángulo'),
            float(detection_data.get('confidence', 0.0)),
            int(detection_data.get('bbox_x1', 0)),
            int(detection_data.get('bbox_y1', 0)),
            int(detection_data.get('bbox_x2', 0)),
            int(detection_data.get('bbox_y2', 0)),
            int(detection_data.get('width', 0)),
            int(detection_data.get('height', 0)),
            int(detection_data.get('area', 0)),
            float(detection_data.get('aspect_ratio', 0.0)),
            detection_data.get('fecha_deteccion', datetime.now()),
            detection_data.get('primera_aparicion', datetime.now())
        )
        
        cursor.execute(query, values)
        conn.commit()
        
        return True
        
    except Error as e:
        print(f"❌ Error MySQL: {e}")
        return False
    except Exception as e:
        print(f"❌ Error guardando detección: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def prepare_detection_for_db(detection, category):
    """Prepara los datos de detección para la base de datos"""
    now = datetime.now()
    
    # Manejo seguro de bbox
    bbox = detection.get('bbox', [0, 0, 0, 0])
    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox = [0, 0, 0, 0]
    
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    
    # Manejo seguro de color RGB
    color_rgb = detection.get('color_rgb', 'N/A')
    if isinstance(color_rgb, (list, tuple)) and len(color_rgb) >= 3:
        color_rgb = f"{int(color_rgb[0])},{int(color_rgb[1])},{int(color_rgb[2])}"
    elif not isinstance(color_rgb, str):
        color_rgb = 'N/A'
    
    return {
        'tracking_id': str(detection.get('tracking_id', 'UNKNOWN')),
        'category': str(category),
        'class_name': str(detection.get('class', detection.get('class_name', 'unknown'))),
        'color_dominante': str(detection.get('color', detection.get('color_dominante', 'N/A'))),
        'color_rgb': color_rgb,
        'forma': str(detection.get('shape', detection.get('forma', 'Rectángulo'))),
        'confidence': float(detection.get('confidence', 0.0)),
        'bbox_x1': int(bbox[0]),
        'bbox_y1': int(bbox[1]),
        'bbox_x2': int(bbox[2]),
        'bbox_y2': int(bbox[3]),
        'width': width,
        'height': height,
        'area': width * height,
        'aspect_ratio': round(width / height, 2) if height > 0 else 0,
        'fecha_deteccion': now,
        'primera_aparicion': now
    }


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
        
        # Guardar detecciones en MySQL
        if isinstance(detections, dict):
            for category, objects in detections.items():
                for obj in objects:
                    detection_data = prepare_detection_for_db(obj, category)
                    save_detection_to_db(detection_data)
        elif isinstance(detections, list):
            # Si es una lista, asumir categoría genérica
            for obj in detections:
                category = obj.get('category', 'objeto')
                detection_data = prepare_detection_for_db(obj, category)
                save_detection_to_db(detection_data)
        
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
            
            # Guardar detecciones en MySQL (cada 30 frames para no saturar)
            if frame_count % 30 == 0:
                if isinstance(detections, dict):
                    for category, objects in detections.items():
                        for obj in objects:
                            detection_data = prepare_detection_for_db(obj, category)
                            save_detection_to_db(detection_data)
                elif isinstance(detections, list):
                    for obj in detections:
                        category = obj.get('category', 'objeto')
                        detection_data = prepare_detection_for_db(obj, category)
                        save_detection_to_db(detection_data)
            
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
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)


@app.route('/api/status')
def status():
    """Estado del sistema"""
    # Verificar conexión MySQL
    mysql_status = False
    try:
        conn = get_db_connection()
        if conn:
            mysql_status = True
            conn.close()
    except:
        pass
    
    return jsonify({
        'detector_ready': detector is not None,
        'video_loaded': current_video is not None and current_video.isOpened() if current_video else False,
        'detection_active': detection_active,
        'is_image': is_image,
        'mysql_connected': mysql_status,
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


@app.route('/api/db/detections')
def get_db_detections():
    """Obtiene detecciones desde MySQL"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'No se pudo conectar a MySQL'}), 500
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM detecciones 
            ORDER BY fecha_deteccion DESC 
            LIMIT 100
        """)
        
        detections = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'detections': detections,
            'total': len(detections)
        })
        
    except Error as e:
        return jsonify({'error': f'Error MySQL: {str(e)}'}), 500


@socketio.on('connect')
def handle_connect():
    """Cliente conectado"""
    print('✅ Cliente web conectado')
    
    # Verificar MySQL
    mysql_ok = False
    try:
        conn = get_db_connection()
        if conn:
            mysql_ok = True
            conn.close()
    except:
        pass
    
    emit('connection_response', {
        'status': 'connected',
        'detector_ready': detector is not None,
        'mysql_connected': mysql_ok
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
    
    # Verificar conexión MySQL
    print("🔌 Verificando conexión MySQL...")
    conn = get_db_connection()
    if conn:
        print("✅ MySQL conectado correctamente")
        conn.close()
    else:
        print("⚠️ MySQL no disponible - las detecciones NO se guardarán en DB")
    print()
    
    # Inicializar detector
    if not initialize_detector():
        print("❌ No se pudo inicializar el detector. Saliendo...")
        exit(1)
    
    print()
    print("🌐 Servidor Web: http://localhost:5000")
    print("📡 WebSocket: ws://localhost:5000")
    print("🗄️ MySQL: localhost:3307/detector_db")
    print()
    print("📋 Funcionalidades:")
    print("  ✅ Sube IMÁGENES → Detección instantánea")
    print("  ✅ Sube VIDEOS → Detección frame por frame")
    print("  ✅ Tracking automático de objetos")
    print("  ✅ Guardado automático en MySQL")
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