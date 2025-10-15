"""
Sistema de Detección Multiclase con YOLO
Detecta: Personas, Vehículos, Animales y Objetos
Autor: Sistema SecureVision AI Pro
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import json

class MultiObjectDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Inicializa el detector con YOLOv8
        
        Args:
            model_path: Ruta al modelo YOLO
            confidence_threshold: Umbral de confianza para detecciones
        """
        print("🚀 Inicializando detector YOLO...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Categorías de detección
        self.categories = {
            'personas': [0],  # person
            'vehiculos': [2, 3, 5, 7],  # car, motorcycle, bus, truck
            'animales': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # bird, cat, dog, horse, etc.
            'objetos': list(range(24, 80))  # otros objetos comunes
        }
        
        # Nombres de clases COCO
        self.class_names = self.model.names
        
        # Sistema de tracking
        self.tracked_objects = defaultdict(lambda: {
            'id': None,
            'category': None,
            'first_seen': None,
            'last_seen': None,
            'count': 0,
            'bbox': None
        })
        
        self.next_id = 1
        self.detection_history = []
        
        # Contadores
        self.counters = {
            'personas': 0,
            'vehiculos': 0,
            'animales': 0,
            'objetos': 0,
            'total': 0
        }
        
        print("✅ Detector inicializado correctamente")
    
    def categorize_detection(self, class_id):
        """Categoriza una detección según su clase"""
        for category, class_ids in self.categories.items():
            if class_id in class_ids:
                return category
        return 'objetos'
    
    def calculate_iou(self, box1, box2):
        """Calcula Intersection over Union entre dos cajas"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def assign_tracking_id(self, bbox, category, class_name):
        """Asigna ID de tracking a una detección"""
        current_time = time.time()
        best_match = None
        best_iou = 0
        
        # Buscar coincidencia con objetos existentes
        for obj_id, obj_data in list(self.tracked_objects.items()):
            if obj_data['category'] == category:
                iou = self.calculate_iou(bbox, obj_data['bbox'])
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    best_match = obj_id
        
        if best_match:
            # Actualizar objeto existente
            self.tracked_objects[best_match].update({
                'last_seen': current_time,
                'bbox': bbox,
                'count': self.tracked_objects[best_match]['count'] + 1
            })
            return best_match
        else:
            # Nuevo objeto detectado
            new_id = f"{category[0].upper()}{self.next_id:04d}"
            self.next_id += 1
            self.counters[category] += 1
            self.counters['total'] += 1
            
            self.tracked_objects[new_id] = {
                'id': new_id,
                'category': category,
                'class_name': class_name,
                'first_seen': current_time,
                'last_seen': current_time,
                'count': 1,
                'bbox': bbox
            }
            
            return new_id
    
    def clean_old_tracks(self, timeout=3.0):
        """Limpia objetos que no se han visto recientemente"""
        current_time = time.time()
        to_remove = []
        
        for obj_id, obj_data in self.tracked_objects.items():
            if current_time - obj_data['last_seen'] > timeout:
                to_remove.append(obj_id)
                
                # Guardar en historial
                self.detection_history.append({
                    'id': obj_id,
                    'category': obj_data['category'],
                    'class_name': obj_data['class_name'],
                    'first_seen': obj_data['first_seen'],
                    'last_seen': obj_data['last_seen'],
                    'duration': obj_data['last_seen'] - obj_data['first_seen']
                })
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
    
    def detect(self, frame):
        """
        Realiza detección en un frame
        
        Args:
            frame: Frame de video (numpy array)
            
        Returns:
            frame_anotado, detecciones, estadisticas
        """
        start_time = time.time()
        
        # Ejecutar detección YOLO
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        annotated_frame = frame.copy()
        
        # Procesar cada detección
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extraer información
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                # Categorizar
                category = self.categorize_detection(class_id)
                
                # Asignar tracking ID
                bbox = [x1, y1, x2, y2]
                tracking_id = self.assign_tracking_id(bbox, category, class_name)
                
                # Dibujar en frame
                annotated_frame = self.draw_detection(
                    annotated_frame, 
                    bbox, 
                    tracking_id, 
                    category, 
                    class_name, 
                    confidence
                )
                
                # Guardar detección
                detections.append({
                    'id': tracking_id,
                    'category': category,
                    'class_name': class_name,
                    'bbox': bbox,
                    'confidence': confidence
                })
        
        # Limpiar tracks antiguos
        self.clean_old_tracks()
        
        # Calcular estadísticas
        process_time = (time.time() - start_time) * 1000
        fps = 1000 / process_time if process_time > 0 else 0
        
        stats = {
            'detections': len(detections),
            'active_tracks': len(self.tracked_objects),
            'counters': self.counters.copy(),
            'process_time_ms': round(process_time, 2),
            'fps': round(fps, 1)
        }
        
        return annotated_frame, detections, stats
    
    def draw_detection(self, frame, bbox, tracking_id, category, class_name, confidence):
        """Dibuja una detección en el frame"""
        x1, y1, x2, y2 = bbox
        
        # Colores por categoría
        colors = {
            'personas': (6, 182, 212),      # Cyan
            'vehiculos': (34, 197, 94),     # Verde
            'animales': (251, 191, 36),     # Amarillo
            'objetos': (168, 85, 247)       # Púrpura
        }
        
        color = colors.get(category, (255, 255, 255))
        
        # Dibujar rectángulo principal
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Esquinas decorativas
        corner_length = 15
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 4)
        
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 4)
        
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 4)
        
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 4)
        
        # Etiqueta superior
        label = f"{tracking_id} - {class_name}"
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - 10),
            (x1 + label_width + 10, y1),
            color,
            -1
        )
        
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Barra de confianza
        conf_width = int((x2 - x1) * confidence)
        cv2.rectangle(frame, (x1, y2 + 2), (x1 + conf_width, y2 + 8), (34, 197, 94), -1)
        
        # Porcentaje
        conf_text = f"{int(confidence * 100)}%"
        cv2.putText(
            frame,
            conf_text,
            (x1 + 5, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return frame
    
    def get_statistics(self):
        """Obtiene estadísticas completas del sistema"""
        return {
            'counters': self.counters.copy(),
            'active_tracks': len(self.tracked_objects),
            'total_history': len(self.detection_history),
            'tracked_objects': {
                obj_id: {
                    'category': data['category'],
                    'class_name': data['class_name'],
                    'duration': time.time() - data['first_seen']
                }
                for obj_id, data in self.tracked_objects.items()
            }
        }
    
    def reset(self):
        """Reinicia todos los contadores y tracking"""
        self.tracked_objects.clear()
        self.detection_history.clear()
        self.counters = {
            'personas': 0,
            'vehiculos': 0,
            'animales': 0,
            'objetos': 0,
            'total': 0
        }
        self.next_id = 1
        print("🔄 Sistema reiniciado")


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("   SISTEMA DE DETECCIÓN MULTICLASE - YOLO")
    print("=" * 60)
    
    # Inicializar detector
    detector = MultiObjectDetector()
    
    # Abrir video o cámara
    print("\n📹 Abriendo cámara...")
    cap = cv2.VideoCapture(0)  # Cambiar a ruta de video si es necesario
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        exit()
    
    print("✅ Cámara abierta correctamente")
    print("\nControles:")
    print("  - Presiona 'q' para salir")
    print("  - Presiona 'r' para reiniciar contadores")
    print("  - Presiona 's' para ver estadísticas")
    print("\n" + "=" * 60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar
        annotated_frame, detections, stats = detector.detect(frame)
        
        # Mostrar estadísticas en pantalla
        y_offset = 30
        cv2.putText(annotated_frame, f"FPS: {stats['fps']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(annotated_frame, f"Personas: {stats['counters']['personas']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (6, 182, 212), 2)
        
        y_offset += 30
        cv2.putText(annotated_frame, f"Vehiculos: {stats['counters']['vehiculos']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 197, 94), 2)
        
        y_offset += 30
        cv2.putText(annotated_frame, f"Animales: {stats['counters']['animales']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 191, 36), 2)
        
        y_offset += 30
        cv2.putText(annotated_frame, f"Objetos: {stats['counters']['objetos']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (168, 85, 247), 2)
        
        # Mostrar frame
        cv2.imshow('Sistema de Detección Multiclase', annotated_frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
        elif key == ord('s'):
            stats = detector.get_statistics()
            print("\n" + "=" * 60)
            print("ESTADÍSTICAS DEL SISTEMA")
            print("=" * 60)
            print(json.dumps(stats, indent=2))
            print("=" * 60 + "\n")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Sistema finalizado correctamente")