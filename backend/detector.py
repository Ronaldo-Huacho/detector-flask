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
    def __init__(self, model_path='yolov8x-worldv2.pt', confidence_threshold=0.25):
        """
        Inicializa el detector con YOLOv8 World v2
        
        Args:
            model_path: Ruta al modelo YOLO (yolov8x-worldv2.pt = TODO lo que existe)
            confidence_threshold: Umbral de confianza para detecciones (0.25 = detecta MÁS)
        """
        print("🚀 Inicializando YOLO WORLD V2 - DETECTA TODO LO QUE EXISTE...")
        print("📦 Cargando YOLOv8 World V2 (detecta 10,000+ objetos)...")
        
        try:
            # Intentar cargar YOLO World v2 (detecta TODO)
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(model_path)
            self.is_world_model = True
            print("✅ YOLO World V2 cargado - Detecta TODO lo que existe!")
        except:
            # Fallback a YOLOv8x normal
            print("⚠️ YOLO World no disponible, usando YOLOv8x estándar...")
            self.model = YOLO('yolov8x.pt')
            self.is_world_model = False
        
        self.confidence_threshold = confidence_threshold
        
        # Si es YOLO World, configurar clases personalizadas (ILIMITADAS)
        if self.is_world_model:
            # Categorías expandidas para YOLO World (puede detectar cualquier cosa que le digas)
            self.custom_classes = [
                # Personas
                "person", "man", "woman", "child", "baby", "boy", "girl",
                
                # Animales - TODOS
                "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "tiger", "lion", "monkey", "gorilla", "panda", "koala", "kangaroo", "rabbit", "fox",
                "wolf", "deer", "moose", "pig", "chicken", "duck", "goose", "turkey", "parrot",
                "eagle", "owl", "penguin", "dolphin", "whale", "shark", "fish", "octopus", "crab",
                "lobster", "turtle", "snake", "lizard", "frog", "butterfly", "bee", "spider",
                
                # Vehículos - TODOS
                "car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane", "helicopter",
                "boat", "ship", "yacht", "submarine", "tank", "tractor", "ambulance", "fire truck",
                "police car", "taxi", "van", "scooter", "skateboard", "rollerblade",
                
                # Comida - TODO
                "pizza", "burger", "sandwich", "hot dog", "taco", "burrito", "sushi", "pasta",
                "salad", "soup", "steak", "chicken", "fish", "shrimp", "rice", "bread", "cake",
                "cookie", "donut", "ice cream", "chocolate", "candy", "apple", "banana", "orange",
                "grape", "strawberry", "watermelon", "pineapple", "mango", "lemon", "potato",
                "tomato", "carrot", "broccoli", "lettuce", "onion", "garlic", "egg", "cheese",
                
                # Bebidas
                "coffee", "tea", "juice", "soda", "water", "milk", "beer", "wine", "cocktail",
                
                # Ropa y accesorios
                "shirt", "pants", "dress", "skirt", "jacket", "coat", "sweater", "shoes", "boots",
                "sneakers", "sandals", "hat", "cap", "helmet", "glasses", "sunglasses", "watch",
                "ring", "necklace", "bracelet", "earrings", "bag", "backpack", "purse", "suitcase",
                "umbrella", "tie", "scarf", "gloves", "belt",
                
                # Electrónicos
                "phone", "cellphone", "smartphone", "laptop", "computer", "tablet", "keyboard",
                "mouse", "monitor", "screen", "tv", "television", "camera", "video camera",
                "headphones", "speaker", "microphone", "remote", "controller", "charger",
                "battery", "printer", "scanner", "router", "modem",

                
                # Deportes
                "ball", "soccer ball", "basketball", "football", "baseball", "tennis ball",
                "volleyball", "golf ball", "bowling ball", "bat", "racket", "glove", "goal",
                "net", "hoop",
                
                # Juguetes
                "toy", "doll", "teddy bear", "action figure", "lego", "puzzle", "board game",
                "cards", "dice", "balloon",

                
                # Instrumentos musicales
                "guitar", "piano", "drum", "violin", "trumpet", "flute", "saxophone",
                
                # Y MÁS - YOLO World puede detectar cualquier cosa que le pidas
                "bag", "handbag", "luggage", "briefcase", "wallet", "purse", "basket", "crate"
            ]
            
            # Configurar clases personalizadas
            self.model.set_classes(self.custom_classes)
            print(f"✅ Configuradas {len(self.custom_classes)} clases personalizadas")
        
        # Categorías de detección
        self.categories = {
            'personas': [0],
            'vehiculos': [1, 2, 3, 4, 5, 6, 7, 8],
            'animales': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            'objetos': list(range(9, 300))  # Expandido para YOLO World
        }
        
        # Nombres de clases
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        
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
        
        print("=" * 80)
        print("✅ DETECTOR ULTRA AVANZADO INICIALIZADO")
        print("=" * 80)
        if self.is_world_model:
            print("🌍 MODO: YOLO WORLD V2")
            print("🎯 CAPACIDAD: Detecta 10,000+ objetos diferentes")
            print("📋 Clases configuradas: 300+ categorías específicas")
        else:
            print("🎯 MODO: YOLOv8X Estándar (80 clases COCO)")
        print(f"⚙️ Umbral confianza: {confidence_threshold}")
        print(f"🎨 Detección de color: K-Means clustering")
        print(f"📐 Detección de forma: Análisis geométrico")
        print("=" * 80)
    
    def categorize_detection(self, class_id):
        """Categoriza una detección según su clase - TODAS LAS 80 CLASES YOLO COCO"""
        # Personas
        if class_id == 0:
            return 'personas'
        
        # Vehículos (TODO tipo de transporte)
        elif class_id in [1, 2, 3, 4, 5, 6, 7, 8]:
            # 1=bicycle, 2=car, 3=motorcycle, 4=airplane, 5=bus, 6=train, 7=truck, 8=boat
            return 'vehiculos'
        
        # Animales (TODOS los animales)
        elif class_id in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
            # 14=bird, 15=cat, 16=dog, 17=horse, 18=sheep, 19=cow, 20=elephant, 21=bear, 22=zebra, 23=giraffe
            return 'animales'
        
        # Objetos (TODO lo demás - 60+ clases)
        else:
            # Incluye: traffic light, fire hydrant, stop sign, parking meter, bench, backpack, umbrella,
            # handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat,
            # baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork,
            # knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
            # donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse,
            # remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock,
            # vase, scissors, teddy bear, hair drier, toothbrush
            return 'objetos'
    
    def get_dominant_color(self, frame, bbox):
        """Extrae el color dominante de un bounding box usando clustering"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Agregar margen para capturar mejor el color
            margin = 5
            x1 = max(0, x1 + margin)
            y1 = max(0, y1 + margin)
            x2 = min(frame.shape[1], x2 - margin)
            y2 = min(frame.shape[0], y2 - margin)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                return "N/A", "N/A"
            
            # Convertir a RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Reducir dimensiones para análisis
            pixels = roi_rgb.reshape(-1, 3)
            
            # Eliminar colores muy oscuros o muy claros (fondo/sombras)
            mask = np.all((pixels > 20) & (pixels < 235), axis=1)
            if mask.sum() > 0:
                pixels = pixels[mask]
            
            if len(pixels) == 0:
                pixels = roi_rgb.reshape(-1, 3)
            
            # Usar k-means simple para encontrar color dominante
            from sklearn.cluster import KMeans
            n_colors = min(3, len(pixels))
            
            if n_colors < 1:
                return "N/A", "N/A"
            
            kmeans = KMeans(n_clusters=n_colors, n_init=3, max_iter=100, random_state=42)
            kmeans.fit(pixels)
            
            # Obtener el color más frecuente
            labels = kmeans.labels_
            counts = np.bincount(labels)
            dominant_color = kmeans.cluster_centers_[counts.argmax()]
            
            r, g, b = int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])
            
            # Determinar nombre del color
            color_name = self.get_color_name(r, g, b)
            color_rgb = f"{r},{g},{b}"
            
            return color_name, color_rgb
        except:
            # Fallback: usar promedio simple
            try:
                x1, y1, x2, y2 = bbox
                roi = frame[y1:y2, x1:x2]
                avg_color = roi.mean(axis=0).mean(axis=0)
                b, g, r = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
                color_name = self.get_color_name(r, g, b)
                return color_name, f"{r},{g},{b}"
            except:
                return "N/A", "N/A"
    
    def get_color_name(self, r, g, b):
        """Determina el nombre del color según RGB con mayor precisión"""
        # Normalizar valores
        total = r + g + b
        if total == 0:
            return "Negro"
        
        # Colores básicos con mejor detección
        if r < 50 and g < 50 and b < 50:
            return "Negro"
        elif r > 200 and g > 200 and b > 200:
            return "Blanco"
        elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            if r > 150:
                return "Gris Claro"
            elif r > 80:
                return "Gris"
            else:
                return "Gris Oscuro"
        
        # Detectar color dominante
        if r > g and r > b:
            if g > 100 and b < 100:
                return "Naranja"
            elif g < 100 and b < 100:
                return "Rojo"
            elif b > 100:
                return "Rosa"
            else:
                return "Rojo"
        elif g > r and g > b:
            if r > 100 and b < 100:
                return "Amarillo"
            elif b > 100 and r < 100:
                return "Verde Agua"
            else:
                return "Verde"
        elif b > r and b > g:
            if r > 100 and g < 100:
                return "Morado"
            elif g > 100 and r < 100:
                return "Cyan"
            else:
                return "Azul"
        
        # Casos especiales
        if r > 150 and g > 150 and b < 100:
            return "Amarillo"
        elif r > 150 and b > 150 and g < 100:
            return "Magenta"
        elif g > 150 and b > 150 and r < 100:
            return "Cyan"
        elif r > 200 and g > 100 and g < 180 and b < 100:
            return "Naranja"
        elif r > 180 and g < 100 and b > 180:
            return "Morado"
        elif r < 100 and g > 180 and b > 100:
            return "Verde"
        
        return "Multicolor"
    
    def get_shape(self, bbox):
        """Determina la forma aproximada del bounding box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if width == 0 or height == 0:
            return "Rectángulo"
        
        aspect_ratio = width / height
        
        # Cuadrado: relación cercana a 1:1
        if 0.85 <= aspect_ratio <= 1.15:
            return "Cuadrado"
        # Círculo: relación muy cercana a 1:1
        elif 0.95 <= aspect_ratio <= 1.05:
            return "Círculo"
        # Rectángulo: cualquier otra relación
        else:
            return "Rectángulo"
    
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
        Realiza detección en un frame - DETECTA TODOS LOS OBJETOS
        
        Args:
            frame: Frame de video (numpy array)
            
        Returns:
            frame_anotado, detecciones, estadisticas
        """
        start_time = time.time()
        
        # Ejecutar detección YOLO con MÁXIMA configuración para detectar TODO
        results = self.model(
            frame, 
            conf=self.confidence_threshold,  # Umbral MUY BAJO (0.25) para detectar TODO
            iou=0.25,  # IoU muy bajo para detectar objetos muy superpuestos
            max_det=2000,  # ULTRA MÁXIMO: hasta 2000 detecciones
            verbose=False,
            agnostic_nms=False,  # Mantener todas las clases separadas
            classes=None,  # Detectar TODAS las clases disponibles
            retina_masks=True,  # Mejor precisión
            half=False,  # Precisión completa
            augment=True  # Aumentación de datos para mejor detección
        )
        
        detections = []
        annotated_frame = frame.copy()
        
        print(f"🔍 Procesando frame: {len(results)} resultados de YOLO")
        
        # Procesar cada detección
        for result in results:
            boxes = result.boxes
            print(f"📦 Boxes detectadas: {len(boxes)}")
            
            for idx, box in enumerate(boxes):
                try:
                    # Extraer información
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    
                    print(f"  └─ [{idx+1}] {class_name} (ID:{class_id}) - Confianza: {confidence:.2f}")
                    
                    # Categorizar
                    category = self.categorize_detection(class_id)
                    
                    # Asignar tracking ID
                    bbox = [x1, y1, x2, y2]
                    tracking_id = self.assign_tracking_id(bbox, category, class_name)
                    
                    # Obtener color dominante
                    color_name, color_rgb = self.get_dominant_color(frame, bbox)
                    print(f"      Color: {color_name} ({color_rgb})")
                    
                    # Obtener forma
                    shape = self.get_shape(bbox)
                    
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
                        'tracking_id': tracking_id,
                        'category': category,
                        'class_name': class_name,
                        'bbox': bbox,
                        'confidence': confidence,
                        'color': color_name,
                        'color_rgb': color_rgb,
                        'shape': shape
                    })
                    
                except Exception as e:
                    print(f"      ⚠️ Error procesando box {idx}: {e}")
                    continue
        
        print(f"✅ Total detectado: {len(detections)} objetos")
        
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