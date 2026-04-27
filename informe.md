# Informe Final: Sistema de Reconocimiento Facial (TP1)

## 1. Arquitectura General y Flujo de Trabajo

El sistema está construido como una aplicación de microservicios desplegada en **Docker**, compuesta por:

| Servicio     | Tecnología           | Puerto | Función                                      |
|-------------|----------------------|--------|----------------------------------------------|
| `backend`   | FastAPI + PyTorch    | 8000   | API REST: detección, registro y predicción   |
| `frontend`  | Gradio               | 8080   | Interfaz gráfica de usuario                  |
| `postgres`  | PostgreSQL + pgvector| 5432   | Base de datos vectorial de embeddings        |
| `jupyter`   | JupyterLab           | 8888   | Entorno de experimentación                   |

### Pipeline de Reconocimiento (face_service.py)

El pipeline sigue 4 etapas claramente separadas dentro del código:

1. **Detección** (`detect_faces`): Utiliza **RetinaFace** (vía InsightFace `buffalo_s`) para encontrar bounding boxes y 5 keypoints faciales (ojos, nariz, comisuras de la boca).
2. **Alineación** (`align_face`): Aplica una **transformación afín** (`norm_crop`) usando los keypoints para rotar, escalar y centrar cada rostro a un tamaño estándar de 112×112 píxeles.
3. **Embeddings** (`extract_embedding_from_face`): El rostro alineado pasa por un modelo convolucional (ResNet-18 fine-tuned) que produce un **vector de 512 dimensiones**, normalizado con L2.
4. **Comparación** (`identify`): Los embeddings de la consulta se comparan contra la base de datos PostgreSQL usando **Similitud Coseno**. Si el score supera el umbral configurado, se asigna la identidad; caso contrario, se marca como `unknown`.

## 2. Dataset

Se construyó un dataset propio con reglas específicas para evaluar robustez:

| Persona     | Imágenes | Regla                                          |
|-------------|----------|-------------------------------------------------|
| Ambrogi     | 4        | Todas las disponibles (prueba de desbalance)    |
| Gianfranco  | 10       | Cap estándar                                    |
| Gianluca    | 10       | Cap estándar                                    |
| Lucas       | 10       | Cap estándar                                    |
| Matías      | 10       | Cap estándar                                    |
| Roberto     | 10       | Cap estándar                                    |
| **Valentino** | **0 (excluido)** | **Prueba fuera de distribución (unknown)** |

**Valentino** es hermano genético de Gianluca. Al excluirlo del entrenamiento, podemos evaluar empíricamente si el modelo capta rasgos familiares compartidos o lo clasifica como desconocido.

## 3. Entrenamiento (Fine-Tuning)

### Configuración

| Parámetro       | Valor            |
|-----------------|------------------|
| Backbone        | ResNet-18 (timm) |
| Pre-entrenamiento | ImageNet        |
| Embedding size  | 512              |
| Épocas          | 50               |
| Batch size      | 8                |
| Learning rate   | 1e-4 (Adam)      |
| Loss            | CrossEntropyLoss |
| Data augmentation | HorizontalFlip + ColorJitter |
| Dispositivo     | CPU (Docker)     |

### Resultados

El modelo alcanzó **100% de accuracy** con una loss de **0.0438** en la época 50.

La convergencia fue progresiva:
- Época 10: 70.37% accuracy
- Época 20: 88.89% accuracy
- Época 30: 100.00% accuracy (primera vez)
- Época 50: 100.00% accuracy (loss estabilizada en ~0.04)

Las métricas completas (por época) están disponibles en `output/training_metrics.json`.

### Exportación del Modelo

El modelo final se exporta como `nn.Sequential(backbone, embedding_head)`, lo que permite cargarlo directamente en la API sin dependencias del script de entrenamiento.

## 4. Evaluación y Visualización

### PCA y t-SNE

Se generaron gráficos de reducción de dimensionalidad aplicados a los embeddings de **todas** las personas del dataset, incluyendo a Valentino:

- `output/pca.png` — Proyección PCA a 2 componentes principales
- `output/tsne.png` — Proyección t-SNE con Valentino marcado con ★

### Análisis de Valentino (vecino más cercano)

| Imagen de Valentino | Vecino más cercano | Similitud Coseno |
|--------------------|--------------------|------------------|
| valentino_01.png   | Roberto            | 0.8633           |
| valentino_02.png   | Matías             | 0.8918           |

El modelo no lo confunde directamente con su hermano Gianluca, lo que indica buena capacidad de discriminación. Los valores de similitud (0.86–0.89) están cercanos al umbral de reconocimiento, lo que es coherente con el hecho de tratarse de un sujeto desconocido para el sistema.

El reporte completo de evaluación está en `output/evaluation_report.json`.

## 5. Base de Datos (Seed)

Se pre-cargaron **2 imágenes por persona** (excepto Valentino) en PostgreSQL mediante el script `seed_db.py`, totalizando **12 registros** en la tabla `embeddings`.

## 6. Archivos Generados

| Archivo                          | Descripción                                       |
|----------------------------------|---------------------------------------------------|
| `models/face_detection.pth`      | Modelo PyTorch entrenado (ResNet-18, 512-dim)     |
| `output/training_metrics.json`   | Métricas de entrenamiento (loss/acc por época)    |
| `output/evaluation_report.json`  | Similitud intra-clase + vecinos de Valentino      |
| `output/pca.png`                 | Gráfico PCA de embeddings                         |
| `output/tsne.png`                | Gráfico t-SNE de embeddings                       |
| `train.py`                       | Script de entrenamiento                           |
| `evaluate.py`                    | Script de evaluación (PCA/t-SNE/métricas)         |
| `seed_db.py`                     | Script de carga inicial de la base de datos       |
| `Dataset/dataset.md`             | Documentación del dataset                         |
| `informe.md` / `informe.html`   | Este informe en ambos formatos                    |
