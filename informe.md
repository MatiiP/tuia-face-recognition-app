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
3. **Embeddings** (`extract_embedding_from_face`): El rostro alineado pasa por un modelo convolucional (ResNet-50 fine-tuned) que produce un **vector de 512 dimensiones**, normalizado con L2.
4. **Comparación** (`identify`): Los embeddings de la consulta se comparan contra la base de datos PostgreSQL usando **Similitud Coseno**. Si el score supera el umbral configurado (actualmente ajustado a 0.90 por la alta agrupación geométrica), se asigna la identidad; caso contrario, se marca como `unknown`.

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
| Backbone        | ResNet-50 (timm) |
| Pre-entrenamiento | ImageNet        |
| Embedding size  | 512              |
| Épocas          | 50               |
| Batch size      | 8                |
| Learning rate   | 1e-4 (Adam)      |
| Loss            | CrossEntropyLoss |
| Data augmentation | HorizontalFlip + ColorJitter |
| Dispositivo     | CPU (Docker)     |

### Resultados

El modelo alcanzó **~99.53% de accuracy en Train** y **~98.16% en Validación** con una loss muy baja en la época 50.

La convergencia demostró el poder de ResNet-50:
- Época 10: 75.35% Train | 82.35% Val
- Época 30: 97.32% Train | 97.43% Val
- Época 50: 99.53% Train | 98.16% Val

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
| valentino_01.png   | Gianluca           | ~0.9508          |

**Análisis Crítico:** Al utilizar ResNet-50, el modelo logró extraer características faciales muy profundas, agrupando los rostros con gran precisión. Sin embargo, al probar con la imagen excluida de Valentino (hermano de Gianluca), el modelo arrojó una similitud extrema (0.9508). Esto indica un "sobreajuste genético" o falso positivo debido al alto parecido familiar. Para mitigar esto en producción, el umbral (`SIMILARITY_THRESHOLD`) se elevó a `0.90` / `0.96` o se requerirían algoritmos de margen angular más estrictos (ArcFace).

El reporte completo de evaluación está en `output/evaluation_report.json`.

## 5. Base de Datos (Seed)

Se pre-cargaron **2 imágenes por persona** (excepto Valentino) en PostgreSQL mediante el script `seed_db.py`, totalizando **12 registros** en la tabla `embeddings`.

## 6. Archivos Generados

| Archivo                          | Descripción                                       |
|----------------------------------|---------------------------------------------------|
| `models/face_detection.pth`      | Modelo PyTorch entrenado (ResNet-50, 512-dim)     |
| `output/training_metrics.json`   | Métricas de entrenamiento (loss/acc por época)    |
| `output/evaluation_report.json`  | Similitud intra-clase + vecinos de Valentino      |
| `output/pca.png`                 | Gráfico PCA de embeddings                         |
| `output/tsne.png`                | Gráfico t-SNE de embeddings                       |
| `train.ipynb`                    | Notebook de entrenamiento (Pipeline completo)     |
| `evaluate.py`                    | Script de evaluación (PCA/t-SNE/métricas)         |
| `seed_db.py`                     | Script de carga inicial de la base de datos       |
| `Dataset/dataset.md`             | Documentación del dataset                         |
| `informe.md` / `informe.html`   | Este informe en ambos formatos                    |
