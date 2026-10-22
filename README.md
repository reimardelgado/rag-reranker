# RAG Híbrido con Docker — Proyecto de Máster

Sistema RAG (Retrieval-Augmented Generation) con búsqueda híbrida (densa + léxica), fusión RRF y reranking con cross-encoder, desplegado en Docker con interfaz Open WebUI.

> **Estado actual (2026-10-07):** Todas las dependencias Python y las imágenes Docker están fijadas a versiones verificadas. La persistencia de ChromaDB fue corregida y respaldada. El archivo `.env` requiere tanto `GEMINI_API_KEY` como `WEBUI_SECRET_KEY`. Consultar **[ACTUALIZACION_PROYECTO_2026.md](ACTUALIZACION_PROYECTO_2026.md)** para el registro completo de cambios, pasos de activación y backlog técnico.

---

## Arquitectura del sistema

- **Vectorización local** con Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Recuperación híbrida:**
  - Dense retrieval → ChromaDB
  - BM25 léxico → `rank-bm25`
  - Fusión → Reciprocal Rank Fusion (RRF)
- **Reranking final** con cross-encoder (`BAAI/bge-reranker-base`)
- **Generación de respuesta** con Gemini (API) u Ollama (local)
- **Interfaz gráfica** → Open WebUI
- **Ingesta PDF** con OCR opcional (Tesseract + PaddleOCR)

---

## Servicios Docker

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| `rag-backend` | 8088 | Backend FastAPI (RAG, ingesta, BM25, reranker) |
| `chroma` | 8000 | Base vectorial ChromaDB |
| `ollama` | 11434 | LLM local (opcional) |
| `open-webui` | 8081 | Interfaz gráfica de chat |

---

## Requisitos previos

- **Docker** v20+
- **Docker Compose** v2
- (Opcional) Cuenta en [Google AI Studio](https://aistudio.google.com) para usar Gemini

---

## 1. Clonar el repositorio

```bash
git clone https://github.com/Diomaris/rag-hybrid-reranker-project.git
cd rag-hybrid-reranker-project
```

---

## 2. Configurar variables de entorno

Crea el archivo `.env` en la raíz del proyecto copiando la plantilla y completando tu API key:

```bash
cp .env.example .env
```

En PowerShell también puedes copiarlo con `Copy-Item .env.example .env`.

Edita `.env` con tus valores:

```env
# Chroma
CHROMA_HOST=chroma
CHROMA_PORT=8000
COLLECTION=kb_docs

# Embeddings locales
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Gemini (endpoint OpenAI-compatible de Google)
GEM_BASE=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_API_KEY=TuApiKey

# Motor por defecto
DEFAULT_TOP_K=6
DEFAULT_PROVIDER=gemini
DEFAULT_GEMINI_MODEL=gemini-3.8-flash

# Watcher automático (activar si se quiere ingesta automática al añadir PDFs)
ENABLE_WATCHER=false
```

> **Importante:** el archivo `.env` está en `.gitignore` y nunca debe subirse al repositorio.

### Obtener API Key de Gemini

1. Ir a [https://aistudio.google.com](https://aistudio.google.com)
2. Iniciar sesión con una cuenta de Google
3. Hacer clic en **Get API Key** → **Create API Key**
4. Copiar la clave y pegarla en `GEMINI_API_KEY`

---

## 3. Construir e iniciar los servicios

```bash
docker compose build
docker compose up -d
```

> La primera vez puede tardar varios minutos mientras descarga imágenes y dependencias.

Verificar que todos los servicios estén corriendo:

```bash
docker compose ps
```

Detener todos los servicios:

```bash
docker compose down
```

---

## 4. Acceder al sistema

| Interfaz | URL |
|----------|-----|
| Open WebUI (chat) | http://localhost:8081 |
| Backend FastAPI (health check) | http://localhost:8088/health |
| Documentación API (Swagger) | http://localhost:8088/docs |

---

## 5. Configurar Open WebUI con Gemini

1. Abrir http://localhost:8081 e iniciar sesión como administrador
2. Ir a **Ajustes → Conexiones**
3. Añadir una nueva conexión OpenAI-compatible con:
   - **URL:** `https://generativelanguage.googleapis.com/v1beta/openai/`
   - **API Key:** tu `GEMINI_API_KEY`
4. Guardar

---

## 6. Colocar documentos para ingesta

Los PDFs deben colocarse en la carpeta `backend/docs/`:

```
backend/docs/
 ├── mi_documento.pdf
 ├── otro_archivo.pdf
```

---

## 7. Ingesta de documentos

### Ingesta manual (recomendada para la primera ejecución)

Identificar el nombre del contenedor backend:

```bash
docker compose ps
```

Ejecutar la ingesta:

```bash
docker exec -it rag-backend python ingest_ocr_batch.py \
  /app/docs/mi_documento.pdf \
  --chroma-host chroma \
  --chroma-port 8000 \
  --collection kb_docs
```

Con OCR activado (para PDFs escaneados):

```bash
docker exec -it rag-backend python ingest_ocr_batch.py \
  /app/docs/mi_documento.pdf \
  --chroma-host chroma \
  --chroma-port 8000 \
  --collection kb_docs \
  --enable-ocr \
  --ocr-lang spa+eng
```

### Ingesta automática (Watcher)

El sistema incluye un watcher que vigila `backend/docs/` y lanza la ingesta automáticamente al detectar nuevos PDFs. Para activarlo, establece en `.env`:

```env
ENABLE_WATCHER=true
```

Y reinicia el backend:

```bash
docker compose restart backend
```

---

## 8. Verificar la ingesta

Contar chunks totales en ChromaDB:

```bash
docker exec -it rag-backend python stats_overall.py
```

Ver chunks por fuente:

```bash
docker exec -it rag-backend python stats_by_source.py mi_documento.pdf
```

Listar todas las fuentes:

```bash
docker exec -it rag-backend python list_sources.py
```

Reconstruir índice BM25 tras ingestar nuevos documentos:

```bash
curl -X POST http://localhost:8088/rebuild_bm25
```

---

## 9. Probar el RAG por API

Consulta directa al endpoint `/answer`:

```bash
curl -X POST http://localhost:8088/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explica los enfoques teóricos de la Arqueología según el texto",
    "top_k": 6,
    "model": "gemini-3.8-flash"
  }'
```

Recuperación híbrida (sin generación):

```bash
curl -X POST http://localhost:8088/retrieve_hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "técnicas de excavación", "top_k": 6}'
```

---

## 10. Endpoints principales del backend

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Estado del servicio |
| POST | `/ingest` | Subir e ingestar un PDF |
| POST | `/retrieve` | Recuperación densa (ChromaDB) |
| POST | `/retrieve_lexical` | Recuperación léxica (BM25) |
| POST | `/retrieve_hybrid` | Recuperación híbrida (RRF + reranker) |
| POST | `/answer` | RAG completo con Gemini |
| POST | `/rebuild_bm25` | Reconstruir índice BM25 |
| GET | `/bm25_status` | Estado del índice BM25 |

---

## 11. Modelos locales con Ollama (despliegue local/VPS)

Descargar un modelo:

```bash
docker exec -it ollama ollama pull llama3.1:8b
```

Listar modelos instalados:

```bash
docker exec -it ollama ollama list
```

En Open WebUI, selecciona el modelo descargado desde el selector superior del chat.

> **Nota sobre Codespaces:** en GitHub Codespaces se recomienda usar Gemini como LLM principal debido a las limitaciones de almacenamiento. Ollama es recomendado para despliegues locales o en VPS.

---

## 12. Estructura del proyecto

```
rag-hybrid-reranker-project/
├── backend/
│   ├── app.py                  # Backend FastAPI principal
│   ├── ingest_ocr_batch.py     # Ingesta PDF con OCR avanzado
│   ├── ingest_batch.py         # Ingesta PDF sin OCR
│   ├── watcher.py              # Watcher automático de docs
│   ├── watch_and_ingest.py     # Watcher con estado persistente
│   ├── list_sources.py         # Listar fuentes en ChromaDB
│   ├── stats_overall.py        # Estadísticas globales
│   ├── stats_by_source.py      # Estadísticas por fuente
│   ├── verify_chroma_chunks.py # Verificar chunks en ChromaDB
│   ├── query_chroma_sanity.py  # Consulta de sanity check
│   ├── chroma_delete_source.py # Eliminar fuente de ChromaDB
│   ├── config.py               # Configuración centralizada (Settings)
│   ├── requirements.txt        # Dependencias Python (fijadas)
│   ├── Dockerfile              # Imagen del backend
│   ├── .dockerignore           # Exclusiones del contexto de build
│   ├── rag-backend-openapi.yaml # Especificación OpenAPI (parcial)
│   └── docs/                   # Carpeta para documentos a ingestar
├── docker-compose.yml          # Orquestación de servicios
├── .env.example                # Plantilla de variables de entorno
├── .gitignore
├── ACTUALIZACION_PROYECTO_2026.md # Registro de actualización 2026-10-07
└── README.md
```

---

## 13. Solución de problemas frecuentes

**El backend no conecta con Chroma:**
Verifica que `CHROMA_HOST=chroma` en el `.env` (debe ser el nombre del servicio Docker, no `localhost`).

**Error `GEMINI_API_KEY` no encontrada:**
Asegúrate de que el archivo `.env` existe en la raíz del proyecto y contiene la variable. Reinicia el backend tras editarlo: `docker compose restart backend`.

**BM25 no disponible:**
Ejecuta `curl -X POST http://localhost:8088/rebuild_bm25` después de cada ingesta masiva.

**PDF sin texto extraído:**
Usa `--enable-ocr` en la ingesta para activar OCR con Tesseract.

---

## Licencia

Proyecto desarrollado como trabajo de prácticas para el Máster en Ingeniería Informática.
