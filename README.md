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
git clone https://github.com/reimardelgado/rag-reranker.git
cd rag-reranker
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

### 5.1 Gate de acceso para imágenes contextualizadas

> **Decisión actual: NO-GO.** La generación de imágenes contextualizadas debe
> permanecer deshabilitada (`GROUNDED_IMAGE_ENABLED=false`). El backend de
> producción todavía usa la ruta de texto y la captura redacted de Open WebUI
> v0.11.4 no demostró el envío de identidad confiable de usuario/chat/mensaje.
> La evidencia administrativa sin valores privados está en
> `backend/tests/fixtures/openwebui/v0.11.4-admin-access-decision.json`.

Open WebUI conserva la responsabilidad sobre RBAC, capacidades, credenciales
del motor de imágenes, archivos y UI. Una futura habilitación se limita a un
**canary explícitamente autorizado** y requiere que un administrador verifique
todos estos controles sin exportar identificadores, tokens, prompts ni claves:

1. **RBAC de mínimo privilegio:** usar un usuario canary no administrador con el
   permiso de generación de imágenes; mantener denegado el permiso global por
   defecto para los demás usuarios.
2. **Capacidades del modelo:** autorizar solamente el modelo canary y comprobar
   `builtin_tools` e `image_generation` de forma explícita. No depender de
   valores implícitos para aprobar el rollout.
3. **Herramientas nativas:** seleccionar el modo Native y comprobar, mediante
   una captura no-forward con redacción, que `generate_image` aparece una sola
   vez en el request dirigido al backend.
4. **Motor y toggle:** configurar el motor/modelo de imagen en Open WebUI, pero
   activar el toggle de Image Generation únicamente en el chat canary. La
   presencia de una credencial se registra sólo como booleano; su valor nunca
   se exporta.
5. **Frontera autenticada:** entregar por una conexión privada los headers de
   usuario, chat y mensaje raíz junto con el token de servicio independiente.
   Los nombres esperados son `X-OpenWebUI-User-Id`,
   `X-OpenWebUI-Chat-Id`, `X-OpenWebUI-Message-Id` y
   `X-RAG-Service-Token`; sus valores no deben aparecer en logs o exports.
6. **Allowlist y réplica:** hacer coincidir usuario y modelo con las allowlists
   del backend y mantener exactamente una réplica. El modo multi-réplica sigue
   siendo **NO-GO** hasta aprobar el spike Redis distribuido.
7. **Provenance y rollback:** exigir provenance adyacente antes de habilitar.
   Ante cualquier gate ausente, desactivar `GROUNDED_IMAGE_ENABLED`, retirar el
   toggle/capacidad canary y conservar el chat de texto.

La configuración se aprueba sólo cuando un export administrativo redacted
demuestra todos los gates con `authorized_canary_only=true`. El export debe
contener únicamente versión, booleanos, conteos, nombres de controles y estado
GO/NO-GO. No puede contener IDs de usuario/chat/modelo, auth, credenciales,
prompts, evidencia ni firmas privadas del proveedor. Configurar el motor o
observar `generate_image` **no** equivale a aprobar generación paga.

#### Contrato de configuración y puesta en marcha segura

La plantilla `.env.example` es normativa y **no es una receta de activación**.
Los booleanos aceptan exclusivamente `true` o `false`; valores ambiguos impiden
iniciar la configuración. Las allowlists son listas separadas por comas, sin
espacios, entradas vacías ni duplicados. Con el modo deshabilitado, mantené
vacías las allowlists y las dos credenciales de correlación.

Antes de una futura habilitación, además de aprobar el gate anterior, se deben
cumplir simultáneamente estas invariantes:

- `GROUNDED_IMAGE_TRUSTED_BOUNDARY=true`,
  `GROUNDED_IMAGE_PROVENANCE_ENABLED=true` y adaptador
  `GROUNDED_IMAGE_ADAPTER=gemini_native`.
- Allowlist no vacía para el único usuario canary y el único modelo canary.
- `RAG_TRUSTED_SERVICE_TOKEN` independiente de 32-512 bytes y
  `RAG_CORRELATION_HMAC_KEY` independiente de 32-128 bytes aleatorios en
  base64url. Ambos se inyectan desde un gestor de secretos; nunca se copian a
  Git, tickets, comandos compartidos, métricas, logs o exports.
- Mínimo de dos referencias distintas; TTL no deslizante de 600 segundos;
  paquete máximo de 262144 bytes; identificadores de hasta 1024 bytes; máximo
  exacto de tres iteraciones de herramientas; exactamente una réplica.

Sólo después de comprobar esas precondiciones de manera redacted se puede
considerar cambiar `GROUNDED_IMAGE_ENABLED=true`. Aun así, el estado actual del
repositorio es **inactivo**: `backend/app.py` no compone grounding, correlación,
enforcement ni generación. Por lo tanto, editar variables por sí solo NO
habilita ni autoriza una llamada paga.

#### Contrato de chat y errores recuperables

El contrato versionado está en `backend/rag-backend-openapi.yaml`. El endpoint
OpenAI-compatible es `POST /v1/chat/completions` (con alias
`/chat/completions`) y conserva mensajes de sistema/usuario/asistente/tool,
`tools`, `tool_choice`, campos soportados adicionales y partes multimodales.
La recuperación deriva texto exclusivamente del último turno de usuario sin
mutar el payload. Un turno sólo con adjuntos devuelve un error recuperable y
nunca dispara generación sin evidencia.

Cuando el adaptador opcional está compuesto, los errores usan un envelope
OpenAI-compatible acotado, con `error.message`, `error.type`, `error.code` y un
`request_id` opaco. En SSE, un fallo posterior al inicio se entrega como evento
de error y termina una sola vez con `[DONE]`. Los códigos normativos son:

| HTTP | Código | Acción segura |
|------|--------|---------------|
| 401 | `AUTH_REQUIRED` | Rechazar identidad/token ausente o inválido; no recuperar ni llamar proveedores. |
| 403 | `FORBIDDEN` | Rechazar permiso, capability, toggle, usuario o modelo fuera de allowlist. |
| 409 | `IMAGE_DISABLED`, `UNSUPPORTED_TOOL` | Mantener el flujo de imagen cerrado; no degradar a generación libre. |
| 409 | `STATE_EXPIRED`, `LOOP_LIMIT`, `DUPLICATE_SUPPRESSED` | Detener la continuación o duplicado sin repetir recuperación ni llamada paga. |
| 422 | `NO_IMAGE_EVIDENCE`, `INVALID_GROUNDING`, `INVALID_TOOL_CALL` | Rechazar evidencia, prompt o tool call inválido antes de emitir `generate_image`. |
| 429 | `RATE_LIMITED` | Propagar throttling recuperable sin reintento automático de generación paga. |
| 502 | `PROVIDER_UNAVAILABLE`, `PROVIDER_INCOMPATIBLE` | Fallar cerrado ante 5xx o protocolo no conforme. |
| 504 | `PROVIDER_TIMEOUT` | Cancelar/cerrar el iterador sin convertir el fallo en éxito. |

Estos códigos documentan el límite de protocolo; no afirman que el flujo de
imagen esté activo. Los errores legacy de texto pueden conservar el envelope
FastAPI `detail` hasta que se componga explícitamente el adaptador aprobado.

#### Seguridad, redacción y rollback

- La evidencia se trata como datos JSON no confiables y acotados, nunca como
  política, configuración de herramientas o instrucciones ejecutables.
- No se registran ni persisten tokens, claves HMAC, valores de headers
  confiables, IDs crudos, prompts, chunks de evidencia, resultados de tools,
  datos de usuario ni firmas privadas del proveedor.
- La provenance pública contiene sólo hash/versión del prompt, referencias
  fuente/página, elementos documentados/inferidos/inciertos/prohibidos, IDs
  públicos de modelos y timestamp UTC. Debe aparecer junto a
  **“Interpretive reconstruction”** y la advertencia de que los píxeles
  generados no son evidencia documental.
- La correlación usa HMAC sobre la tupla confiable usuario/chat/mensaje raíz,
  caché LRU/TTL acotada e idempotencia; nunca confía en IDs del body.
- El proceso local no garantiza atomicidad entre réplicas. El despliegue de
  imágenes con más de una réplica permanece **NO-GO** hasta que pase el spike
  Redis de dos workers definido en `docs/redis-correlation-decision.md`.

**Rollback inmediato:** establecer `GROUNDED_IMAGE_ENABLED=false`, retirar el
toggle/capability del chat canary, vaciar las allowlists y revocar las dos
credenciales de correlación. El chat de texto debe seguir disponible. No se
requiere borrar ni modificar credenciales del motor de imágenes administradas
por Open WebUI para que el backend vuelva al modo text-only.

#### Observabilidad, shadow y SLOs del canary

`backend/metrics.py` define el contrato de observabilidad **inactivo** y de
cardinalidad fija. `backend/app.py` no lo compone: agregar métricas no habilita
grounding, tools ni generación. El snapshot de dashboard contiene únicamente
contadores agregados por etapa/resultado y buckets de latencia; no acepta ni
expone IDs crudos, valores de headers confiables, auth, tokens, claves HMAC,
prompts, evidencia, datos de usuario, resultados de tools o firmas privadas.

El dashboard de una futura ventana debe mostrar, como mínimo:

- requests de texto e intent de imagen; decisiones eligible/refused/denied/invalid;
- fallos de validación, correlación, expiración, aislamiento, loop e idempotencia;
- duplicados suprimidos y cualquier violación de llamada paga duplicada;
- outcomes acotados del proveedor: success/timeout/rate-limited/unavailable/incompatible;
- p95 por buckets de texto total, retrieval, rerank, grounding warm,
  continuación cacheada, primer byte, primer tool call y ejecución de imagen;
- eventos agregados del kill switch y disponibilidad de texto durante rollback.

**Shadow obligatorio y sin costo.** Mantener `GROUNDED_IMAGE_ENABLED=false` y
evaluar localmente el intent, recuperación y validación sin emitir tools, sin
contactar al proveedor y sin invocar el motor de imágenes. Una ventana es
inspeccionable recién con al menos 100 intents de imagen y 1000 requests de texto,
cero llamadas al proveedor o motor de imágenes, cero intentos pagos,
cero fallos de paridad de texto y cero violaciones de aislamiento. Cualquier
incumplimiento invalida la ventana; no se reetiqueta como éxito.

Los SLOs normativos para un canary futuro son:

- cero fugas entre usuarios, cero llamadas pagas duplicadas y cero retrievals
  repetidos dentro de la misma continuación cacheada;
- 100% de grounding inválido bloqueado antes de emitir `generate_image`;
- regresión p95 de texto <= 200 ms **o** <= 10% respecto del baseline redacted;
- p95 de grounding warm <= 3000 ms;
- p95 de continuación cacheada <= 100 ms, con cero retrieval adicional;
- p95 del primer tool call <= 5000 ms, excluyendo la generación de imagen.

Una evaluación canary exige, además, al menos 20 resultados de imagen aprobados
y 500 requests de texto en la ventana, RBAC/capability/toggle y allowlists
aprobados, identidad y servicio confiables probados, provenance adyacente,
adaptador nativo, composición productiva revisada, una sola réplica (o spike
Redis aprobado), smoke pago explícitamente autorizado y drill de rollback
aprobado. Los conteos son mínimos de evidencia, no autorización para ejecutar.

**Drill de rollback y kill switch:** antes de cualquier canary, registrar en el
dashboard el request de rollback, poner `GROUNDED_IMAGE_ENABLED=false`, retirar
toggle/capability y allowlists canary, revocar las credenciales de correlación y
comprobar que (1) nuevos intents de imagen no llegan a retrieval/proveedor, (2)
no aumenta ningún contador de generación paga y (3) el chat de texto sigue
respondiendo dentro de su SLO. Si falta una de esas tres pruebas, el drill falla
y el gate queda NO-GO. No se guardan requests ni valores privados como evidencia.

**Resultado actual: NO-GO operativo.** No existen ventanas shadow/canary ni drill
de rollback observados; el acceso confiable continúa fallando, la composición
productiva está inactiva, paid generation no está aprobada y multi-réplica sigue
bloqueado por Redis. Conservar text-only. **No ejecutar el smoke pago de 12.3**
hasta una aprobación separada y explícita.

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
rag-reranker/
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
