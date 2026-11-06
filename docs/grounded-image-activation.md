# Runbook: activar la generación de imágenes contextualizadas

**Rama:** `feat/grounded-image-runtime` · **Análisis:** [grounded-image-runtime-analysis.md](grounded-image-runtime-analysis.md)

El orden es obligatorio: **0 → 1 → 2 → 3 → 4**. Cada fase tiene una salida segura,
que es volver al texto. Ninguna fase anterior a la 3 puede generar una imagen ni
consumir el motor de imagen.

## Cómo funciona (resumen)

1. El usuario pide en Open WebUI, con el toggle **Image Generation** activo, algo como
   *"Genera una imagen del oppidum ibero"*.
2. Open WebUI envía el chat y la herramienta `generate_image` al backend.
3. El backend comprueba acceso, recupera evidencia (mínimo 2 fuentes/página
   distintas), pide a Gemini una instrucción estructurada, la valida y le hace un
   hash, y devuelve **una** llamada `generate_image` cuyo `prompt` es la
   instrucción canónica validada.
4. **Open WebUI ejecuta su motor de imagen** con sus propias credenciales y
   muestra la imagen. Este es el único paso pago.
5. Open WebUI devuelve el resultado de la herramienta. El backend responde con un
   pie de imagen, la etiqueta **"Interpretive reconstruction"**, la advertencia
   *"Generated pixels are not documentary evidence."* y el sidecar de procedencia
   (fuentes y páginas, hechos documentados, inferidos e inciertos, prohibidos,
   modelos y hash).

Si cualquier paso falla, el backend devuelve un error acotado (`NO_IMAGE_EVIDENCE`,
`INVALID_GROUNDING`, `UNSUPPORTED_TOOL`, `FORBIDDEN`, `AUTH_REQUIRED`,
`STATE_EXPIRED`, `DUPLICATE_SUPPRESSED`, etc.) y **nunca** emite una imagen libre.

## Fase 0 — Preparación (sin costo)

1. Genera los dos secretos. No los pegues en Git, tickets ni chats:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"   # RAG_TRUSTED_SERVICE_TOKEN
   python -c "import secrets,base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('='))"  # RAG_CORRELATION_HMAC_KEY
   ```
2. Ponlos en `.env`, que está ignorado por Git.
3. En `.env`, deja todavía `GROUNDED_IMAGE_ENABLED=false`.
4. Ejecuta los tests offline:
   `pip install -r backend/requirements-dev.txt && pytest -q`. Deben pasar
   todos; el smoke `paid` queda *deselected*.

## Fase 1 — Identidad confiable desde Open WebUI (sin costo)

1. En `.env`, pon `OPENWEBUI_FORWARD_USER_INFO_HEADERS=true`. Compose lo traduce a
   `ENABLE_FORWARD_USER_INFO_HEADERS` en Open WebUI.
2. Compose pasa `OPENAI_API_KEYS=${RAG_TRUSTED_SERVICE_TOKEN}`. **Open WebUI
   guarda las conexiones en su base de datos**, así que comprueba también en
   *Admin → Settings → Connections* que la conexión `http://rag-backend:8088/v1`
   usa como API key el mismo valor que `RAG_TRUSTED_SERVICE_TOKEN`.
3. Recrea Open WebUI: `docker compose up -d open-webui`.
4. Verifica qué encabezados llegan, sin registrar valores. El backend sólo usa
   `X-OpenWebUI-User-Id`, `X-OpenWebUI-Chat-Id`, `X-OpenWebUI-Message-Id` y el
   bearer.
   - Si **no** llega `X-OpenWebUI-Message-Id`, usa
     `GROUNDED_IMAGE_CORRELATION_SCOPE=chat`.

## Fase 2 — Shadow (sin costo, obligatorio)

1. En `.env`: `GROUNDED_IMAGE_SHADOW=true` y `GROUNDED_IMAGE_ENABLED=false`.
2. `docker compose up -d backend`.
3. Usa el chat con normalidad. Las peticiones de imagen se responden por la ruta
   de texto, pero se mide la intención, la recuperación y si hay evidencia
   suficiente. **No se llama al proveedor ni se emite ninguna herramienta.**
4. Consulta `GET http://localhost:8088/v1/grounded-image/metrics`.
   `shadow_gate.outcome` pasa a `GO` cuando hay al menos 100 intenciones de
   imagen y 1000 de texto, con cero llamadas pagas o al proveedor.
5. Si `decisions.refused` es alto, el corpus no tiene suficientes fuentes para
   esas peticiones. Revísalo antes de seguir.
6. Comprueba que `GET /bm25_status` devuelve `"ready": true`. El índice léxico se
   construye solo al arrancar y tras cada `/ingest`; sin él, la recuperación es solo
   densa y muchas peticiones legítimas se rechazan.

**Puerta de evidencia.** Un fragmento solo cuenta como evidencia si menciona al menos
la mitad de los términos del tema (`GROUNDED_IMAGE_MIN_SUBJECT_COVERAGE=0.5`), y cada
término debe aparecer en alguna evidencia. Así, "astronauta en Marte" o "Torre Eiffel"
se rechazan aunque la recuperación devuelva fragmentos del corpus. Las palabras de
estilo (*realista*, *acuarela*…) no cuentan como tema. Límite conocido: peticiones
con muchos términos (p. ej. "muralla del oppidum de Puente Tablas") pueden
rechazarse si ningún fragmento recuperado nombra todos; es un falso negativo
deliberado.

## Fase 3 — Canary (requiere tu aprobación explícita: consume el motor de imagen)

1. En Open WebUI:
   - Configura el motor y el modelo de imagen.
   - Crea un usuario canary **no administrador** con permiso de generación de imágenes.
   - Quita ese permiso del rol por defecto.
   - Habilita las capacidades `image_generation` y `builtin_tools` solo en el modelo canary.
   - Usa function calling en modo **Native**.
2. En `.env`:
   ```env
   GROUNDED_IMAGE_SHADOW=false
   GROUNDED_IMAGE_ENABLED=true
   GROUNDED_IMAGE_TRUSTED_BOUNDARY=true
   GROUNDED_IMAGE_PROVENANCE_ENABLED=true
   GROUNDED_IMAGE_AUTHORIZED_USERS=<id-del-usuario-canary>
   GROUNDED_IMAGE_AUTHORIZED_MODELS=<id-del-modelo-en-open-webui>
   GROUNDED_IMAGE_IMAGE_MODEL_ID=<nombre-público-del-motor-de-imagen>
   ```
3. Haz `docker compose up -d backend` y revisa el log:
   `[grounded-image] composed (stage=canary)`.
4. Con el usuario canary y el toggle **Image Generation** activo en un chat,
   escribe: *"Genera una imagen de &lt;algo documentado&gt;"*.
5. Comprueba lo siguiente:
   - Aparece una sola imagen.
   - El pie lleva "Interpretive reconstruction" y la advertencia.
   - El sidecar cita fuentes y páginas reales.
   - Una petición de texto sigue respondiendo igual que antes.
   - En `/v1/grounded-image/metrics`, `paid_generation.attempted` coincide con
     las imágenes pedidas.
6. Opcional, también con aprobación: ejecuta el smoke en vivo sin motor de
   imagen. Consume tokens de Gemini:
   `RAG_PAID_SMOKE=1 RAG_SMOKE_BASE_URL=http://localhost:8088 RAG_SMOKE_USER=<id> RAG_TRUSTED_SERVICE_TOKEN=<token> pytest -m paid backend/tests/smoke -q`

## Fase 4 — Rollback (practicarlo antes de ampliar)

1. En `.env`, pon `GROUNDED_IMAGE_ENABLED=false` y vacía las allowlists. Después,
   `docker compose up -d backend`.
2. Retira el toggle o la capacidad del chat canary en Open WebUI.
3. Revoca los dos secretos y genera otros nuevos antes de volver a habilitar.
4. Verifica tres cosas:
   - Una petición de imagen ya no llega al proveedor.
   - No sube ningún contador pago.
   - El texto sigue respondiendo.

## Riesgos conocidos a verificar en vivo

| Riesgo | Síntoma | Acción |
|---|---|---|
| Open WebUI no envía `X-OpenWebUI-Message-Id` | `AUTH_REQUIRED` en cada petición de imagen | `GROUNDED_IMAGE_CORRELATION_SCOPE=chat` |
| La conexión guardada en la BD de Open WebUI no usa el token | `AUTH_REQUIRED` | Corregir la API key en *Admin → Connections* |
| Toggle de imagen apagado | `UNSUPPORTED_TOOL` | Activar *Image Generation* en el chat |
| Corpus con una sola fuente para el tema | `NO_IMAGE_EVIDENCE` | Ingestar más documentos o reformular |
| BM25 no construido (`/bm25_status` → `ready: false`) | `NO_IMAGE_EVIDENCE` en temas documentados | `curl -X POST http://localhost:8088/rebuild_bm25` y revisar el log `[BM25]` |
| Open WebUI genera la imagen por su cuenta (toggle activo en modo *Default*) | Imagen sin pie ni procedencia | Function calling **Native**; desactivar *Admin → Images* fuera del canary |
| Gemini rechaza el esquema de `generate_image` de Open WebUI | `PROVIDER_INCOMPATIBLE` o 4xx | Revisar el esquema; ajustar el adaptador |
| Mensaje `role=tool` de Open WebUI con forma distinta | `INVALID_TOOL_CALL` en la ronda 2 | Capturar la forma con redacción y ajustar |
| Más de una réplica | Duplicados o fallos de correlación | Mantener exactamente 1 (NO-GO Redis) |
