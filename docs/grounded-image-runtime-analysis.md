# Reanálisis: qué falta para generar imágenes (no solo texto)

**Cambio SDD:** `grounded-image-runtime-composition` (engram, proyecto `rag-reranker`)
**Rama:** `feat/grounded-image-runtime`
**Fecha:** 2026-11-04
**Antecedente:** cambio `contextual-rag-image-generation`, tareas 1.1–12.2 completas (172 tests offline).

## 1. Situación de partida

El cambio anterior dejó **todas las piezas construidas pero sin conectar**:

| Pieza | Archivo | Estado |
|---|---|---|
| Intención explícita de imagen | `grounding/intent.py` | Lista, no usada |
| Paquete de evidencia y umbral (2 fuentes/página) | `grounding/evidence.py` | Lista, no usada |
| Instrucción estructurada + validación + hash | `grounding/instruction.py`, `validation.py` | Lista, no usada |
| Procedencia y etiqueta "Interpretive reconstruction" | `grounding/provenance.py` | Lista, no usada |
| Adaptador nativo Gemini (tools + thought signatures) | `providers/gemini_native.py` | Listo, no usado |
| Normalización OpenAI JSON/SSE | `providers/base.py` | Lista, inyectable |
| Enforcer de `generate_image` y continuación | `api/chat.py` | Listos, no instanciados |
| Correlación HMAC, TTL, idempotencia, 3 iteraciones | `correlation.py` | Lista, no usada |
| Configuración estricta y gate de acceso | `config.py` | Parseada, no aplicada |
| Métricas shadow/canary | `metrics.py` | Lista, no usada |

`backend/app.py` sigue componiendo solo la ruta de texto. Por eso **hoy es imposible generar una imagen**, aunque se active `GROUNDED_IMAGE_ENABLED`.

## 2. Brechas encontradas

### B1. No existe el orquestador (bloqueante)
Ninguna pieza encadena: acceso → intención → recuperación → instrucción → validación → llamada forzada → estado → continuación → procedencia. El diseño original lo describía en el flujo de datos, pero **ninguna tarea 1.1–12.3 lo implementaba**. Es la brecha principal.

### B2. Nadie construye la `GroundedImageInstruction` (bloqueante)
La validación exige hechos documentados con `reference_ids`, elementos inferidos, incertidumbres y adiciones prohibidas, pero no hay código que los produzca a partir de la evidencia. Hace falta un **constructor de instrucciones** que llame al modelo de texto con la evidencia como dato no confiable y le exija una respuesta estructurada (función forzada). Las referencias y la consulta las fija el código, no el modelo. Todo pasa después por la validación existente.

### B3. La recuperación devuelve texto plano, no evidencia estructurada (bloqueante)
`retrieve_chat_blocks` devuelve cadenas `"[fuente p.N]\ntexto"`. El gate necesita `EvidenceItem(source, page, content)` con IDs de referencia estables. Falta un adaptador bloque → `EvidencePacket`. La consulta también debe limpiarse de los verbos de imagen ("genera una imagen de…") antes de recuperar.

### B4. La continuación Gemini rechazaría el prompt reescrito (bug latente, bloqueante)
`ImageToolEnforcer` reemplaza el prompt del modelo por el prompt canónico validado (correcto por diseño). Pero `GeminiNativeAdapter` guarda los argumentos **originales** del modelo y, en la continuación, exige que Open WebUI devuelva exactamente esos argumentos. Open WebUI devuelve los argumentos reescritos → `ContinuationMismatch` → el segundo turno fallaría siempre. Hace falta que el adaptador acepte un re-enlace explícito de los argumentos validados (sin tocar la thought signature).

### B5. El endpoint no ve los encabezados HTTP (bloqueante)
`openai_chat_completions(req)` solo recibe el cuerpo. La correlación confiable necesita los encabezados `X-OpenWebUI-User-Id`, `X-OpenWebUI-Chat-Id`, `X-OpenWebUI-Message-Id` y el token de servicio.

### B6. Open WebUI no envía identidad ni token de servicio (bloqueante de configuración)
- La captura de la tarea 5.1 mostró que no llegan encabezados de identidad: en `docker-compose.yml` falta `ENABLE_FORWARD_USER_INFO_HEADERS=true` en Open WebUI.
- Open WebUI sí envía `Authorization: Bearer <OPENAI_API_KEYS>`, que hoy vale `dummy`. Ese es el canal natural del token de servicio: si se configura `OPENAI_API_KEYS=<RAG_TRUSTED_SERVICE_TOKEN>`, el backend puede autenticar a Open WebUI sin un encabezado personalizado.
- Que Open WebUI v0.11.4 reenvíe un ID de mensaje no está probado. Se añade un modo de correlación `chat`: usa usuario + chat + hash del último mensaje de usuario, que es estable dentro del mismo bucle de herramienta. El modo `message`, más estricto, queda como predeterminado.

### B7. Permiso, capacidad y toggle no son observables directamente
El gate pide tres booleanos de Open WebUI. Lo único observable en el cable es que Open WebUI **solo inyecta `generate_image` cuando el permiso, la capacidad del modelo y el toggle del chat están activos**. Se usa la presencia de exactamente una herramienta `generate_image` compatible como evidencia de los tres. Las allowlists de usuario y modelo del backend siguen siendo obligatorias.

### B8. Falta el modo shadow real
Las métricas definen la ventana shadow (≥100 intenciones de imagen, ≥1000 de texto, cero llamadas a proveedor/imagen), pero no hay forma de ejecutarla. Hace falta `GROUNDED_IMAGE_SHADOW=true`: evalúa intención, recuperación y suficiencia de evidencia, cuenta métricas, **no** llama al proveedor ni emite herramientas y responde por la ruta de texto de siempre.

### B9. Falta exponer métricas agregadas
Se necesita un endpoint de solo lectura con el snapshot agregado (sin dimensiones por usuario) para evaluar los gates shadow/canary. Solo existe si la función está compuesta.

### B10. Falta el smoke test de pago opt-in (tarea 12.3)
Se añade marcado `paid`, excluido por defecto, que solo corre con `RAG_PAID_SMOKE=1` y credenciales. **No se ejecuta en esta rama.**

### B11. Configuración adicional
- Modelo nativo de Gemini para el constructor y el turno forzado (`GROUNDED_IMAGE_CHAT_MODEL`).
- ID público del modelo de imagen para la procedencia (`GROUNDED_IMAGE_IMAGE_MODEL_ID`), que es el motor que Open WebUI tenga configurado.
- Modo de correlación (`GROUNDED_IMAGE_CORRELATION_SCOPE=message|chat`).

## 3. Flujo objetivo

```
Open WebUI ──(1) chat + tools[generate_image] + headers──▶ RAG backend
  backend: acceso (flag, allowlists, token, identidad, tool presente)
         → intención explícita → recuperación única (single-flight) → EvidencePacket
         → constructor de instrucción (Gemini, función forzada) → validación + hash
         → reserva de generación (idempotencia) → turno forzado generate_image (Gemini nativo)
         → enforcer (reemplaza prompt por el canónico) → estado de correlación (TTL 10 min)
  ◀──(2) tool_call generate_image(prompt canónico) [SSE o JSON]
Open WebUI ejecuta generate_image con SU motor y credenciales; muestra la imagen
Open WebUI ──(3) assistant tool_call + role=tool result──▶ backend
  backend: correlación → valida par tool_call/tool → continuación Gemini (signature exacta)
         → texto final + "Interpretive reconstruction" + advertencia + sidecar de procedencia
  ◀──(4) respuesta final
```

Cualquier fallo en el camino de imagen devuelve un error OpenAI acotado. Nunca se degrada a generación libre. Con el flag apagado, el comportamiento de texto es idéntico al actual.

## 4. Qué queda fuera de esta rama (requiere tu acción o aprobación)

1. **Prueba de pago real:** no se ejecuta ninguna llamada a un motor de imagen. Necesita tu aprobación explícita.
2. **Configuración de Open WebUI:** motor de imagen, usuario canary, permisos y toggle se configuran en el panel de administración.
3. **Secretos:** `RAG_TRUSTED_SERVICE_TOKEN` y `RAG_CORRELATION_HMAC_KEY` se generan y se inyectan desde `.env`, nunca en Git.
4. **Multi-réplica:** sigue en NO-GO hasta el spike de Redis.
5. **Verificación en vivo del ID de mensaje** que reenvía Open WebUI, para elegir entre `message` y `chat`.

## 5. Tareas

| # | Tarea | Archivos |
|---|---|---|
| 1 | Adaptador bloques → `EvidencePacket` y consulta limpia para recuperar | `grounding/retrieval_evidence.py` |
| 2 | Constructor de instrucción con función forzada y validación | `grounding/builder.py` |
| 3 | Re-enlace de argumentos validados en el adaptador Gemini | `providers/gemini_native.py` |
| 4 | Configuración: shadow, modelo de chat, modelo de imagen, alcance de correlación | `config.py`, `.env.example` |
| 5 | Orquestador de imagen (inicio, continuación, shadow, métricas) | `api/grounded_image.py` |
| 6 | Ruta de chat con encabezados y despacho imagen/texto | `api/chat.py` |
| 7 | Composición en `app.py` detrás de flags + endpoint de métricas | `app.py` |
| 8 | Compose: reenvío de identidad y token de servicio por `Authorization` | `docker-compose.yml` |
| 9 | Smoke de pago opt-in (12.3), excluido por defecto | `tests/smoke/`, `pytest.ini` |
| 10 | Tests offline de extremo a extremo de los dos turnos | `tests/integration/` |
| 11 | Runbook de activación paso a paso | `docs/grounded-image-activation.md`, `README.md` |
