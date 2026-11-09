# Modo local (Ollama + ComfyUI) y modo Gemini

El backend funciona en uno de dos modos, elegidos con `DEFAULT_PROVIDER` en `.env`:

| | `DEFAULT_PROVIDER=ollama` (**por defecto**) | `DEFAULT_PROVIDER=gemini` (de pago) |
|---|---|---|
| Chat de texto | Ollama (`OLLAMA_DEFAULT_MODEL`) | Gemini; si falla, Ollama |
| `GEMINI_API_KEY` | No hace falta | Obligatoria |
| Modelos visibles en Open WebUI | Solo los de Ollama | Gemini y Ollama |
| Un cliente pide `gemini-*` | Se atiende con el modelo local por defecto | Va a Gemini |
| Agente de imagen (`GROUNDED_IMAGE_ADAPTER`) | `ollama` | `gemini_native` |
| Motor de imagen (Open WebUI) | ComfyUI o AUTOMATIC1111 | Gemini |
| Costo por uso | 0 | Tokens de Gemini + ~$0.039/imagen |

`GROUNDED_IMAGE_ADAPTER` y `GROUNDED_IMAGE_CHAT_MODEL` vacíos siguen al modo. Si se
rellenan, mandan ellos (por ejemplo, texto local con agente Gemini).

El log de arranque muestra el modo activo: `[LLM] modo=ollama modelo_por_defecto=...`.

## Cómo trabaja el agente de imagen con Ollama

El flujo de imagen fuerza dos funciones concretas (la instrucción con evidencia y
`generate_image`). Ollama no admite `tool_choice`, así que el adaptador
`backend/providers/ollama_native.py` ejecuta cada llamada forzada como **salida
estructurada**: envía el esquema JSON de la función en `format` y Ollama restringe
la generación a ese esquema. Esto funciona incluso con modelos de 3B. La ronda del
pie de imagen va sin herramientas. Todas las garantías del flujo (validación de
evidencia, prompt canónico, procedencia, correlación) son las mismas que con Gemini.

## Perfiles de hardware

Los modelos son orientativos; confirmar los tags con `ollama list` y en la biblioteca de Ollama.

| | PC de desarrollo (4 GB VRAM) | Servidor (20 GB VRAM) |
|---|---|---|
| Ollama | **En CPU** (sin override de GPU) para dejar la GPU a ComfyUI | En GPU (`docker-compose.gpu.yml`) |
| `OLLAMA_DEFAULT_MODEL` | `qwen2.5:3b` (~2 GB) | `qwen2.5:14b` (~9 GB) o `qwen2.5:7b` |
| `OLLAMA_KEEP_ALIVE` | `1m` | `30m` |
| `OLLAMA_TIMEOUT_SECONDS` | `600` | `300` |
| ComfyUI | SD 1.5, 512×512, `--lowvram` | SDXL 1024×1024 (~7 GB) o FLUX.1 schnell fp8 |
| Tiempo aproximado | Respuesta en decenas de segundos; imagen 30–90 s | Respuesta en segundos; imagen 5–20 s |

Con 20 GB caben a la vez un LLM de 14B en Q4 y SDXL. FLUX junto a un 14B queda justo;
con FLUX usar un LLM de 7B.

Licencias de los pesos de imagen: SD 1.5 y SDXL usan licencias OpenRAIL; FLUX.1
schnell es Apache 2.0; FLUX.1 dev **no** permite uso comercial.

## Puesta en marcha en Windows

### 1. Docker Desktop

- Backend **WSL2** activado y driver NVIDIA reciente en Windows. No hace falta instalar CUDA.
- Comprobar la GPU dentro de Docker:
  `docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi`

### 2. Arrancar el stack

```powershell
# Servidor (Ollama con GPU)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
# PC de 4 GB (Ollama en CPU, la GPU queda para ComfyUI)
docker compose up -d --build
```

Descargar el modelo una vez:

```powershell
docker exec ollama ollama pull qwen2.5:3b     # PC
docker exec ollama ollama pull qwen2.5:14b    # servidor
```

**Alternativa:** usar la app nativa de Ollama para Windows (usa la GPU directamente)
y poner `OLLAMA_BASE_URL=http://host.docker.internal:11434` en `.env`.

### 3. ComfyUI (nativo en Windows)

1. Descargar la versión portable de ComfyUI para NVIDIA y descomprimirla.
2. Copiar un checkpoint en `ComfyUI\models\checkpoints\` (SD 1.5 en el PC, SDXL o FLUX
   en el servidor).
3. Arrancar escuchando en la red para que los contenedores lleguen por
   `host.docker.internal`:
   ```powershell
   .\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --listen 0.0.0.0 --port 8188
   # PC de 4 GB: añadir --lowvram
   ```
   Limitar el puerto 8188 en el firewall de Windows a la red local o a Docker.
4. En ComfyUI, montar el flujo de texto a imagen y exportarlo con
   **Workflow → Export (API)**. Ese JSON es el que se carga en Open WebUI.

**AUTOMATIC1111** también vale: arrancar con `--api --listen` (y `--lowvram` o
`--medvram` en el PC) y elegir el motor `automatic1111` (puerto 7860).

### 4. Open WebUI

En **Admin → Settings → Images**:

- Motor: **ComfyUI**, URL `http://host.docker.internal:8188`.
- Cargar el workflow exportado y mapear los nodos: prompt, modelo, tamaño, pasos y semilla.
- Tamaño y pasos según el perfil (512×512 y 20 pasos en el PC).

`IMAGE_GENERATION_ENGINE` y `COMFYUI_BASE_URL` del compose solo cuentan en el
**primer** arranque de Open WebUI; después manda lo guardado en el panel.

Para el flujo con evidencia, sigue el runbook
[grounded-image-activation.md](grounded-image-activation.md): function calling en
modo **Native** y toggle de imagen solo en el chat canary. En modo local el canary
no tiene costo.

## Cambiar de modo

1. En `.env`: `DEFAULT_PROVIDER=gemini` y `GEMINI_API_KEY=...` (o al revés).
2. Si cambia el motor de imagen: **Admin → Settings → Images** en Open WebUI.
3. `docker compose up -d backend` (y Open WebUI si se tocó).
4. Comprobar en el log `[LLM] modo=...` y que `/v1/models` lista lo esperado.
5. Si el flujo de imagen está activo: `GROUNDED_IMAGE_AUTHORIZED_MODELS` y
   `GROUNDED_IMAGE_IMAGE_MODEL_ID` deben coincidir con el nuevo modelo y motor.
