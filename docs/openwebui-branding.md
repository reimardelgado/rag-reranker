# Personalizar la interfaz de Open WebUI

Open WebUI (v0.11.4 en `docker-compose.yml`) permite que el asistente tenga
identidad propia sin tocar su código. De menor a mayor esfuerzo:

## 1. Nombre de la aplicación

`WEBUI_NAME` en `.env` (por ejemplo `WEBUI_NAME=Asistente Ibero`). Open WebUI puede
añadir su propia marca al nombre; ver la sección de licencia.

## 2. Un "modelo" con identidad propia (lo que más cambia la experiencia)

En **Workspace → Models → +**, crear un preset sobre el modelo que sirve el backend:

- **Nombre y avatar** propios (p. ej. "Asistente de Arqueología Ibera" con un logo
  del proyecto), en lugar de `qwen2.5:14b` o `gemini-3.8-flash`.
- **Descripción** y **sugerencias de inicio** (prompt suggestions) con preguntas
  típicas del corpus.
- **Capacidades**: aquí se activan `image_generation` y `builtin_tools` para el canary.
- **Visibilidad**: compartirlo con el grupo de usuarios del proyecto.

Después, en **Admin → Settings → Models**, ocultar los modelos base para que los
usuarios solo vean el preset.

**Importante:** Open WebUI también conecta **directamente** con Ollama
(`OLLAMA_BASE_URL` del servicio `open-webui`). Esos modelos aparecen en la lista con
el mismo nombre que los del backend, pero **se saltan el RAG**. El preset debe
construirse sobre el modelo de la conexión OpenAI (`rag-backend`). Para evitar
confusiones, ocultar los modelos de la conexión directa o desactivarla en
**Admin → Settings → Connections**.

## 3. Mensajes y apariencia

- **Banners** (Admin → Settings → Interface): aviso fijo, por ejemplo "Las imágenes
  son reconstrucciones interpretativas".
- **Sugerencias por defecto** y **prompt de sistema** global en el mismo panel.
- **Tema** claro/oscuro, en los ajustes de cada usuario.

## 4. Logo, favicon y CSS

Se puede montar un volumen con archivos propios sobre los estáticos del contenedor
(favicon, logo, una hoja de estilos). Funciona, pero hay que revisarlo en cada
actualización de Open WebUI, porque las rutas internas cambian entre versiones.

## Licencia (verificar antes de cambiar marca)

Desde la v0.6.6, la licencia de Open WebUI restringe quitar o alterar su marca
("Open WebUI", logo) salvo en despliegues pequeños (hasta 50 usuarios en 30 días),
con licencia enterprise o con permiso expreso. Antes de sustituir logo o nombre,
revisar el `LICENSE` de la versión desplegada. Los puntos 1–3 (presets, avatares,
banners, sugerencias) son configuración normal y no tocan la marca.
