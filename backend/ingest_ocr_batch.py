# -*- coding: utf-8 -*-

import os, sys, re, gc, time, hashlib, argparse
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Iterable, List, Optional, Dict, Any
from config import settings

import pymupdf as fitz
from PIL import Image
import pytesseract

try:
    import numpy as np
    from sklearn.cluster import KMeans
    _HAS_NUMPY = True
except Exception:
    np = None; KMeans = None; _HAS_NUMPY = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None; _HAS_CV2 = False

_paddle_ocr_instance = None

def _get_paddle():
    global _paddle_ocr_instance
    if _paddle_ocr_instance is not None:
        return _paddle_ocr_instance
    try:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR
        _paddle_ocr_instance = PaddleOCR(lang="es", use_textline_orientation=True)
        info("PaddleOCR inicializado OK.")
        return _paddle_ocr_instance
    except Exception as e:
        warn(f"PaddleOCR no disponible ({e}). Se usará solo Tesseract.")
        return None

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except Exception:
    chromadb = None; SentenceTransformer = None

# ===================== Config por defecto =====================

TESSERACT_EXE = os.getenv("TESSERACT_CMD", "tesseract")
OCR_DPI         = 400
OCR_LANG        = "spa+eng"
OCR_OEM         = 1
OCR_PSM         = 11

CHROMA_HOST     = settings.CHROMA_HOST
CHROMA_PORT     = settings.CHROMA_PORT
COLLECTION      = settings.COLLECTION
EMBED_MODEL     = settings.EMBED_MODEL
MAX_CHARS       = 1200
OVERLAP         = 150
BATCH           = 256
DEDUP_CHUNKS    = False
FOOTER_PCT      = 0.92   # umbral Y para considerar pie de página

PAGE_NUM_PATTERNS = [
    r"^\s*\d+\s*$",
    r"^\s*[ivxlcdmIVXLCDM]+\s*$",
    r"^\s*(page|p[aá]g(?:ina)?|p|pg)\.?\s*\d+(\s*(/|-|de)\s*\d+)?\s*$",
    r"^\s*\d+\s*(/|-|de)\s*\d+\s*$",
]

# ===================== Utilidades =====================

def info(msg: str): print(f"[INFO] {msg}")
def warn(msg: str): print(f"[WARN] {msg}")
def err(msg: str):  print(f"[ERROR] {msg}")


def ensure_tesseract(tesseract_path: str):
    if tesseract_path and os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        info(f"[OCR] Tesseract: {tesseract_path}")
    else:
        warn("[OCR] Tesseract no encontrado en la ruta; se intentará PATH del sistema.")


def clean_text(text: Optional[str]) -> str:
    import unicodedata
    text = (text or "").replace("\x00", " ")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"(\w+)-\s+(\w+)",       r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_page_num(s: str) -> bool:
    s = (s or "").strip()
    return any(re.match(p, s, flags=re.I) for p in PAGE_NUM_PATTERNS)


def remove_page_number_lines(text: Optional[str]) -> str:
    text = text or ""
    return "\n".join(ln for ln in text.splitlines() if not is_page_num(ln)).strip()


def is_garbage_text(text: Optional[str]) -> bool:
    text = text or ""
    if not text.strip():
        return True
    alnum = sum(c.isalnum() for c in text)
    if len(text) > 0 and (alnum / len(text)) < 0.4:
        return True
    if any(m in text for m in ["CID+", "□", "\ufffd"]):
        return True
    return False


# ===================== Filtro de líneas basura (v4.9) =====================

# Vocabulario mínimo de palabras funcionales en español/inglés para scoring
_COMMON_WORDS_ES = frozenset([
    "de","la","el","en","y","a","que","se","un","una","los","las","del","por",
    "con","para","como","pero","más","su","al","no","es","le","si","lo","me",
    "te","tu","fue","ser","ha","han","hay","cuando","donde","todo","todos",
    "este","esta","estos","estas","ese","esa","cual","sobre","bajo","entre",
    "the","of","and","to","in","a","is","it","that","was","for","on","are",
    "with","as","at","be","by","this","from","or","an","but","not","also",
    "we","you","have","had","he","she","they","do","did","his","her","which",
    # palabras de contenido frecuente en libros técnicos/arte
    "color","luz","pintura","dibujo","sombra","técnica","forma","color",
    "parte","base","papel","lienzo","blanco","negro","agua","mezcla",
])

def _line_garbage_score(line: str) -> float:
    """
    Devuelve score 0.0 (basura pura) a 1.0 (texto limpio) para una línea.
    Detecta el patrón clásico de OCR roto: "MBM Mem ale em ie Me MC ect Mae"
    que tiene tokens muy cortos, alta densidad de mayúsculas y pocas palabras reales.
    """
    line = line.strip()
    if not line:
        return 1.0
    tokens = line.split()
    if not tokens:
        return 1.0

    # Líneas muy cortas (1-2 tokens) no las penalizamos aquí
    if len(tokens) <= 2:
        return 1.0

    # Score basado en longitud media de token (tokens rotos son muy cortos: 2-3 chars)
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    len_score = min(1.0, avg_len / 5.0)  # tokens normales ≥5 chars de media

    # Score basado en ratio de palabras del vocabulario común
    lower_tokens = [t.lower().strip(".,;:!?()[]{}") for t in tokens]
    vocab_hits = sum(1 for t in lower_tokens if t in _COMMON_WORDS_ES and len(t) > 1)
    vocab_score = min(1.0, (vocab_hits / len(tokens)) * 3.0)  # >33% vocab = limpio

    # Score basado en ratio de tokens que parecen palabras reales (≥4 chars, alfa)
    real_word_hits = sum(1 for t in tokens if len(t) >= 4 and t.isalpha())
    real_score = min(1.0, real_word_hits / len(tokens) + 0.2)

    # Penalización por densidad anormal de mayúsculas (patrón "Me MC ect Mae")
    upper_count = sum(1 for t in tokens if t and t[0].isupper())
    cap_ratio = upper_count / len(tokens)
    # En texto normal ≈20-40% de tokens empiezan en mayúscula.
    # Basura: casi todos empiezan en mayúscula con tokens cortos.
    cap_penalty = 1.0
    if cap_ratio > 0.7 and avg_len < 4.0:
        cap_penalty = 0.3

    score = (len_score * 0.35 + vocab_score * 0.40 + real_score * 0.25) * cap_penalty
    return score


def filter_garbage_lines(text: str, threshold: float = 0.20,
                          min_line_len: int = 3) -> str:
    """
    Filtra líneas de texto que son basura de OCR (score < threshold).
    Preserva: líneas de código, números, URLs, líneas cortas de contexto.
    """
    if not text:
        return text
    result_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            result_lines.append(line)
            continue
        # Preservar líneas que son claramente estructurales o numéricas
        if re.match(r"^[\d\s\.\,\:\-\|\/]+$", stripped):
            result_lines.append(line); continue
        # Preservar URLs y rutas
        if re.search(r"https?://|www\.|\\[A-Za-z]|/[a-z]", stripped, re.I):
            result_lines.append(line); continue
        # Preservar líneas con [Figura] u otras etiquetas del pipeline
        if stripped.startswith("["):
            result_lines.append(line); continue
        # Preservar líneas muy cortas (podrían ser encabezados válidos)
        if len(stripped) < min_line_len:
            result_lines.append(line); continue
        score = _line_garbage_score(stripped)
        if score >= threshold:
            result_lines.append(line)
    return "\n".join(result_lines)


def has_font_artifacts(text: str) -> bool:
    """
    Detecta texto con artefactos de fuente no embebida correctamente:
    secuencias de caracteres como €@, é, ê, símbolos PUA Unicode,
    que PyMuPDF genera cuando no puede mapear glifos.
    Un bloque es artefacto si >30% de sus caracteres son no-ASCII
    fuera del rango latino extendido normal (á é í ó ú ñ ü etc. son válidos).
    """
    if not text:
        return False
    # Caracteres válidos: ASCII imprimible + latín extendido (U+00C0–U+024F)
    bad = 0
    for ch in text:
        cp = ord(ch)
        if cp < 32 and ch not in ('\n', '\t'):
            bad += 1
        elif 127 <= cp < 192:   # C1 controls y algunos símbolos raros
            bad += 1
        elif 0x0250 <= cp <= 0x036F:  # IPA y diacríticos combinados sueltos
            bad += 1
        elif 0xE000 <= cp <= 0xF8FF:  # Zona de uso privado (PUA) — siempre artefacto
            bad += 1
    return bad / max(1, len(text)) > 0.20


def clean_table_artifacts(text: str) -> str:
    """Elimina filas de tabla vacías (solo pipes/espacios)."""
    lines = text.splitlines()
    return "\n".join(ln for ln in lines if not re.match(r"^\s*(\|\s*)+$", ln))


# ===================== Drop cap / Letra capital =====================

def _get_block_body_size(text_blocks: list) -> float:
    """
    Calcula el tamaño de fuente del 'cuerpo' de la página como la moda
    de todos los spans. Usado para detectar drop caps y títulos.
    """
    sizes = []
    for b in text_blocks:
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                s = sp.get("size", 0)
                if s > 0:
                    sizes.append(round(s, 1))
    if not sizes:
        return 12.0
    # moda simple
    from collections import Counter
    return Counter(sizes).most_common(1)[0][0]


def _is_drop_cap_block(block: dict, body_size: float, drop_cap_ratio: float = 2.0) -> bool:
    """
    Devuelve True si el bloque es una letra capital:
    - Tiene exactamente 1 carácter de texto (o 1–2 chars)
    - Su tamaño de fuente es ≥ drop_cap_ratio × body_size
    """
    spans = [sp for ln in block.get("lines", []) for sp in ln.get("spans", [])]
    text  = "".join(sp.get("text", "") for sp in spans).strip()
    if len(text) > 3:
        return False
    sizes = [sp.get("size", 0) for sp in spans if sp.get("size", 0) > 0]
    if not sizes:
        return False
    max_size = max(sizes)
    return max_size >= drop_cap_ratio * body_size


def merge_drop_caps(filtered: list, body_size: float) -> list:
    """
    Fusiona cada drop cap con el bloque que le corresponde en el PDF.

    La letra capital (drop cap) en PyMuPDF aparece como un bloque cuya bbox
    está en la esquina superior izquierda del párrafo. El bloque de texto del
    párrafo empieza a la *derecha* del drop cap (misma Y aproximada) o
    ligeramente por debajo (si el drop cap ocupa varias líneas).

    Estrategia:
      1. Para cada bloque detectado como drop cap, buscar en toda la lista
         (no solo el siguiente) el bloque cuya bbox_x0 sea mayor que la del
         drop cap Y cuya bbox_y0 esté dentro de una ventana de ±2×body_size.
      2. Si ese bloque ya empieza con la letra (extracción doble), no duplicar.
      3. Marcar el bloque destino como "ya procesado" para no usarlo dos veces.
    """
    if not filtered:
        return filtered

    # Marcar cuáles son drop caps
    drop_cap_indices = set()
    for idx, blk in enumerate(filtered):
        if _is_drop_cap_block(blk, body_size):
            drop_cap_indices.add(idx)

    if not drop_cap_indices:
        return filtered

    absorbed = set()   # índices de bloques ya fusionados (target)
    patches  = {}      # idx_target → texto con drop cap prepended

    for dc_idx in sorted(drop_cap_indices):
        dc_blk    = filtered[dc_idx]
        drop_char = dc_blk["text"].strip()
        dc_x0, dc_y0, dc_x1, dc_y1 = dc_blk["bbox"]
        dc_h      = dc_y1 - dc_y0   # altura visual del drop cap

        # Estrategia corregida:
        # El párrafo al que pertenece el drop cap puede empezar en la misma X
        # (no siempre está a la derecha). Lo que lo distingue es que:
        #   1. Su zona Y solapa con la del drop cap (cy0 ≤ dc_y1 + margen)
        #   2. Tiene contenido largo (es el párrafo, no un título de 2-3 palabras)
        #   3. No es todo mayúsculas corto (eso sería un título de sección)
        # Ganador = el bloque con más texto que cumpla los criterios.
        best_idx   = None
        best_score = -1
        for j, cand in enumerate(filtered):
            if j == dc_idx or j in drop_cap_indices or j in absorbed:
                continue
            cx0, cy0, cx1, cy1 = cand["bbox"]
            ctext = cand["text"].strip()

            # La Y del candidato debe solapar con la zona vertical del drop cap
            v_overlap = cy0 <= dc_y1 + dc_h * 0.6 and cy1 >= dc_y0
            if not v_overlap:
                continue

            # Descartar bloques muy cortos (títulos, subtítulos)
            if len(ctext) < 15:
                continue

            # Descartar si es todo mayúsculas y corto (título de sección)
            words = ctext.split()
            if all(w == w.upper() for w in words) and len(ctext) < 80:
                continue

            # Score: preferir el bloque más largo en la zona
            score = len(ctext)
            if score > best_score:
                best_score = score
                best_idx   = j

        if best_idx is not None:
            target_text = filtered[best_idx]["text"]
            if target_text and target_text[0].lower() == drop_char.lower():
                # El párrafo ya tiene la letra (extracción doble): no duplicar
                patches[best_idx] = target_text
            else:
                patches[best_idx] = drop_char + target_text
            absorbed.add(best_idx)

    # Reconstruir lista: omitir drop caps, aplicar patches a los targets
    result = []
    for idx, blk in enumerate(filtered):
        if idx in drop_cap_indices:
            continue   # eliminar el bloque drop cap suelto
        if idx in patches:
            result.append({**blk, "text": patches[idx], "is_drop_cap_merged": True})
        else:
            result.append(blk)
    return result


# ===================== Extracción de imágenes con texto =====================

def extract_image_text_native(page: fitz.Page) -> str:
    """
    Extrae texto de bloques tipo imagen (type==1) que contienen texto
    embebido en el PDF (common en gráficos vectoriales exportados).
    Para PDFs nativos con texto dentro de figuras esto funciona sin OCR.
    Devuelve el texto concatenado, o "" si no hay nada útil.
    """
    texts = []
    try:
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if b.get("type") != 1:
                continue
            # Algunos bloques imagen tienen un campo "lines" con texto embebido
            for ln in b.get("lines", []):
                for sp in ln.get("spans", []):
                    t = (sp.get("text", "") or "").strip()
                    if t and not is_page_num(t):
                        texts.append(t)
    except Exception:
        pass
    return clean_text(" ".join(texts)) if texts else ""


def ocr_image_blocks(page: fitz.Page, doc: fitz.Document, page_index: int,
                     dpi: int = 300, lang: str = "spa+eng", oem: int = 1) -> str:
    """
    Para cada bloque imagen en la página, renderiza solo esa región y
    aplica Tesseract PSM=6 (bloque uniforme de texto). Útil para:
    - Infografías con texto
    - Tablas escaneadas embebidas como imagen
    - Capturas de pantalla dentro del PDF
    Devuelve texto concatenado de todas las imágenes con contenido útil.
    """
    results = []
    try:
        blocks = page.get_text("dict").get("blocks", [])
        img_blocks = [b for b in blocks if b.get("type") == 1]
        if not img_blocks:
            return ""

        scale = dpi / 72
        mat   = fitz.Matrix(scale, scale)

        for b in img_blocks:
            try:
                x0, y0, x1, y1 = b["bbox"]
                # Ignorar imágenes muy pequeñas (iconos, decoraciones)
                w_pt = x1 - x0
                h_pt = y1 - y0
                if w_pt < 30 or h_pt < 20:
                    continue

                # Renderizar la región de la imagen
                clip = fitz.Rect(x0, y0, x1, y1)
                pix  = doc[page_index].get_pixmap(matrix=mat, clip=clip, alpha=False)
                pil  = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                # Si la imagen es demasiado pequeña en píxeles, saltar
                if pil.width < 60 or pil.height < 40:
                    continue

                cfg  = f"--oem {oem} --psm 6 -c preserve_interword_spaces=1"
                txt  = pytesseract.image_to_string(
                    pil.convert("L"), lang=lang, config=cfg
                ) or ""
                txt  = clean_text(txt)
                txt  = remove_page_number_lines(txt)

                if txt and not is_garbage_text(txt) and len(txt) > 15:
                    results.append(f"[Figura] {txt}")
            except Exception:
                continue
    except Exception:
        pass

    return "\n\n".join(results)


# ===================== Chunking =====================

def chunk(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[str]:
    """Sliding-window con solapamiento, respetando párrafos cuando es posible."""
    text = (text or "").strip()
    n = len(text)
    if n == 0:
        return []
    parts = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            cut = text.rfind("\n\n", start, end)
            if cut > start + overlap:
                end = cut + 2
        part = text[start:end].strip()
        if len(part) > 50:
            parts.append(part)
        if end >= n:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return parts


def chunk_structured(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    """Chunking por bloques semánticos (separados por doble salto de línea)."""
    blocks = [b.strip() for b in re.split(r"\n\n+", text) if b.strip()]
    if not blocks:
        return []
    chunks = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > max_chars:
            chunks.append(current.strip())
            current = block
        else:
            current = (current + "\n\n" + block).strip() if current else block
    if current.strip():
        chunks.append(current.strip())
    return chunks


def smart_chunk(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[str]:
    """
    Elige estrategia de chunking:
    - Muchos párrafos definidos → chunk_structured
    - Texto fluido → sliding-window
    Bloques muy largos de structured se re-dividen con sliding-window.
    """
    double_newlines = len(re.findall(r"\n\n", text))
    total_lines     = max(1, len(text.splitlines()))
    if double_newlines / total_lines > 0.15:
        result = chunk_structured(text, max_chars)
        final  = []
        for c in result:
            if len(c) > max_chars * 1.5:
                final.extend(chunk(c, max_chars, overlap))
            else:
                final.append(c)
        return final
    return chunk(text, max_chars, overlap)


# ===================== OCR helpers =====================

def pil_to_cv2(img: Image.Image):
    if not _HAS_CV2 or not _HAS_NUMPY:
        return None
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def apply_gamma(image, gamma: float):
    if not _HAS_CV2 or not _HAS_NUMPY:
        return image
    invG  = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** invG * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# ===================== Corrección de iluminación avanzada (v4.9) =====================

def homomorphic_filter(gray, sigma: float = 30.0, boost: float = 1.5) -> np.ndarray:
    """
    Filtro homomórfico para normalizar iluminación irregular.
    Separa la reflectancia de la iluminación en el dominio logarítmico.
    Muy efectivo para páginas con luz lateral, sombras de encuadernación,
    o iluminación desigual de escáner de cama plana.
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return gray
    rows, cols = gray.shape
    # Log transform (evitar log(0))
    img_log = np.log1p(gray.astype(np.float32))
    # Gaussian blur = componente de iluminación de baja frecuencia
    blur = cv2.GaussianBlur(img_log, (0, 0), sigma)
    # Restar iluminación: queda la reflectancia
    img_hf = img_log - blur
    # Amplificar y normalizar
    img_hf = img_hf * boost
    # Exponential inverse
    img_out = np.expm1(img_hf)
    # Normalizar a [0, 255]
    img_out = cv2.normalize(img_out, None, 0, 255, cv2.NORM_MINMAX)
    return img_out.astype(np.uint8)


def single_scale_retinex(gray, sigma: float = 80.0) -> np.ndarray:
    """
    Retinex de escala única para compensar variaciones de iluminación globales.
    Útil para páginas amarillentas, con manchas de humedad, o escaneadas con
    tapa del escáner abierta.
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return gray
    img_f = gray.astype(np.float32) + 1.0
    blur  = cv2.GaussianBlur(img_f, (0, 0), sigma)
    retinex = np.log10(img_f) - np.log10(blur + 1.0)
    out = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)


def adaptive_clahe(gray) -> np.ndarray:
    """
    CLAHE con tamaño de tile dinámico basado en la resolución de la imagen.
    A mayor resolución (DPI alto), tiles más grandes para no fragmentar caracteres.
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return gray
    h, w = gray.shape
    # Tile ~2% del lado más corto, mínimo 8 px
    tile = max(8, int(min(h, w) * 0.02))
    # Asegurar que es impar (para compatibilidad interna)
    tile = tile if tile % 2 == 0 else tile + 1
    clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tile, tile))
    return clahe_obj.apply(gray)


def correct_illumination(gray, mode: str = "auto") -> np.ndarray:
    """
    Selecciona y aplica la corrección de iluminación más adecuada.
    mode: "auto" detecta automáticamente, "homomorphic", "retinex", "clahe", "combined"
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return gray

    if mode == "auto":
        # Detectar tipo de problema de iluminación
        # Calcular gradiente de iluminación (media en franjas horizontales)
        h = gray.shape[0]
        strip_h = max(1, h // 8)
        strip_means = [gray[i*strip_h:(i+1)*strip_h, :].mean() for i in range(8)]
        # Si hay mucha variación entre franjas → iluminación irregular → homomórfico
        variation = max(strip_means) - min(strip_means)
        if variation > 40:
            mode = "homomorphic"
        # Si la imagen es muy oscura en general → retinex
        elif gray.mean() < 100:
            mode = "combined"
        else:
            mode = "clahe"

    if mode == "homomorphic":
        return homomorphic_filter(gray)
    elif mode == "retinex":
        return single_scale_retinex(gray)
    elif mode == "clahe":
        return adaptive_clahe(gray)
    elif mode == "combined":
        # Homomorph primero, luego CLAHE
        g = homomorphic_filter(gray, sigma=25.0)
        return adaptive_clahe(g)
    return gray


# ===================== Detección de tipografía (v4.9) =====================

def detect_font_type(gray) -> str:
    """
    Clasifica el tipo de fuente predominante en la imagen para adaptar
    los parámetros de Tesseract.
    Retorna: "serif", "sans", "script", "bold", "normal"
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return "normal"
    try:
        # Binarizar con Otsu
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (255 - bw)

        # ── Detección de serif: las serifas crean protuberancias horizontales
        # Las fuentes serif tienen más variación horizontal en trazos verticales
        horiz_profile = ink.sum(axis=1).astype(float)
        vert_profile  = ink.sum(axis=0).astype(float)
        h_std = np.std(horiz_profile) / (np.mean(horiz_profile) + 1)
        v_std = np.std(vert_profile)  / (np.mean(vert_profile)  + 1)

        # ── Grosor de trazo: fuentes bold tienen trazos más gruesos
        # Erosión para medir grosor: si queda mucho contenido → trazo grueso
        kernel_thin = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(ink, kernel_thin, iterations=2)
        ink_ratio_after = eroded.sum() / (ink.sum() + 1)

        # ── Continuidad: fuentes script tienen trazos muy continuos
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            avg_area = np.mean(areas) if areas else 0
            # Script: pocos contornos grandes (letras unidas)
            script_score = len([a for a in areas if a > avg_area * 3])
        else:
            script_score = 0
            avg_area = 0

        # Clasificar
        if script_score > 5 and len(contours) < 50:
            return "script"
        if ink_ratio_after > 0.55:
            return "bold"
        if h_std > 2.5 and v_std < h_std * 0.8:
            return "serif"
        if h_std < 1.5:
            return "sans"
        return "normal"
    except Exception:
        return "normal"


def get_tess_config_for_font(font_type: str, base_oem: int = 1,
                              base_psm: int = 11) -> Tuple[str, List[int]]:
    """
    Devuelve (config_string, psm_list) optimizados para el tipo de fuente.
    """
    base_cfg = (
        f"--oem {base_oem} "
        "-c preserve_interword_spaces=1 "
        "-c tessedit_do_invert=0 "
    )
    if font_type == "script":
        # Cursiva / handwriting: PSM 6 (bloque uniforme) funciona mejor
        # Aumentar tolerancia de segmentación de caracteres
        cfg = base_cfg + "-c classify_bln_numeric_mode=0 "
        psm_list = [6, 4, 3, 1]
    elif font_type == "bold":
        # Negrita condensada: Otsu suele sobre-binarizar; usar PSM más conservador
        cfg = base_cfg + "-c textord_min_linesize=1.5 "
        psm_list = [6, 11, 4, 3]
    elif font_type == "serif":
        # Serif clásico: PSM 11 (texto disperso) suele funcionar bien
        cfg = base_cfg
        psm_list = [11, 6, 3, 1]
    elif font_type == "sans":
        # Sans-serif moderno: PSM 3 (automático con OSD) es bueno
        cfg = base_cfg
        psm_list = [3, 6, 11, 4]
    else:
        # Normal: comportamiento heredado
        cfg = base_cfg
        psm_list = [11, 6, 4, 3, 1]

    return cfg, psm_list


# ===================== Detección de orientación mejorada (v4.9) =====================

def _score_text_orientation(img: Image.Image, lang: str = "spa+eng",
                             oem: int = 1) -> float:
    """
    Hace un OCR rápido (PSM 6) y devuelve la ratio de palabras del vocabulario
    sobre el total de tokens. Usado para comparar rotaciones.
    """
    try:
        cfg = f"--oem {oem} --psm 6 -c preserve_interword_spaces=1"
        txt = pytesseract.image_to_string(img, lang=lang, config=cfg) or ""
        tokens = txt.lower().split()
        if len(tokens) < 3:
            # Para imágenes con poco texto, usar ratio alnum como proxy
            alnum = sum(c.isalnum() for c in txt)
            return alnum / max(len(txt), 1)
        hits = sum(1 for t in tokens if t.strip(".,;:!?") in _COMMON_WORDS_ES)
        return hits / len(tokens)
    except Exception:
        return 0.0


def _hough_deskew_angle(gray) -> Optional[float]:
    """
    Detecta el ángulo de inclinación usando Hough Lines.
    Útil cuando OSD de Tesseract no converge (poca densidad de texto).
    Retorna el ángulo en grados o None si no se puede determinar.
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return None
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=gray.shape[1] // 8,
                                maxLineGap=20)
        if lines is None or len(lines) < 5:
            return None
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Filtrar ángulos que no corresponden a texto horizontal (±45°)
            if -45 < angle < 45:
                angles.append(angle)
        if not angles:
            return None
        # Usar la mediana para robustez
        return float(np.median(angles))
    except Exception:
        return None


def detect_orientation_and_rotate(pil_img: Image.Image,
                                   lang: str = "spa+eng",
                                   use_scoring: bool = False) -> Image.Image:
    """
    Detección de orientación mejorada (v4.9):
    1. Intenta OSD de Tesseract primero.
    2. Si falla o el ángulo es 0, prueba Hough Lines para deskew fino.
    3. Si use_scoring=True, prueba las 4 rotaciones principales y elige
       la que produce mejor score de texto (más lento pero más robusto).
    """
    if use_scoring and _HAS_CV2 and _HAS_NUMPY:
        # Probar 4 rotaciones y seleccionar la mejor
        best_score = -1.0
        best_img   = pil_img
        for angle in (0, 90, 180, 270):
            if angle == 0:
                rotated = pil_img
            else:
                rotated = pil_img.rotate(angle, expand=True)
            # Reducir para hacer el scoring más rápido
            w, h = rotated.size
            scale = min(1.0, 800 / max(w, h))
            if scale < 1.0:
                small = rotated.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            else:
                small = rotated
            score = _score_text_orientation(small.convert("L"), lang=lang)
            if score > best_score:
                best_score = score
                best_img   = rotated
        return best_img

    # Método rápido: OSD primero, Hough como fallback
    try:
        osd   = pytesseract.image_to_osd(pil_img)
        m     = re.search(r"(?i)(Rotate|Orientation in degrees)\D+(\d+)", osd)
        angle = int(m.group(2)) if m else 0
        if angle in (90, 180, 270):
            return pil_img.rotate(360 - angle, expand=True)
    except Exception:
        pass

    # Fallback: Hough Lines para deskew fino (±5°)
    if _HAS_CV2 and _HAS_NUMPY:
        try:
            arr  = np.array(pil_img.convert("L"))
            hough_angle = _hough_deskew_angle(arr)
            if hough_angle is not None and abs(hough_angle) > 0.3:
                return pil_img.rotate(-hough_angle, expand=False,
                                       resample=Image.BICUBIC)
        except Exception:
            pass

    return pil_img


def deshadow(gray):
    """Elimina sombras mediante morfología de fondo (divide por cierre morfológico)."""
    if not _HAS_CV2 or not _HAS_NUMPY:
        return gray
    dil    = max(15, int(min(gray.shape[:2]) * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dil, dil))
    bg     = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.divide(gray, bg, scale=255)


def deskew_image(gray):
    """Deskew fino por minAreaRect (legado v4.8, usado como último recurso)."""
    if not _HAS_CV2 or not _HAS_NUMPY:
        return gray
    thr    = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr == 0))
    if coords.size == 0:
        return gray
    rect  = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.3:
        return gray
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_for_ocr(pil_img: Image.Image,
                       deskew=True, clahe=True, deshadow_en=True,
                       denoise=True, adaptive_th=True,
                       gamma=None,
                       illumination_mode: str = "auto",
                       font_type: str = "auto") -> Image.Image:
    """
    Pipeline de preprocesado v4.9.
    Novedades respecto a v4.8:
    - correct_illumination() reemplaza al CLAHE fijo (más robusto en páginas con sombra).
    - detect_font_type() adapta la binarización al tipo de letra.
    - deskew ahora usa Hough primero y minAreaRect solo como fallback.
    Devuelve PIL modo 'L'.
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return pil_img.convert("L")
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")
    img_bgr = pil_to_cv2(pil_img)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Gamma manual si se especifica
    if gamma is not None:
        gray = apply_gamma(gray, gamma)

    # Detección automática de fondo oscuro (escaneado invertido)
    if gray.mean() < 80:
        gray = 255 - gray

    # Reducción de ruido leve
    gray = cv2.medianBlur(gray, 3)

    # Desombrado morfológico
    if deshadow_en:
        gray = deshadow(gray)

    # ── Corrección de iluminación (v4.9) ──
    if clahe:
        gray = correct_illumination(gray, mode=illumination_mode)

    # ── Deskew (v4.9): Hough primero, minAreaRect como fallback ──
    if deskew:
        hough_angle = _hough_deskew_angle(gray)
        if hough_angle is not None and 0.3 < abs(hough_angle) < 15:
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), -hough_angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        else:
            gray = deskew_image(gray)

    # ── Detección de tipo de fuente para binarización adaptativa (v4.9) ──
    if font_type == "auto":
        font_type = detect_font_type(gray)

    # Binarización según tipo de fuente
    thr_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if font_type in ("script", "bold"):
        # Fuentes difíciles: umbral adaptativo con bloque más grande
        block_size = 41 if font_type == "bold" else 51
        thr_loc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block_size, 10)
    else:
        # Serif / sans / normal: comportamiento heredado
        thr_loc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 11)

    ink_otsu = (255 - thr_otsu).sum()
    ink_loc  = (255 - thr_loc).sum()
    bw = thr_loc if ink_loc > 1.1 * ink_otsu else thr_otsu

    # Corrección de inversión (tinta blanca sobre fondo negro)
    inv = 255 - bw
    if (255 - inv).sum() > (255 - bw).sum() and inv.mean() < 180:
        bw = inv

    return Image.fromarray(bw).convert("L")


def _content_bbox_safe(pil_img: Image.Image, pad_ratio=0.02) -> Image.Image:
    """Recorta a la caja de contenido. No recorta si eliminaría >15% del alto."""
    if not _HAS_CV2 or not _HAS_NUMPY:
        return pil_img
    arr = np.array(pil_img)
    if   arr.ndim == 2:              gray = arr
    elif arr.shape[2] == 1:          gray = arr[:, :, 0]
    elif arr.shape[2] == 3:          gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif arr.shape[2] == 4:          gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    else:                            gray = np.array(pil_img.convert("L"))
    thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv  = 255 - thr
    cnts = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts:
        return pil_img
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    pad_x = max(2, int(pad_ratio * pil_img.width))
    pad_y = max(2, int(pad_ratio * pil_img.height))
    x0 = max(0, x - pad_x);  y0 = max(0, y - pad_y)
    x1 = min(pil_img.width,  x + w + pad_x)
    y1 = min(pil_img.height, y + h + pad_y)
    if (pil_img.height - (y1 - y0)) / pil_img.height > 0.15:
        return pil_img
    return pil_img.crop((x0, y0, x1, y1))


def _tess_best(img: Image.Image, lang="spa+eng", oem=1,
               psm_list=(11, 6, 4, 3, 1),
               base_cfg: str = "") -> Tuple[str, int, int]:
    """
    Prueba varios PSM y retorna (texto_limpio, psm_usado, longitud).
    v4.9: acepta base_cfg para configuración por tipo de fuente.
    El ganador se elige por: longitud de texto limpio × (1 + vocab_score).
    """
    def run(psm):
        if base_cfg:
            # base_cfg ya incluye --oem; solo añadir --psm
            cfg = base_cfg + f"--psm {psm}"
        else:
            cfg = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1 -c tessedit_do_invert=0"
        txt = pytesseract.image_to_string(img, lang=lang, config=cfg) or ""
        txt_clean = clean_text(txt)
        # Score combinado: longitud + ratio vocab para evitar elegir texto basura largo
        tokens = txt_clean.lower().split()
        if tokens:
            vocab_hits = sum(1 for t in tokens if t.strip(".,;:!?") in _COMMON_WORDS_ES)
            vocab_score = vocab_hits / len(tokens)
        else:
            vocab_score = 0.0
        combined_score = len(txt_clean) * (1.0 + vocab_score)
        return txt_clean, psm, combined_score

    best_txt, best_psm, best_score = "", None, 0.0
    for p in psm_list:
        t, pu, S = run(p)
        if S > best_score:
            best_txt, best_psm, best_score = t, pu, S
    return best_txt, best_psm, int(best_score)


def _split_grid(w, h, cols, rows, overlap=0.05):
    tiles = []
    ow = int(w * overlap); oh = int(h * overlap)
    cw = w // cols;        ch = h // rows
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * cw - (ow if c > 0 else 0))
            y0 = max(0, r * ch - (oh if r > 0 else 0))
            x1 = min(w, (c+1) * cw + (ow if c < cols-1 else 0))
            y1 = min(h, (r+1) * ch + (oh if r < rows-1 else 0))
            tiles.append((x0, y0, x1, y1))
    return tiles


def _dedup_lines(text: str) -> str:
    out, seen = [], set()
    for ln in (text or "").splitlines():
        key = re.sub(r"\W+", "", ln.lower())[:90]
        if key and key not in seen:
            seen.add(key); out.append(ln)
    return "\n".join(out)


def _bottom_sweep(img: Image.Image, lang, oem, psm_list) -> str:
    W, H    = img.size
    band_h  = int(0.35 * H)
    y_start = max(0, H - band_h)
    rows    = 5
    step    = band_h // rows if rows > 0 else band_h
    parts   = []
    for i in range(rows):
        y0   = y_start + max(0, i * step - int(0.03 * H))
        y1   = H if i == rows-1 else (y_start + (i+1) * step + int(0.03 * H))
        crop = img.crop((0, y0, W, y1))
        t, _, _ = _tess_best(crop, lang=lang, oem=oem, psm_list=psm_list)
        parts.append(t)
    return _dedup_lines("\n".join(p for p in parts if p.strip()))


# ===================== OCR con PaddleOCR =====================

def ocr_page_paddle(pil_img: Image.Image) -> str:
    """OCR con PaddleOCR sobre imagen PIL. Detecta columnas con KMeans."""
    paddle = _get_paddle()
    if paddle is None or not _HAS_NUMPY:
        return ""
    try:
        img_array = np.array(pil_img.convert("RGB"))
        result    = paddle.ocr(img_array)
    except Exception as e:
        warn(f"PaddleOCR error: {e}")
        return ""
    if not result or not result[0]:
        return ""

    boxes_data = []
    for line in result[0]:
        try:
            box  = line[0]; text = line[1][0]; conf = line[1][1]
            if conf < 0.5:
                continue
            xc = [p[0] for p in box]; yc = [p[1] for p in box]
            boxes_data.append({
                "text":     text.strip(),
                "x_min":    min(xc), "x_max": max(xc),
                "y_min":    min(yc), "y_max": max(yc),
                "x_center": (min(xc) + max(xc)) / 2,
            })
        except Exception:
            continue
    if not boxes_data:
        return ""

    x_centers = np.array([[b["x_center"]] for b in boxes_data])
    n_clusters = 1
    if len(x_centers) > 10 and KMeans is not None:
        try:
            km2    = KMeans(n_clusters=2, n_init=10).fit(x_centers)
            spread = abs(np.mean(x_centers[km2.labels_ == 0]) -
                         np.mean(x_centers[km2.labels_ == 1]))
            if spread > 80:
                n_clusters = 2
        except Exception:
            n_clusters = 1

    if n_clusters > 1:
        km      = KMeans(n_clusters=2, n_init=10).fit(x_centers)
        centers = [np.mean(x_centers[km.labels_ == c]) for c in range(2)]
        left_lbl = int(np.argmin(centers))
        for i, b in enumerate(boxes_data):
            b["column"] = 0 if km.labels_[i] == left_lbl else 1
    else:
        for b in boxes_data:
            b["column"] = 0

    boxes_data.sort(key=lambda b: (b["column"], b["y_min"], b["x_min"]))

    lines, cur_line, cur_y = [], [], None
    for b in boxes_data:
        if cur_y is None:
            cur_line, cur_y = [b], b["y_min"]
        elif abs(b["y_min"] - cur_y) < 15:
            cur_line.append(b)
        else:
            lines.append(cur_line)
            cur_line, cur_y = [b], b["y_min"]
    if cur_line:
        lines.append(cur_line)

    ordered = []
    for line in lines:
        joined = " ".join(b["text"] for b in sorted(line, key=lambda b: b["x_min"]))
        joined = re.sub(r"  +", " ", joined).strip()
        if len(joined) > 1:
            ordered.append(joined)
    return "\n".join(ordered)


# ===================== OCR con Tesseract (página completa) =====================

def ocr_page(doc: fitz.Document, page_index: int, dpi: int,
             deskew=True, clahe=True, deshadow_en=True, denoise=True,
             adaptive_th=True, gamma=None,
             ocr_lang="spa+eng", ocr_psm=11, ocr_oem=1,
             use_osd=False, keep_numbers=True,
             tiling=False, tiles_cols=2, tiles_rows=5,
             target_phrase=None, debug=False,
             illumination_mode: str = "auto",
             orient_scoring: bool = False,
             garbage_line_thr: float = 0.20) -> Tuple[str, dict]:
    """
    OCR robusto v4.9.
    Novedades:
    - detect_font_type() para configurar Tesseract según la tipografía.
    - correct_illumination() con modo automático.
    - detect_orientation_and_rotate() mejorado (Hough + scoring opcional).
    - filter_garbage_lines() sobre el resultado final.
    - Pipeline fallback: si el texto sigue siendo basura, reintenta con
      preset "script", "bold", y modo de iluminación "combined".
    """
    metrics = {}

    def _render(d):
        mat = fitz.Matrix(d / 72, d / 72)
        pix = doc[page_index].get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    def _run_tess(pil_img, current_dpi, force_font_type=None, force_illum=None):
        # ── Orientación ──
        if use_osd or orient_scoring:
            pil_img = detect_orientation_and_rotate(
                pil_img, lang=ocr_lang, use_scoring=orient_scoring
            )
        pil_img = _content_bbox_safe(pil_img)

        # ── Detección de tipo de fuente (solo una vez por imagen) ──
        # Conversión rápida a gray para detección
        if _HAS_CV2 and _HAS_NUMPY:
            arr_gray = np.array(pil_img.convert("L"))
        else:
            arr_gray = None

        detected_font = force_font_type
        if detected_font is None and arr_gray is not None:
            detected_font = detect_font_type(arr_gray)
        if detected_font is None:
            detected_font = "normal"

        illum = force_illum if force_illum is not None else illumination_mode

        # ── Preprocesado ──
        pil_pre = preprocess_for_ocr(
            pil_img,
            deskew=deskew, clahe=clahe, deshadow_en=deshadow_en,
            denoise=denoise, adaptive_th=adaptive_th, gamma=gamma,
            illumination_mode=illum,
            font_type=detected_font,
        )

        # ── Config Tesseract adaptada al tipo de fuente ──
        base_cfg, psm_list = get_tess_config_for_font(detected_font, base_oem=ocr_oem)
        best_txt, best_psm, _ = _tess_best(pil_pre, lang=ocr_lang, oem=ocr_oem,
                                            psm_list=psm_list, base_cfg=base_cfg)

        if tiling:
            W, H = pil_pre.size
            tile_texts = []
            for (tx0, ty0, tx1, ty1) in _split_grid(W, H, tiles_cols, tiles_rows):
                t, _, _ = _tess_best(pil_pre.crop((tx0, ty0, tx1, ty1)),
                                     lang=ocr_lang, oem=ocr_oem,
                                     psm_list=psm_list, base_cfg=base_cfg)
                tile_texts.append(t)
            combined = _dedup_lines(best_txt + "\n" + "\n".join(tile_texts))
        else:
            combined = best_txt

        sweep = _bottom_sweep(pil_img, lang=ocr_lang, oem=ocr_oem, psm_list=psm_list)
        if sweep:
            combined = _dedup_lines(combined + "\n" + sweep)
        if not keep_numbers:
            combined = remove_page_number_lines(combined)

        # ── Filtro de líneas basura (v4.9) ──
        combined = filter_garbage_lines(combined, threshold=garbage_line_thr)

        return combined, best_psm, detected_font

    # ── Intento principal ──
    pil = _render(dpi)
    text, best_psm, font_used = _run_tess(pil, dpi)
    metrics.update(psm_used=best_psm, dpi_used=dpi, font_type=font_used)

    # ── Fallback si sigue siendo basura (v4.9) ──
    # Evalúa si el texto producido aún tiene alto ratio de garbage
    lines_total = [ln for ln in text.splitlines() if ln.strip()]
    if lines_total:
        garbage_lines = sum(1 for ln in lines_total
                            if _line_garbage_score(ln) < garbage_line_thr)
        garbage_ratio = garbage_lines / len(lines_total)
    else:
        garbage_ratio = 1.0

    if garbage_ratio > 0.5:
        if debug:
            info(f"  [OCR v4.9] Fallback activado (garbage_ratio={garbage_ratio:.2f})")
        fallback_attempts = [
            ("script",  "auto"),
            ("bold",    "combined"),
            ("normal",  "homomorphic"),
        ]
        for fb_font, fb_illum in fallback_attempts:
            if fb_font == font_used and fb_illum == illumination_mode:
                continue  # Ya probado
            text_fb, psm_fb, _ = _run_tess(pil, dpi,
                                             force_font_type=fb_font,
                                             force_illum=fb_illum)
            lines_fb = [ln for ln in text_fb.splitlines() if ln.strip()]
            if lines_fb:
                gb_fb = sum(1 for ln in lines_fb
                            if _line_garbage_score(ln) < garbage_line_thr)
                ratio_fb = gb_fb / len(lines_fb)
            else:
                ratio_fb = 1.0
            if debug:
                info(f"  [OCR v4.9] Fallback font={fb_font} illum={fb_illum} "
                     f"→ garbage_ratio={ratio_fb:.2f} chars={len(text_fb)}")
            if ratio_fb < garbage_ratio or (ratio_fb == garbage_ratio
                                             and len(text_fb) > len(text)):
                text         = text_fb
                best_psm     = psm_fb
                garbage_ratio = ratio_fb
                font_used     = fb_font
                metrics.update(psm_used=psm_fb, font_type=fb_font,
                                fallback=f"{fb_font}/{fb_illum}")
            if garbage_ratio < 0.3:
                break  # Suficientemente limpio

    # ── Reintento por target_phrase ──
    if target_phrase and target_phrase.lower() not in text.lower():
        for retry_dpi in (350, 300):
            if debug:
                info(f"  [OCR] Reintento DPI={retry_dpi}")
            pil_r = _render(retry_dpi)
            text_r, psm_r, _ = _run_tess(pil_r, retry_dpi,
                                           force_illum="combined")
            if len(text_r) > len(text):
                text, best_psm = text_r, psm_r
                metrics.update(psm_used=psm_r, dpi_used=retry_dpi)
            if target_phrase.lower() in text.lower():
                break

    if debug:
        info(f"  [OCR v4.9] pág.{page_index+1} DPI={metrics['dpi_used']} "
             f"PSM={metrics['psm_used']} font={metrics.get('font_type','?')} "
             f"chars={len(text)}")
    return text, metrics




# ===================== Extracción nativa =====================

def should_force_ocr(page: fitz.Page, native_text: str,
                     min_chars: int, min_words: int, img_ratio: float) -> bool:
    txt = (native_text or "").strip()
    if len(txt) < min_chars:
        return True
    if len(re.findall(r"\w+", txt, flags=re.UNICODE)) < min_words:
        return True
    d      = page.get_text("dict")
    n_text = len([b for b in d.get("blocks", []) if b.get("type") == 0])
    n_imgs  = len([b for b in d.get("blocks", []) if b.get("type") == 1])
    if n_imgs >= 1 and (n_text == 0 or (n_imgs / max(1, n_text)) >= img_ratio):
        return True
    return False


def extract_text_smart_native(doc: fitz.Document, page_index: int,
                               header_pct: float   = 0.12,
                               footer_pct: float   = FOOTER_PCT,
                               size_tol: float     = 0.06,
                               x_center_tol: int   = 28,
                               max_x_triggers: int = 3,
                               ocr_images: bool    = False,
                               ocr_images_dpi: int = 300,
                               ocr_images_lang: str = "spa+eng") -> str:
    """
    Extracción nativa mejorada v4.8:
    - Filtrado encabezado / pie configurable
    - Detección y fusión de drop caps (letras capitales)
    - Detección de columnas (heurística midpoint)
    - Detección de tablas con tolerancia dinámica por tamaño de fuente
    - Extracción de texto en bloques imagen (embebido)
    - OCR opcional de regiones imagen (--ocr-images)
    """
    page = doc[page_index]
    W, H = page.rect.width, page.rect.height
    top_y = H * header_pct

    blocks      = page.get_text("dict").get("blocks", [])
    text_blocks = [b for b in blocks if b.get("type") == 0]

    if not text_blocks:
        # Intentar capturar texto de imágenes embebidas igualmente
        img_text = extract_image_text_native(page)
        if ocr_images:
            img_text += "\n\n" + ocr_image_blocks(page, doc, page_index,
                                                    dpi=ocr_images_dpi,
                                                    lang=ocr_images_lang)
        return clean_text(img_text)

    # Calcular tamaño de cuerpo para drop cap detection
    body_size = _get_block_body_size(text_blocks)

    # ── Filtro encabezado / pie con protección de tablas ──
    # Problema conocido: tablas en la parte superior de la página pueden caer
    # en la franja header_pct y descartarse. Solución: si un bloque en la zona
    # top tiene ≥2 spans con texto NO itálico y tamaño ≈ body_size, es contenido
    # real (cabecera de tabla o título de artículo) y se conserva.
    def _block_is_real_content_in_header(b: dict) -> bool:
        """Devuelve True si el bloque en zona header parece contenido real
        (cabecera de tabla, título de sección) y NO un header repetido."""
        spans = [sp for ln in b.get("lines", []) for sp in ln.get("spans", [])]
        if not spans:
            return False
        # Si tiene múltiples spans con tamaño ≈ body_size → muy probablemente tabla
        close_to_body = sum(
            1 for sp in spans
            if sp.get("size", 0) > 0 and abs(sp["size"] - body_size) <= body_size * 0.3
        )
        if close_to_body >= 2:
            return True
        # Si el bloque horizontal es ancho (>50% de la página) → tabla o título
        bx0, _, bx1, _ = b["bbox"]
        if (bx1 - bx0) > W * 0.5:
            return True
        return False

    filtered = []
    for b in text_blocks:
        x0, y0, x1, y1 = b["bbox"]

        # Zona de encabezado: descartar SALVO que sea contenido real
        if y1 <= top_y:
            if not _block_is_real_content_in_header(b):
                continue
        # Pie de página
        if y0 >= H * footer_pct:
            continue

        txt = "".join(
            (sp.get("text", "") or "")
            for ln in b.get("lines", [])
            for sp in ln.get("spans", [])
        ).strip()
        txt = clean_text(txt)
        txt = remove_page_number_lines(txt)

        # Descartar bloques con artefactos de fuente no embebida
        if has_font_artifacts(txt):
            continue

        if txt:
            filtered.append({
                "bbox":      b["bbox"],
                "text":      txt,
                "body_size": body_size,
            })

    if not filtered:
        return ""

    # ── Drop caps: fusionar letra capital con el párrafo siguiente ──
    filtered = merge_drop_caps(filtered, body_size)

    # ── Detección de columnas ──
    xs        = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in filtered]
    n_columns = 1 if (max(xs) - min(xs)) < W * 0.25 else 2
    columns   = [[] for _ in range(n_columns)]

    if n_columns == 1:
        columns[0] = filtered
    else:
        mid_x = W / 2
        for b in filtered:
            cx = (b["bbox"][0] + b["bbox"][2]) / 2
            columns[0 if cx < mid_x else 1].append(b)

    # ── Orden de lectura ──
    ordered_blocks = []
    for col in columns:
        ordered_blocks.extend(sorted(col, key=lambda b: b["bbox"][1]))

    # ── Detección de tablas con tolerancia dinámica ──
    # La tolerancia vertical se escala con el tamaño de fuente del bloque
    final_blocks = []
    i = 0
    while i < len(ordered_blocks):
        current = ordered_blocks[i]
        y0_cur  = current["bbox"][1]

        # Tolerancia: ~0.6× el alto típico de línea (estimado como body_size * 1.2)
        row_tol = max(8, body_size * 0.8)

        row_group = [current]
        j = i + 1
        while j < len(ordered_blocks):
            ny0 = ordered_blocks[j]["bbox"][1]
            if abs(ny0 - y0_cur) < row_tol:
                row_group.append(ordered_blocks[j])
                j += 1
            else:
                break

        if len(row_group) > 1:
            # Ordenar celdas de izquierda a derecha
            row_group.sort(key=lambda b: b["bbox"][0])
            final_blocks.append(" | ".join(b["text"] for b in row_group))
            i = j
        else:
            final_blocks.append(current["text"])
            i += 1

    page_text = clean_table_artifacts("\n\n".join(final_blocks))

    # ── Texto en bloques imagen ──
    img_embedded = extract_image_text_native(page)
    if img_embedded:
        page_text += "\n\n" + img_embedded

    # ── OCR de regiones imagen (solo si se pide) ──
    if ocr_images:
        img_ocr = ocr_image_blocks(page, doc, page_index,
                                    dpi=ocr_images_dpi, lang=ocr_images_lang)
        if img_ocr:
            page_text += "\n\n" + img_ocr

    return clean_text(page_text)


# ===================== Pipeline de páginas =====================

def _text_quality_score(text: str, garbage_line_thr: float = 0.20) -> float:
    """
    Calcula un score de calidad 0.0–1.0 para un bloque de texto completo.
    Combina:
    - Ratio de líneas limpias vs basura
    - Score promedio de líneas válidas
    - Longitud neta (bonus por más contenido de calidad)
    Usado para elegir entre texto nativo, Tesseract y PaddleOCR.
    """
    if not text or not text.strip():
        return 0.0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    scores = [_line_garbage_score(ln) for ln in lines]
    clean_lines  = sum(1 for s in scores if s >= garbage_line_thr)
    clean_ratio  = clean_lines / len(lines)
    avg_score    = sum(s for s in scores if s >= garbage_line_thr) / max(clean_lines, 1)

    # Bonus logarítmico por longitud de contenido limpio
    clean_chars = sum(len(ln) for ln, s in zip(lines, scores) if s >= garbage_line_thr)
    import math
    len_bonus = min(0.2, math.log1p(clean_chars / 200) * 0.05)

    return clean_ratio * 0.5 + avg_score * 0.4 + len_bonus


def _select_best_text(candidates: List[str],
                      garbage_line_thr: float = 0.20,
                      debug: bool = False,
                      page_num: int = 0) -> str:
    """
    Elige el mejor texto entre varios candidatos (Tesseract, Paddle, nativo)
    usando _text_quality_score() en lugar de longitud bruta.

    Si el ganador tiene calidad > 0.5 pero hay otro candidato con calidad
    similar y más texto limpio, se fusionan las líneas únicas del segundo.
    """
    if not candidates:
        return ""
    # Filtrar vacíos
    valid = [(c, _text_quality_score(c, garbage_line_thr))
             for c in candidates if c and c.strip()]
    if not valid:
        return ""

    valid.sort(key=lambda x: x[1], reverse=True)
    best_text, best_score = valid[0]

    if debug and page_num:
        for idx, (t, s) in enumerate(valid):
            label = ["tess", "paddle", "native"][idx] if idx < 3 else f"cand{idx}"
            info(f"  [SELECT p.{page_num}] {label}: score={s:.3f} chars={len(t)}")

    # Si hay un segundo candidato con score cercano (±0.1) y más chars limpios,
    # añadir sus líneas únicas al ganador
    if len(valid) > 1:
        second_text, second_score = valid[1]
        if second_score >= best_score - 0.10 and len(second_text) > len(best_text) * 1.3:
            # Fusionar líneas únicas del segundo que tengan buen score
            existing = set(re.sub(r"\W+", "", ln.lower())[:80]
                           for ln in best_text.splitlines() if ln.strip())
            extra_lines = []
            for ln in second_text.splitlines():
                key = re.sub(r"\W+", "", ln.lower())[:80]
                if key and key not in existing:
                    if _line_garbage_score(ln) >= garbage_line_thr:
                        extra_lines.append(ln)
                        existing.add(key)
            if extra_lines:
                best_text = best_text + "\n" + "\n".join(extra_lines)
                if debug:
                    info(f"  [SELECT p.{page_num}] Fusionadas {len(extra_lines)} lineas del segundo candidato")

    return best_text


def parse_force_pages(s: Optional[str]) -> set:
    res = set()
    if not s:
        return res
    for tok in re.split(r"[,\s;]+", s.strip()):
        if not tok:
            continue
        if "-" in tok:
            a, _, b = tok.partition("-")
            if a.isdigit() and b.isdigit():
                res.update(range(min(int(a), int(b)), max(int(a), int(b)) + 1))
        elif tok.isdigit():
            res.add(int(tok))
    return res


def _is_scanned_pdf(doc: fitz.Document, sample_pages: int = 5) -> bool:
    """
    Detecta si un PDF es principalmente escaneado (páginas de imagen sin texto nativo).
    Muestrea hasta sample_pages páginas y devuelve True si la mayoría no tienen texto.
    """
    total = len(doc)
    check = min(sample_pages, total)
    # Muestrear páginas distribuidas (inicio, medio, final)
    indices = [int(total * k / max(check - 1, 1)) for k in range(check)]
    empty = 0
    for i in indices:
        try:
            txt = doc[i].get_text("text").strip()
            if len(txt) < 30:
                empty += 1
        except Exception:
            empty += 1
    return empty / max(check, 1) >= 0.6


def load_pages_with_ocr(pdf_path: str, args) -> Iterable[Tuple[int, str, bool, dict]]:
    forced_pages = parse_force_pages(getattr(args, "force_ocr_pages", ""))
    if getattr(args, "force_ocr_all", False):
        try:
            with fitz.open(pdf_path) as _probe:
                forced_pages |= set(range(1, len(_probe) + 1))
        except Exception:
            pass

    ocr_enabled  = bool(getattr(args, "enable_ocr",  False))
    ocr_images   = bool(getattr(args, "ocr_images",  False))
    img_dpi      = getattr(args, "ocr_images_dpi",  300)
    img_lang     = getattr(args, "ocr_lang",         OCR_LANG)
    footer_pct   = getattr(args, "footer_pct",       FOOTER_PCT)

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        yield -1, str(total), False, {}

        # Auto-detección de PDF escaneado: si no tiene texto nativo
        # en la mayoría de páginas, activar OCR automáticamente.
        auto_ocr = False
        if not ocr_enabled and not getattr(args, "no_auto_ocr", False):
            auto_ocr = _is_scanned_pdf(doc)
            if auto_ocr:
                info("PDF escaneado detectado: activando OCR automáticamente en todas las páginas.")
                info("Tip: usa --enable-ocr para activar esto explícitamente y evitar este aviso.")

        for i in range(total):
            try:
                native_text = extract_text_smart_native(
                    doc, i,
                    header_pct      = getattr(args, "text_header_band", 0.12),
                    footer_pct      = footer_pct,
                    size_tol        = getattr(args, "size_tol",          0.06),
                    x_center_tol    = getattr(args, "x_center_tol",      28),
                    max_x_triggers  = getattr(args, "max_x_triggers",    3),
                    ocr_images      = ocr_images and (ocr_enabled or auto_ocr),
                    ocr_images_dpi  = img_dpi,
                    ocr_images_lang = img_lang,
                ) or ""

                used_ocr     = False
                metrics      = {}
                force_flag   = (i + 1) in forced_pages
                heuristic_f  = should_force_ocr(
                    doc[i], native_text,
                    getattr(args, "ocr_min_chars",   60),
                    getattr(args, "ocr_min_words",   10),
                    getattr(args, "ocr_image_ratio", 1.5),
                )
                garbage_flag = is_garbage_text(native_text)

                # Activar OCR si: habilitado explícitamente O auto-detección de PDF escaneado
                do_ocr = (ocr_enabled or auto_ocr) and (force_flag or heuristic_f or garbage_flag)

                if do_ocr:
                    tess_text, metrics = ocr_page(
                        doc, i,
                        dpi               = getattr(args, "ocr_dpi",            OCR_DPI),
                        deskew            = not getattr(args, "no_deskew",       False),
                        clahe             = not getattr(args, "no_clahe",        False),
                        deshadow_en       = not getattr(args, "no_deshadow",     False),
                        denoise           = not getattr(args, "no_denoise",      False),
                        adaptive_th       = not getattr(args, "no_adaptive",     False),
                        gamma             = getattr(args, "gamma",               None),
                        ocr_lang          = getattr(args, "ocr_lang",            OCR_LANG),
                        ocr_psm           = getattr(args, "ocr_psm",             OCR_PSM),
                        ocr_oem           = getattr(args, "ocr_oem",             OCR_OEM),
                        use_osd           = getattr(args, "ocr_osd",             False),
                        keep_numbers      = getattr(args, "ocr_keep_numbers",    True),
                        tiling            = getattr(args, "ocr_tiling",          False),
                        tiles_cols        = getattr(args, "ocr_tiles_cols",      2),
                        tiles_rows        = getattr(args, "ocr_tiles_rows",      5),
                        target_phrase     = (getattr(args, "ocr_target_phrase",  "") or None),
                        debug             = getattr(args, "debug_ocr",           False),
                        # v4.9
                        illumination_mode = getattr(args, "illumination_mode",   "auto"),
                        orient_scoring    = getattr(args, "orient_scoring",      False),
                        garbage_line_thr  = getattr(args, "garbage_line_thr",    0.20),
                    )

                    # PaddleOCR: renderizar y comparar
                    dpi_val = getattr(args, "ocr_dpi", OCR_DPI)
                    mat     = fitz.Matrix(dpi_val / 72, dpi_val / 72)
                    pix     = doc[i].get_pixmap(matrix=mat, alpha=False)
                    pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    paddle_text = ocr_page_paddle(pil_img)

                    # ── Selección por calidad, NO por longitud bruta (fix v4.9.1) ──
                    gl_thr   = getattr(args, "garbage_line_thr", 0.20)
                    best_ocr = _select_best_text(
                        [tess_text, paddle_text, native_text],
                        garbage_line_thr = gl_thr,
                        debug    = getattr(args, "debug_ocr", False),
                        page_num = i + 1,
                    )
                    native_text = best_ocr
                    used_ocr    = True
                    metrics["paddle_chars"] = len(paddle_text)
                    metrics["tess_chars"]   = len(tess_text)

                yield (i + 1, native_text, used_ocr, metrics)

            except Exception as e:
                warn(f"Página {i+1}: error ({e}). Continúo...")
                yield (i + 1, "", False, {})


# ===================== Ingesta Chroma =====================

def _make_chunk_id(source: str, page_num: int, idx: int, text: str) -> str:
    base = f"{source}|{page_num}|{idx}|{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def add_batch_to_chroma(collection, embedder,
                        docs: List[str], metas: List[dict],
                        ids: List[str]) -> int:
    if not docs:
        return 0
    vecs = embedder.encode(docs, normalize_embeddings=True).tolist()
    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
    else:
        existing = set()
        try:
            res = collection.get(ids=ids)
            for _id in (res.get("ids") or []):
                existing.add(_id)
        except Exception:
            pass
        nd, nm, ni, nv = [], [], [], []
        for k, _id in enumerate(ids):
            if _id not in existing:
                nd.append(docs[k]); nm.append(metas[k])
                ni.append(ids[k]);  nv.append(vecs[k])
        if nd:
            collection.add(ids=ni, documents=nd, embeddings=nv, metadatas=nm)
    added = len(docs)
    docs.clear(); metas.clear(); ids.clear()
    gc.collect()
    return added


# ===================== CLI =====================

def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Ingestión PDF → Chroma v4.9 (iluminación avanzada, orientación robusta, tipografía adaptativa)."
    )
    ap.add_argument("pdf_path", help="Ruta al PDF")

    # Extracción nativa
    ap.add_argument("--text-header-band", type=float, default=0.12,
                    help="Franja superior para encabezado (0.08–0.16).")
    ap.add_argument("--footer-pct",       type=float, default=FOOTER_PCT,
                    help="Umbral Y (0–1) para pie de página. Default=0.92.")
    ap.add_argument("--size-tol",         type=float, default=0.06)
    ap.add_argument("--x-center-tol",     type=int,   default=28)
    ap.add_argument("--max-x-triggers",   type=int,   default=3)
    ap.add_argument("--drop-cap-ratio",   type=float, default=2.0,
                    help="Ratio mínimo fuente/cuerpo para detectar letra capital. Default=2.0")

    # OCR página completa
    ap.add_argument("--enable-ocr",       action="store_true", default=False)
    ap.add_argument("--no-auto-ocr",      action="store_true", default=False,
                    help="Desactiva la detección automática de PDFs escaneados.")
    ap.add_argument("--force-ocr-all",    action="store_true")
    ap.add_argument("--force-ocr-pages",  default="",
                    help="Páginas 1-based a forzar OCR, ej.: 1,3,8-11")
    ap.add_argument("--ocr-osd",          action="store_true")
    ap.add_argument("--ocr-dpi",          type=int,   default=OCR_DPI)
    ap.add_argument("--ocr-lang",         default=OCR_LANG)
    ap.add_argument("--ocr-psm",          type=int,   default=OCR_PSM)
    ap.add_argument("--ocr-oem",          type=int,   default=OCR_OEM)
    ap.add_argument("--no-deskew",        action="store_true")
    ap.add_argument("--no-clahe",         action="store_true")
    ap.add_argument("--no-deshadow",      action="store_true")
    ap.add_argument("--no-denoise",       action="store_true")
    ap.add_argument("--no-adaptive",      action="store_true")
    ap.add_argument("--gamma",            type=float, default=None)
    ap.add_argument("--tesseract-exe",    default=TESSERACT_EXE)
    ap.add_argument("--ocr-min-chars",    type=int,   default=60)
    ap.add_argument("--ocr-min-words",    type=int,   default=10)
    ap.add_argument("--ocr-image-ratio",  type=float, default=1.5)
    ap.add_argument("--ocr-keep-numbers", action="store_true", default=True)
    ap.add_argument("--ocr-tiling",       action="store_true")
    ap.add_argument("--ocr-tiles-cols",   type=int,   default=2)
    ap.add_argument("--ocr-tiles-rows",   type=int,   default=5)
    ap.add_argument("--ocr-target-phrase", default="")
    ap.add_argument("--debug-ocr",        action="store_true")

    # ── NUEVOS v4.9 ──
    ap.add_argument("--illumination-mode", default="auto",
                    choices=["auto", "homomorphic", "retinex", "clahe", "combined"],
                    help="Modo de corrección de iluminación. Default=auto (detecta el tipo).")
    ap.add_argument("--orient-scoring",   action="store_true", default=False,
                    help="Prueba las 4 rotaciones principales y elige la mejor. Más lento pero "
                         "muy robusto para páginas con orientación desconocida.")
    ap.add_argument("--garbage-line-thr", type=float, default=0.20,
                    help="Umbral 0–1 para filtrar líneas basura por score. Default=0.20. "
                         "Subir a 0.30–0.35 para PDFs muy ruidosos.")
    ap.add_argument("--no-fallback",      action="store_true", default=False,
                    help="Desactiva el pipeline fallback de tipografía cuando hay mucho garbage.")

    # OCR imágenes embebidas
    ap.add_argument("--ocr-images",       action="store_true", default=False,
                    help="Activa OCR en bloques imagen dentro del PDF (figuras, infografías).")
    ap.add_argument("--ocr-images-dpi",   type=int, default=300,
                    help="DPI para renderizar regiones imagen antes de OCR. Default=300.")

    # Ingesta
    ap.add_argument("--chroma-host",   default=CHROMA_HOST)
    ap.add_argument("--chroma-port",   type=int, default=CHROMA_PORT)
    ap.add_argument("--collection",    default=COLLECTION)
    ap.add_argument("--embed-model",   default=EMBED_MODEL)
    ap.add_argument("--batch",         type=int, default=BATCH)
    ap.add_argument("--max-chars",     type=int, default=MAX_CHARS)
    ap.add_argument("--overlap",       type=int, default=OVERLAP)
    ap.add_argument("--dedup-chunks",  action="store_true", default=DEDUP_CHUNKS)

    # Selftest
    ap.add_argument("--selftest",         action="store_true",
                    help="HEAD/TAIL por página, sin ingestar.")
    ap.add_argument("--selftest-chunks",  action="store_true",
                    help="Muestra chunks tal como se ingestarían (primeros 2 por página).")
    ap.add_argument("--only-page",        type=int, default=None)
    ap.add_argument("--tail-len",         type=int, default=320)
    return ap


# ===================== Selftest =====================

def _print_page_preview(i: int, text: str, tail_len: int,
                        would_ocr: bool, ocr_reason: str):
    head = text[:min(260, len(text))]
    tail = text[-tail_len:] if text else ""
    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write(f"PÁGINA {i+1}  |  chars={len(text)}")
    if would_ocr:
        sys.stdout.write(f"  |  OCR activaría ({ocr_reason})")
    sys.stdout.write(f"\n[HEAD]\n{head}\n\n[TAIL]\n{tail}\n")


def _print_chunks_preview(i: int, chunks: List[str], max_show: int = 2,
                          trunc: int = 300):
    sys.stdout.write(f"\n--- Página {i+1} ---\n")
    for ci, c in enumerate(chunks[:max_show]):
        preview = c[:trunc] + ("..." if len(c) > trunc else "")
        sys.stdout.write(f"[chunk #{ci}] len={len(c)} (<= {MAX_CHARS} esperado)\n")
        sys.stdout.write(preview + "\n")
    if len(chunks) > max_show:
        sys.stdout.write(f"  ... ({len(chunks) - max_show} chunks más)\n")

    # Parejas de solapamiento
    if len(chunks) >= 2:
        sys.stdout.write("\n=== PAREJAS CONSECUTIVAS PARA VER SOLAPE (~150) ===\n")
        sys.stdout.write(f"\n--- Página {i+1} ---\n")
        for ci in range(min(3, len(chunks) - 1)):
            tail_a = chunks[ci][-120:]
            head_b = chunks[ci+1][:120]
            overlap_detected = any(
                w in head_b for w in tail_a.split() if len(w) > 5
            )
            sys.stdout.write(f"\n(i={ci}) tail(A) ≈120 chars:\n» {tail_a}\n")
            sys.stdout.write(f"\n(i={ci}) head(B) ≈120 chars:\n» {head_b}\n")
            sys.stdout.write(
                "☑ Solape detectable (heurístico)\n" if overlap_detected
                else "⚠ No se detectó solape claro (posible efecto de limpieza/espacios)\n"
            )


# ===================== main =====================

def main(args):
    ensure_tesseract(args.tesseract_exe)

    # ── Selftest HEAD/TAIL ──
    if args.selftest or args.selftest_chunks:
        footer_pct = getattr(args, "footer_pct", FOOTER_PCT)
        ocr_enabled = bool(getattr(args, "enable_ocr", False))
        no_auto_ocr = bool(getattr(args, "no_auto_ocr", False))

        with fitz.open(args.pdf_path) as doc:
            pages = range(len(doc))
            if args.only_page is not None:
                if 1 <= args.only_page <= len(doc):
                    pages = [args.only_page - 1]
                else:
                    err("--only-page fuera de rango"); return

            # Auto-detectar PDF escaneado
            auto_ocr = False
            if not ocr_enabled and not no_auto_ocr:
                auto_ocr = _is_scanned_pdf(doc)
                if auto_ocr:
                    info("PDF escaneado detectado — OCR activado automáticamente en selftest.")

            for i in pages:
                text = extract_text_smart_native(
                    doc, i,
                    header_pct      = args.text_header_band,
                    footer_pct      = footer_pct,
                    size_tol        = args.size_tol,
                    x_center_tol    = args.x_center_tol,
                    max_x_triggers  = args.max_x_triggers,
                    ocr_images      = getattr(args, "ocr_images", False),
                    ocr_images_dpi  = getattr(args, "ocr_images_dpi", 300),
                    ocr_images_lang = getattr(args, "ocr_lang", OCR_LANG),
                ) or ""

                # Aplicar OCR si la página está vacía/basura y OCR disponible
                used_ocr = False
                do_ocr   = (ocr_enabled or auto_ocr)
                reasons  = []

                if is_garbage_text(text):     reasons.append("garbage")
                if should_force_ocr(doc[i], text, args.ocr_min_chars,
                                    args.ocr_min_words, args.ocr_image_ratio):
                    reasons.append("heurística")
                forced = parse_force_pages(args.force_ocr_pages)
                if (i+1) in forced: reasons.append("forzada")

                if do_ocr and reasons:
                    info(f"Página {i+1}: aplicando OCR ({', '.join(reasons)})...")
                    tess_text, _ = ocr_page(
                        doc, i,
                        dpi               = getattr(args, "ocr_dpi",            OCR_DPI),
                        deskew            = not getattr(args, "no_deskew",       False),
                        clahe             = not getattr(args, "no_clahe",        False),
                        deshadow_en       = not getattr(args, "no_deshadow",     False),
                        denoise           = not getattr(args, "no_denoise",      False),
                        adaptive_th       = not getattr(args, "no_adaptive",     False),
                        gamma             = getattr(args, "gamma",               None),
                        ocr_lang          = getattr(args, "ocr_lang",            OCR_LANG),
                        ocr_psm           = getattr(args, "ocr_psm",             OCR_PSM),
                        ocr_oem           = getattr(args, "ocr_oem",             OCR_OEM),
                        use_osd           = getattr(args, "ocr_osd",             False),
                        keep_numbers      = getattr(args, "ocr_keep_numbers",    True),
                        tiling            = getattr(args, "ocr_tiling",          False),
                        tiles_cols        = getattr(args, "ocr_tiles_cols",      2),
                        tiles_rows        = getattr(args, "ocr_tiles_rows",      5),
                        target_phrase     = None,
                        debug             = getattr(args, "debug_ocr",           False),
                        # v4.9
                        illumination_mode = getattr(args, "illumination_mode",   "auto"),
                        orient_scoring    = getattr(args, "orient_scoring",      False),
                        garbage_line_thr  = getattr(args, "garbage_line_thr",    0.20),
                    )
                    # PaddleOCR como segundo intento
                    dpi_val = getattr(args, "ocr_dpi", OCR_DPI)
                    mat     = fitz.Matrix(dpi_val / 72, dpi_val / 72)
                    pix     = doc[i].get_pixmap(matrix=mat, alpha=False)
                    pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    paddle_text = ocr_page_paddle(pil_img)

                    best = _select_best_text(
                        [tess_text, paddle_text, text],
                        garbage_line_thr = getattr(args, "garbage_line_thr", 0.20),
                        debug    = getattr(args, "debug_ocr", False),
                        page_num = i + 1,
                    )
                    if _text_quality_score(best) > _text_quality_score(text):
                        text     = best
                        used_ocr = True

                if args.selftest:
                    _print_page_preview(i, text, args.tail_len,
                                        bool(reasons) and do_ocr,
                                        ", ".join(reasons))
                if args.selftest_chunks:
                    chunks = smart_chunk(text, args.max_chars, args.overlap)
                    sys.stdout.write(
                        f"\n=== MUESTRA DE CHUNKS (primeros 2 por página; "
                        f"truncados a 300 chars) ===\n"
                    )
                    _print_chunks_preview(i, chunks)
        return

    # ── Ingesta a Chroma ──
    if chromadb is None or SentenceTransformer is None:
        err("Faltan chromadb / sentence-transformers. Usa --selftest o --selftest-chunks.")
        return

    client = chromadb.HttpClient(
        host=args.chroma_host,
        port=args.chroma_port,
        settings=Settings(allow_reset=True)
    )
    collection = client.get_or_create_collection(
        name=args.collection, metadata={"hnsw:space": "cosine"}
    )
    info(f"Colección OK: {args.collection}")
    info("Cargando modelo de embeddings...")
    embedder = SentenceTransformer(args.embed_model)

    p = Path(args.pdf_path)
    batch_docs, batch_metas, batch_ids = [], [], []
    dedup_map = defaultdict(set) if args.dedup_chunks else None

    pages_with_text = 0
    empty_pages     = 0
    ocr_pages       = 0
    total_chunks    = 0
    skipped_short   = 0
    t0 = time.time()

    for page_num, payload, used_ocr, metrics in load_pages_with_ocr(str(p), args):
        if page_num == -1:
            info(f"Procesando {payload} páginas..."); continue

        if used_ocr:
            ocr_pages += 1
        payload = (payload or "").strip()
        if not payload:
            empty_pages += 1; continue

        pages_with_text += 1
        chunks = smart_chunk(payload, max_chars=args.max_chars, overlap=args.overlap)

        for idx, c in enumerate(chunks):
            if len(c) < 80:
                skipped_short += 1
                continue
            if args.dedup_chunks:
                h = hashlib.sha1(c.encode("utf-8")).hexdigest()
                if h in dedup_map[page_num]:
                    continue
                dedup_map[page_num].add(h)

            chunk_id = _make_chunk_id(p.name, page_num, idx, c)
            batch_docs.append(c)
            batch_metas.append({
                "source":      p.name,
                "page":        page_num,
                "chunk_index": idx,
                "total_chars": len(c),
                "used_ocr":    int(used_ocr),
            })
            batch_ids.append(chunk_id)
            total_chunks += 1

        if len(batch_docs) >= args.batch:
            try:
                added = add_batch_to_chroma(collection, embedder,
                                             batch_docs, batch_metas, batch_ids)
                info(f"Lote añadido: {added} chunks (total {total_chunks})")
            except Exception as e:
                warn(f"Fallo al subir lote (pág.{page_num}): {e}")
                batch_docs.clear(); batch_metas.clear(); batch_ids.clear()
                gc.collect()

    if batch_docs:
        try:
            add_batch_to_chroma(collection, embedder, batch_docs, batch_metas, batch_ids)
        except Exception as e:
            warn(f"Fallo al subir último lote: {e}")

    mins = (time.time() - t0) / 60.0
    print("\n========== RESUMEN ==========")
    print(f"Documento:             {p.name}")
    print(f"Páginas con texto:     {pages_with_text}")
    print(f"Páginas vacías:        {empty_pages}")
    print(f"Páginas con OCR:       {ocr_pages}")
    print(f"Total chunks ingest.:  {total_chunks}")
    print(f"Chunks <80 chars desc: {skipped_short}")
    print(f"Duración:              {mins:.2f} minutos")
    print("=============================\n")


if __name__ == "__main__":
    parser = build_arg_parser()
    args   = parser.parse_args()
    main(args)