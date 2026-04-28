import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image
from deepface import DeepFace

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="FC Vision · Detector de Jugadores",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    color: #f0f0f0;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #0d1b2a 0%, #0a0a0f 60%);
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 72px;
    letter-spacing: 4px;
    line-height: 1;
    background: linear-gradient(135deg, #ffffff 0%, #00d4ff 50%, #0066ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0px;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    letter-spacing: 6px;
    text-transform: uppercase;
    color: #00d4ff;
    font-weight: 300;
    margin-bottom: 8px;
}
.hero-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    color: #888;
    font-weight: 300;
    max-width: 500px;
    line-height: 1.7;
}
.upload-card {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px;
    margin: 16px 0;
    position: relative;
    overflow: hidden;
}
.upload-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, #0066ff, #00d4ff, #0066ff);
}
.result-card {
    background: linear-gradient(135deg, #0f1923 0%, #111827 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
}
.player-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 2px;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 4px;
}
.emotion-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.metric-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 36px;
    color: #00d4ff;
    line-height: 1;
}
.metric-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 24px 0;
}
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stFileUploader"] section {
    background: #0f1923;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 24px;
}
.player-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #888;
    margin: 4px;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ── Colores de emoción ────────────────────────────────────────────────────────
EMOTION_CONFIG = {
    "happy":    {"color": "#22c55e", "emoji": "😄", "label": "Feliz"},
    "sad":      {"color": "#3b82f6", "emoji": "😢", "label": "Triste"},
    "angry":    {"color": "#ef4444", "emoji": "😠", "label": "Enojado"},
    "surprise": {"color": "#f59e0b", "emoji": "😲", "label": "Sorprendido"},
    "fear":     {"color": "#a855f7", "emoji": "😨", "label": "Con miedo"},
    "disgust":  {"color": "#84cc16", "emoji": "🤢", "label": "Disgustado"},
    "neutral":  {"color": "#6b7280", "emoji": "😐", "label": "Neutral"},
}

NOMBRES_JUGADORES = {
    0: "Dani Olmo",
    1: "Lewandowski",
    2: "Pedri",
    3: "Raphinha",
    4: "Yamal"
}

# ── Carga del modelo ──────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    if not os.path.exists('modelo_identidad_svm.pkl'):
        return None
    return joblib.load('modelo_identidad_svm.pkl')

clasificador_svm = cargar_modelo()

# ── HERO HEADER ───────────────────────────────────────────────────────────────
col_hero, col_tags = st.columns([2, 1])
with col_hero:
    st.markdown('<p class="hero-subtitle"> </p>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">DETECTOR DE<br>JUGADORES</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-desc">Identifica automáticamente a 5 jugadores del FC Barcelona y analiza su estado emocional usando visión por computadora.</p>', unsafe_allow_html=True)


st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Verificar modelo ──────────────────────────────────────────────────────────
if clasificador_svm is None:
    st.error("⚠ Modelo no encontrado. Asegúrate de que 'modelo_identidad_svm.pkl' esté en la misma carpeta.")
    st.stop()

# ── Layout principal ──────────────────────────────────────────────────────────
col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:11px;color:#00d4ff;text-transform:uppercase;letter-spacing:3px;margin-bottom:12px;">📁 Cargar imagen</p>', unsafe_allow_html=True)
    archivo_subido = st.file_uploader(
        "Sube una foto del partido",
        type=['jpg', 'jpeg', 'png', 'webp'],
        label_visibility="collapsed"
    )
    st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:#444;margin-top:8px;">Formatos admitidos: JPG · PNG · WEBP</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo_subido:
        st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:3px;margin:16px 0 8px 0;">Imagen original</p>', unsafe_allow_html=True)
        st.image(archivo_subido, use_container_width=True)

with col_der:
    if archivo_subido is not None:
        imagen_pil = Image.open(archivo_subido)
        img_array  = np.array(imagen_pil)

        if img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        img_bgr       = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_resultado = img_array.copy()

        with st.spinner("Analizando rostros con IA..."):
            try:
                representaciones = DeepFace.represent(
                    img_path=img_bgr,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False
                )

                if isinstance(representaciones, dict):
                    representaciones = [representaciones]

                resultados = []

                for rep in representaciones:

                    emb       = np.array(rep["embedding"]).reshape(1, -1)
                    pred_nombre = clasificador_svm.predict(emb)[0]
                    proba       = clasificador_svm.predict_proba(emb)[0]
                    confianza   = max(proba)
                    nombre      = pred_nombre.title() if confianza >= 0.5 else "Desconocido"

                    region = rep["facial_area"]
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]

                    margen = 20
                    x1 = max(0, x - margen)
                    y1 = max(0, y - margen)
                    x2 = min(img_bgr.shape[1], x + w + margen)
                    y2 = min(img_bgr.shape[0], y + h + margen)
                    rostro_recortado = img_bgr[y1:y2, x1:x2]
                    rostro_recortado = cv2.resize(rostro_recortado, (224, 224))

                    emocion_resultado = DeepFace.analyze(
                        img_path=rostro_recortado,
                        actions=['emotion'],
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    if isinstance(emocion_resultado, dict):
                        emocion_resultado = [emocion_resultado]

                    emocion_key = emocion_resultado[0].get("dominant_emotion", "neutral")
                    emo_cfg     = EMOTION_CONFIG.get(emocion_key, EMOTION_CONFIG["neutral"])

                    resultados.append({
                        "nombre":    nombre,
                        "emocion":   emocion_key,
                        "emo_cfg":   emo_cfg,
                        "confianza": confianza,
                        "x": x, "y": y, "w": w, "h": h
                    })

                    hex_color = emo_cfg["color"].lstrip("#")
                    color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

                    cv2.rectangle(img_resultado, (x, y), (x + w, y + h), color_rgb, 3)

                    etiqueta = f"{nombre} | {emo_cfg['label']}"

                    (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(img_resultado,
                                  (x, max(0, y - th - 12)),
                                  (x + tw + 10, y),
                                  (0, 0, 0),   # fondo negro
                                  -1)

                    cv2.putText(img_resultado, etiqueta,
                                (x + 5, max(0, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (255, 255, 255),  # texto blanco
                                2)

                st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:11px;color:#00d4ff;text-transform:uppercase;letter-spacing:3px;margin-bottom:8px;">Resultado del análisis</p>', unsafe_allow_html=True)
                st.image(img_resultado, use_container_width=True)

                n_rostros   = len(resultados)
                n_conocidos = sum(1 for r in resultados if r["nombre"] != "Desconocido")

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{n_rostros}</div><div class="metric-label">Rostros detectados</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{n_conocidos}</div><div class="metric-label">Jugadores identificados</div></div>', unsafe_allow_html=True)
                with m3:
                    emoji_principal = resultados[0]["emo_cfg"]["emoji"] if resultados else "—"
                    st.markdown(f'<div class="metric-box"><div class="metric-value" style="font-size:28px;">{emoji_principal}</div><div class="metric-label">Emoción principal</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:3px;margin-bottom:12px;">Detalle por jugador</p>', unsafe_allow_html=True)

                for r in resultados:
                    color = r["emo_cfg"]["color"]
                    pct   = int(r["confianza"] * 100)
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div>
                                <div class="player-name">{r['nombre']}</div>
                                <span class="emotion-badge"
                                    style="background:{color}22;color:{color};border:1px solid {color}55;">
                                    {r['emo_cfg']['emoji']} {r['emo_cfg']['label']}
                                </span>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-family:'Bebas Neue',sans-serif;font-size:32px;
                                    color:{color};line-height:1;">{pct}%</div>
                                <div style="font-family:'DM Sans',sans-serif;font-size:10px;
                                    color:#555;text-transform:uppercase;letter-spacing:2px;">confianza</div>
                            </div>
                        </div>
                        <div style="height:3px;width:{pct}%;background:linear-gradient(90deg,{color},{color}88);
                            border-radius:2px;margin-top:12px;"></div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div style="background:#1a0a0a;border:1px solid #7f1d1d;border-radius:12px;padding:20px;margin:16px 0;">
                    <p style="font-family:'Bebas Neue',sans-serif;font-size:20px;color:#ef4444;margin:0;">⚠ ERROR AL PROCESAR</p>
                    <p style="font-family:'DM Sans',sans-serif;font-size:13px;color:#888;margin:8px 0 0 0;">{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:400px;display:flex;flex-direction:column;align-items:center;
            justify-content:center;border:1px dashed #1e3a5f;border-radius:16px;background:#0a0f1a;">
            <div style="font-size:64px;margin-bottom:16px;opacity:0.3;">⚽</div>
            <p style="font-family:'Bebas Neue',sans-serif;font-size:24px;color:#333;
                letter-spacing:3px;text-align:center;">SUBE UNA IMAGEN<br>PARA COMENZAR</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;">
    <p style="font-family:'DM Sans',sans-serif;font-size:11px;color:#333;margin:0;">
        FC Vision · Powered by DeepFace + FaceNet + SVM
    </p>
    <p style="font-family:'DM Sans',sans-serif;font-size:11px;color:#333;margin:0;">
    </p>
</div>
""", unsafe_allow_html=True)