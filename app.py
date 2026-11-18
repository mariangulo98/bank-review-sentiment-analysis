import streamlit as st
import pandas as pd
import torch
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# confugracion
MODEL_PATH = "mariangulo98/bank-sentimentt" 
#"best_distilbert_model" si fuera local
# repo hugging face:https://huggingface.co/mariangulo98/bank-sentimentt

st.set_page_config(
    page_title="Analisis de Sentimiento en Reseñas Bancarias",
    page_icon="🏦",
    layout="wide"
)

# estilos

st.markdown("""
<style>

.main {
    background-color: #f4f6f8;
}

.stButton > button {
    background-color: #0f8a5f;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0b6848;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    border-left: 5px solid #0f8a5f;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

.title-bank {
    color: #1d3557;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# sidebar para presentacion del proyecto
with st.sidebar:

    st.markdown("## Proyecto Final")

    st.markdown("""
### Modelo
DistilBERT

### Clasificación
- Negativo
- Positivo

### Dataset
Bank Reviews Dataset Kaggle

### Autor
María Jesús Angulo
""")

    st.divider()

    st.info(
        "Las reseñas deben ingresarse en inglés porque el modelo fue entrenado únicamente con texto en inglés."
    )

# carga de modelo final
@st.cache_resource
def load_model():

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_PATH
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()


def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    negative_prob = probabilities[0][0].item()

    positive_prob = probabilities[0][1].item()

    sentiment = (
        "Negativo"
        if prediction == 0
        else "Positivo"
    )

    confidence = max(
        negative_prob,
        positive_prob
    )

    return (
        sentiment,
        confidence,
        negative_prob,
        positive_prob
    )

# titulo y encabezado
st.markdown(
    """
    <div style="text-align:center;">
        <h1 class="title-bank">
        Análisis de Sentimiento en Reseñas Bancario
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Clasificación automática de reseñas utilizando DistilBERT."
)

# opcion con analisis de 1 reseña
col1, col2 = st.columns([1, 1])

with col1:

    st.subheader("Análisis Individual")

    review = st.text_area(
        "Ingrese una reseña en inglés",
        height=220,
        placeholder="""
Example:

The customer service was excellent and my issue was resolved quickly.
"""
    )

    analyze_btn = st.button(
        "Analizar Sentimiento",
        use_container_width=True
    )

with col2:

    st.subheader("📊 Resultado")

    if analyze_btn and review.strip():

        sentiment, confidence, neg_prob, pos_prob = (
            predict_sentiment(review)
        )

        card_color = (
            "#dff5e8"
            if sentiment == "Positivo"
            else "#fde2e2"
        )

        title_color = (
            "#0f8a5f"
            if sentiment == "Positivo"
            else "#d62828"
        )

        icon = (
            "✅"
            if sentiment == "Positivo"
            else "❌"
        )

        st.markdown(
            f"""
            <div style="
            background-color:{card_color};
            padding:25px;
            border-radius:15px;
            box-shadow:0px 2px 10px rgba(0,0,0,0.10);
            ">
            <h2 style="color:{title_color};">
            {icon} Sentimiento {sentiment}
            </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Confianza",
            f"{confidence:.2%}"
        )

        st.markdown("### Probabilidades")

        st.write(
            f"🟢 Positivo: {pos_prob:.2%}"
        )

        st.progress(float(pos_prob))

        st.write(
            f"🔴 Negativo: {neg_prob:.2%}"
        )

        st.progress(float(neg_prob))

# opcion con carga de csv con reseñas, deben estar de preferencia entre comillas y en ingles
st.divider()

st.subheader(
    "📁 Análisis Masivo de Reseñas desde CSV"
)

uploaded_file = st.file_uploader(
    "Seleccione un archivo CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Vista previa")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    text_column = st.selectbox(
        "Seleccione la columna con las reseñas",
        list(df.columns)
    )

    process_btn = st.button(
        "🚀 Procesar CSV"
    )

    if process_btn:

        results = []

        progress = st.progress(0)

        total = len(df)

        for i, review in enumerate(df[text_column]):

            sentiment, confidence, neg_prob, pos_prob = (
                predict_sentiment(
                    str(review)
                )
            )

            results.append(
                {
                    "review": review,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 4),
                    "positive_probability": round(pos_prob, 4),
                    "negative_probability": round(neg_prob, 4)
                }
            )

            progress.progress(
                (i + 1) / total
            )

        results_df = pd.DataFrame(results)

        positive_count = (
            results_df["sentiment"] == "Positivo"
        ).sum()

        negative_count = (
            results_df["sentiment"] == "Negativo"
        ).sum()

        # resumen dashboard
        st.subheader(
            "📊 Resumen Ejecutivo"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Total de Reseñas",
                len(results_df)
            )

        with c2:
            st.metric(
                "Positivas",
                positive_count
            )

        with c3:
            st.metric(
                "Negativas",
                negative_count
            )

       
        chart_df = pd.DataFrame({
            "Sentimiento": [
                "Positivo",
                "Negativo"
            ],
            "Cantidad": [
                positive_count,
                negative_count
            ]
        })

        fig_pie = px.pie(
            chart_df,
            names="Sentimiento",
            values="Cantidad",
            hole=0.45,
            color="Sentimiento",
            color_discrete_map={
                "Positivo": "#0f8a5f",
                "Negativo": "#d62828"
            }
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

        # nube de palabras

        st.subheader(
            "Nube de Palabras"
        )

        text = " ".join(
            results_df["review"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        if text.strip():

            try:

                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color="white",
                    colormap="Greens"
                ).generate(text)

                fig_wc, ax = plt.subplots(
                    figsize=(12, 6)
                )

                ax.imshow(
                    wordcloud,
                    interpolation="bilinear"
                )

                ax.axis("off")

                st.pyplot(fig_wc)

            except Exception as e:

                st.warning(
                    f"No fue posible generar la nube de palabras: {e}"
                )

     
        st.subheader(
            "Distribución de Confianza"
        )

        fig_hist = px.histogram(
            results_df,
            x="confidence",
            nbins=20,
            color_discrete_sequence=["#0f8a5f"]
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )


        st.subheader(
            "📋 Resultados Detallados"
        )

        results_view = results_df[
            [
                "sentiment",
                "confidence",
                "positive_probability",
                "negative_probability",
                "review"
            ]
        ]

        st.dataframe(
            results_view,
            use_container_width=True,
            height=400
        )

        # descargar CSV
        csv = (
            results_view
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="📥 Descargar Resultados",
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv",
            use_container_width=True
        )