import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from groq import Groq
import os

# Load model
model = YOLO("best.pt")

# Groq API
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ✅ Updated function (English + Proper Urdu)
def get_bilingual_advice(disease):
    prompt = f"""
    You are an expert agriculture specialist.

    Provide advice for the following crop disease:
    {disease}

    Instructions:
    1. First give SHORT and practical advice in ENGLISH.
    2. Then provide its translation in proper URDU (not Roman Urdu).
    3. Keep it simple and useful for farmers in Pakistan.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# Page config
st.set_page_config(page_title="AI Agriculture Assistant", page_icon="🌱", layout="centered")

# Header
st.markdown("<div style='font-family: Noto Nastaliq Urdu;'>...</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: green;'>🌱 AI Smart Agriculture Assistant</h1>", unsafe_allow_html=True)
st.write("Upload a crop image to detect plant disease and get smart AI farming advice.")

# Sidebar
st.sidebar.title("⚙️ Options")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# Upload image
uploaded_file = st.file_uploader("📤 Upload Crop Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Convert image
    img_array = np.array(image)

    # Prediction
    results = model(img_array, conf=confidence)

    # Show output image
    st.subheader("🔍 Detection Result")
    result_img = results[0].plot()
    st.image(result_img, use_column_width=True)

    # Show detected classes
    st.subheader("📊 Detected Diseases")
    detected = set()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        detected.add(label)

        st.write(f"✅ {label} ({conf:.2f})")

    # 🤖 AI Advice (English + Urdu)
    st.subheader("💡 AI Farming Advice (English + Urdu)")

    if len(detected) == 0:
        st.success("Plant is healthy.\n\nپودا صحت مند ہے۔ مناسب دیکھ بھال جاری رکھیں۔")
    else:
        for d in detected:
            with st.spinner(f"{d} advice generating..."):
                advice = get_bilingual_advice(d)

                # Better UI display
                st.markdown(f"### 🌿 {d}")
                st.markdown(advice)

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by Muhammad Aman")