import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
from groq import Groq
import os

# Load model
model = YOLO("best.pt")

# Groq API Setup
# Hugging Face mein Settings > Secrets mein GROQ_API_KEY lazmi add karein
api_key = "use your api key"
client = Groq(api_key=api_key)

def get_bilingual_advice(disease):
    prompt = f"""
    You are an expert agriculture specialist.
    Provide advice for the following crop disease: {disease}
    1. Short practical advice in ENGLISH.
    2. Translation in proper URDU (Unicode).
    Keep it simple for Pakistani farmers.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error fetching AI advice: {str(e)}"

def predict_and_advise(img, conf_threshold):
    if img is None:
        return None, "Please upload an image first."

    # YOLO Prediction
    results = model(img, conf=conf_threshold)
    
    # Plot results on image
    res_plotted = results[0].plot()
    output_image = Image.fromarray(res_plotted[:, :, ::-1])  # BGR to RGB conversion

    # Extract detected diseases
    detected = set()
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        detected.add(label)

    # Generate Advice
    if not detected:
        full_advice = "✅ Plant is healthy. / پودا صحت مند ہے۔"
    else:
        advice_list = []
        for d in detected:
            advice_list.append(f"### 🌿 Disease: {d}\n{get_bilingual_advice(d)}")
        full_advice = "\n\n---\n\n".join(advice_list)

    return output_image, full_advice

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌱 AI Smart Agriculture Assistant")
    gr.Markdown("Upload a crop image to detect diseases and get AI farming advice in English & Urdu.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Crop Image")
            conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, label="Confidence Threshold")
            btn = gr.Button("Analyze Plant", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="Detection Result")
            output_text = gr.Markdown(label="AI Advice")

    btn.click(fn=predict_and_advise, inputs=[input_img, conf_slider], outputs=[output_img, output_text])
    
    gr.Markdown("---")
    gr.Markdown("👨‍💻 Developed by Muhammad Aman")

demo.launch()
