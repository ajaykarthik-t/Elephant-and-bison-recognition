import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Elephant or Bison Detector",
    page_icon="🐘",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3e4a61;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #718096;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .elephant {
        background-color: #d3f9d8;
        color: #2b8a3e;
        border: 1px solid #8ce99a;
    }
    .bison {
        background-color: #e3fafc;
        color: #0b7285;
        border: 1px solid #99e9f2;
    }
    .neither {
        background-color: #fff5f5;
        color: #c92a2a;
        border: 1px solid #ffc9c9;
    }
    .stImage > img {
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .confidence-bar {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("<h1 class='main-header'>🐘 Elephant or Bison Detector 🦬</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload an image to identify whether it's an elephant or a bison</p>", unsafe_allow_html=True)

# Function to make predictions
def predict_image(img):
    # Resize image to expected size
    img = img.resize((224, 224))
    
    # Preprocess the image
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    return prediction

# Load the pre-trained model
@st.cache_resource
def load_prediction_model():
    try:
        # Try different path options for the model
        import os
        
        # Option 1: Current directory
        if os.path.exists('my_model.keras'):
            return load_model('my_model.keras')
        
        # Option 2: Absolute path from your project structure
        project_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(project_dir, 'my_model.keras')
        
        if os.path.exists(model_path):
            return load_model(model_path)
            
        # Option 3: Go up one directory and try to find the model
        parent_dir = os.path.dirname(project_dir)
        parent_model_path = os.path.join(parent_dir, 'my_model.keras')
        
        if os.path.exists(parent_model_path):
            return load_model(parent_model_path)
            
        # Show more detailed error
        st.error(f"Could not find model file. Searched in:\n- {os.getcwd()}\n- {project_dir}\n- {parent_dir}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Confidence threshold for classification
CONFIDENCE_THRESHOLD = 0.70  # 70% confidence required to classify as elephant or bison

# Main app logic
model = load_prediction_model()

if model:
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        # Example buttons
        st.markdown("### Or try an example:")
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            elephant_example = st.button("Elephant Example")
        with example_col2:
            bison_example = st.button("Bison Example")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process the image
    if uploaded_file is not None:
        # Read the image
        img = Image.open(uploaded_file)
        
        with col2:
            # Display the uploaded image
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        prediction = predict_image(img)
        
        # Get class and confidence
        class_names = ['Bison', 'Elephant']
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index]) * 100
        
        # Determine if the image contains neither an elephant nor a bison
        if confidence < CONFIDENCE_THRESHOLD * 100:
            # Low confidence means it's likely neither
            st.markdown(f"<div class='prediction-box neither'>❓ Neither Elephant nor Bison ({confidence:.2f}%)</div>", 
                        unsafe_allow_html=True)
            
            st.warning("""
            This image doesn't appear to contain either an elephant or a bison.
            The model isn't confident in its prediction.
            Please try uploading a clearer image of an elephant or a bison.
            """)
        else:
            # Create a colored box based on the prediction
            box_class = "elephant" if predicted_class == "Elephant" else "bison"
            animal_emoji = "🐘" if predicted_class == "Elephant" else "🦬"
            
            st.markdown(f"<div class='prediction-box {box_class}'>{animal_emoji} {predicted_class} ({confidence:.2f}%)</div>", 
                        unsafe_allow_html=True)
        
        # Show confidence bars for both classes
        st.markdown("### Confidence Levels:")
        st.progress(float(prediction[0][0]), text=f"Bison: {float(prediction[0][0]) * 100:.2f}%")
        st.progress(float(prediction[0][1]), text=f"Elephant: {float(prediction[0][1]) * 100:.2f}%")
        
        # Display threshold information
        st.info(f"Note: A confidence level of at least {CONFIDENCE_THRESHOLD*100}% is required to classify an image as either an elephant or a bison.")
        
    # Example functionality
    elif elephant_example:
        # This is just a placeholder - replace with path to an actual example image if available
        st.info("In a real app, this would load a pre-stored elephant example image.")
        # If you have example images included in your app:
        # img = Image.open("examples/elephant_example.jpg")
        # Rest of the code would be the same as for uploaded file
        
    elif bison_example:
        # This is just a placeholder - replace with path to an actual example image if available
        st.info("In a real app, this would load a pre-stored bison example image.")
        # If you have example images included in your app:
        # img = Image.open("examples/bison_example.jpg")
        # Rest of the code would be the same as for uploaded file

# Add information section at the bottom
with st.expander("About this app"):
    st.write("""
    This app uses a deep learning model trained to distinguish between elephants and bisons.
    Upload any image, and the model will classify it with a confidence score.
    
    **How it works:**
    1. The uploaded image is resized to 224x224 pixels
    2. The image is normalized by dividing pixel values by 255
    3. A pre-trained deep learning model processes the image
    4. The model outputs probabilities for each class
    5. The class with the highest probability is selected as the prediction
    6. If the confidence is below 70%, the image is classified as "Neither"
    
    **Note:** For best results, use clear images of elephants or bisons with minimal background distraction.
    """)

# Add footer
st.markdown("""
---
<p style="text-align: center; color: #718096; font-size: 0.8rem;">
    Elephant or Bison Detection System | Created with Streamlit
</p>
""", unsafe_allow_html=True)