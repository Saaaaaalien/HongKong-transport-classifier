
import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your saved model (use the correct path)
model = tf.keras.models.load_model('best_model.keras', compile = False)

# Define class names (MATCH YOUR ACTUAL CLASSES)
class_names = ['minibus', 'taxi', 'tram'] 

def predict_image(img):
    """
    Predict transport type from uploaded image
    """
    # Resize and preprocess
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get predicted class and confidence
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = np.max(predictions[0]) * 100
    
    # Return as dictionary for Gradio Label output
    result = {class_name: float(predictions[0][i]) for i, class_name in enumerate(class_names)}
    
    # Also print to console for debugging
    print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    print(f"All probabilities: {result}")
    
    return result

# Create the interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Transport Photo"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Hong Kong Transport Classifier",
    description="Upload a photo of a minibus, taxi, or tram to identify which type of public transport it is.",
    examples=[]
)

# Launch with share=True for temporary public link
iface.launch(share=True)
