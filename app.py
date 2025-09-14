import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf

# Load the saved model
model = keras.models.load_model('happysadmodel.h5')

def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    return prediction

def main():
    st.title('Happy or Sad Image Classifier')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        prediction = predict(image)

        # Case 1: model outputs a single value (binary classification)
        if prediction.shape[1] == 1:
            score = prediction[0][0]
            # Convert ReLU outputs into a pseudo-probability
            prob_sad = score / (score + 1)
            prob_happy = 1 - prob_sad
        else:
            # Case 2: model outputs 2 neurons (Happy, Sad) → apply softmax
            probs = tf.nn.softmax(prediction[0]).numpy()
            prob_happy, prob_sad = probs[0], probs[1]

        sentiment = "Happy" if prob_happy > prob_sad else "Sad"
        confidence = max(prob_happy, prob_sad)

        st.write(f"Prediction: **{sentiment}**")
        st.write(f"Confidence: {confidence:.2f}")
        st.write(f"Happy: {prob_happy:.2f}")
        st.write(f"Sad: {prob_sad:.2f}")

if __name__ == '__main__':
    main()
