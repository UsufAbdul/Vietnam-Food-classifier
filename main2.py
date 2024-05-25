import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array, load_img


def model_prediction(test_image):
    # Load the pre-trained model
    model = tf.keras.models.load_model("trained_model_2.h5", compile=False)
    
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    # Make predictions
    predictions = model.predict(input_arr)
    
    # Return index of the maximum element
    return np.argmax(predictions)


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

#Main Page
if(app_mode=="Home"):
    st.header("HANOI DELIGHT DETECTOR")
    image_path = "Vietnam_foods.jpeg"
    st.image(image_path)
    st.write("""
    **Welcome to the Vietnamese Culinary Canvas**, where the rich tapestry of Vietnams gastronomy comes to life. Dive into a world where each dish tells a story of tradition, flavor, and innovation. From the bustling streets of Hanoi to the tranquil waters of the Mekong Delta, our system brings you closer to the heart and soul of Vietnamese cuisine. Explore, discover, and savor the vibrant culinary heritage of Vietnam.
    """)

#About Project
elif(app_mode == "About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following Vietnam Food Items:")
    # Using st.write with triple quotes to format the list with line breaks
    st.write("""
    - Banh beo
    - Banh bot loc
    - Banh can
    - Banh canh
    - Banh chung
    - Banh cuon
    - Banh duc
    - Banh gio
    - Banh Khot
    - Banh mi
    - Banh pia
    - Banh tet
    - Banh trang nuong
    - Banh xeo
    - Bun bo Hue
    - Bun dau mam tom
    - Bun mam
    - Bun rieu
    - Bun thit nuong
    - Ca kho to
    - Canh chua
    - Cao lau
    - Chao long
    - Com tam
    - Goi cuon
    - Hu tieu
    - Mi quang
    - Nem chua
    - Pho
    - Xoi xeo
    """)


# Prediction Page
elif(app_mode == "Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")

    # Store the image in session state after first upload
    if test_image is not None:
        st.session_state['test_image'] = test_image

    # Check if there's an image in session state before showing it
    if 'test_image' in st.session_state:
        st.image(st.session_state['test_image'], use_column_width=True)

    # Predict button
    if(st.button("Predict")):
        st.snow()
        # Use the image from session state for prediction
        result_index = model_prediction(st.session_state['test_image'])
        # Reading Labels and Prices
        with open("labels_2.txt") as f:
            content = f.readlines()
        labels_and_prices = {}
        for i in content:
            label, price = i.split(',')  # Assuming each line in labels.txt is in the format "label,price"
            labels_and_prices[label.strip()] = price.strip()
        predicted_label = list(labels_and_prices.keys())[result_index]
        predicted_price = labels_and_prices[predicted_label]
        
        # Store the prediction results in session state
        st.session_state['prediction'] = (predicted_label, predicted_price)

    # Display the prediction results from session state
    if 'prediction' in st.session_state:
        predicted_label, predicted_price = st.session_state['prediction']
        st.success(f"Model is predicting it's a **{predicted_label}** with an estimated price of **{predicted_price}**.")


