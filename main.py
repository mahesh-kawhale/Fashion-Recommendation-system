import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print(f"Error while saving file: {e}")
        return False


# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    # Save the file
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        try:
            display_image = Image.open(uploaded_file)
            st.image(display_image)
        except Exception as e:
            st.error(f"Error displaying image: {e}")

        # Extract features from the image
        image_path = os.path.join("uploads", uploaded_file.name)
        try:
            features = feature_extraction(image_path, model)
        except Exception as e:
            st.error(f"Error extracting features: {e}")

        # Get recommendations based on features
        try:
            indices = recommend(features, feature_list)
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")

        # Show recommended images in 5 columns
        if indices is not None and len(indices) > 0 and len(indices[0]) >= 5:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(filenames[indices[0][0]])
            with col2:
                st.image(filenames[indices[0][1]])
            with col3:
                st.image(filenames[indices[0][2]])
            with col4:
                st.image(filenames[indices[0][3]])
            with col5:
                st.image(filenames[indices[0][4]])
        else:
            st.warning("Not enough recommendations to display.")
    else:
        st.error("Some error occurred in file upload")


