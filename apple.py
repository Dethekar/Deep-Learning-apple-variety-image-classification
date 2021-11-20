import tensorflow.keras
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np

def teachable_machine_classification(img, file):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    # Load the model
    model = tensorflow.keras.models.load_model(file)
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    return np.argmax(prediction)


st.title("Using Computer Vision in Supply Chain Management")
st.header("Image Classification - Apple Variety")
st.text("Upload an image of an apple to identify its variety")
# file upload and handling logic
uploaded_file = st.file_uploader("Choose an Image", type=("jpeg","jpg"))
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    st.image(image, width=300, caption='Uploaded an Image.', use_column_width=False)
    #st.write("")
    st.write("Classifying the image.........please wait")
    st.markdown(""" <style> .font {font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} </style> """, unsafe_allow_html=True)
    label = teachable_machine_classification(image, 'keras_model.h5')
    if label == 0:
        st.markdown(' ## The apple variety is :   <h class="font"> apple_6</h>', unsafe_allow_html=True)
    elif label == 1:
        st.markdown('## The apple variety is : <h class="font">apple_crimson_snow_1</h>', unsafe_allow_html=True)
    elif label == 2:
        st.markdown('## The apple variety is : <h class="font">apple_golden</h>', unsafe_allow_html=True)  # apple_golden_2 and apple_golden_3 seem to be clubbed along with apple_golden_1
    elif label == 3:
        st.markdown('## The apple variety is : <h class="font">apple_granny_smith_1</h>', unsafe_allow_html=True)
    elif label == 4:
        st.markdown('## The apple variety is : <h class="font">apple_hit_1</h>', unsafe_allow_html=True)
    elif label == 5:
        st.markdown('## The apple variety is : <h class="font">apple_pink_lady_1</h>', unsafe_allow_html=True)  
    elif label == 6:
        st.markdown('## The apple variety is : <h class="font">apple_red_1</h>', unsafe_allow_html=True)
    elif label == 7:
        st.markdown('## The apple variety is : <h class="font">apple_red_1</h>', unsafe_allow_html=True)
    elif label == 8:
        st.markdown('## The apple variety is : <h class="font">apple_red_2 or apple_braeburn</h>', unsafe_allow_html=True)   # braeburn and apple_red_2 seem to be clubbed as a single class
    elif label == 9:
        st.markdown('## The apple variety is : <h class="font">apple_red_3</h>', unsafe_allow_html=True)
    elif label == 10:
        st.markdown('## The apple variety is : <h class="font">apple_delicios_1</h>', unsafe_allow_html=True) 
    elif label == 11:
        st.markdown('## The apple variety is : <h class="font">apple_red_yellow_1</h>', unsafe_allow_html=True)
    elif label == 12:
        st.markdown('## The apple variety is : <h class="font">apple_rotten_1</h>', unsafe_allow_html=True) 
    else:
        st.write(label)

