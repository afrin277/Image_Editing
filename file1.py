import streamlit as st 
import cv2

st.title("Machine Learning")
st.write("Hello World")

st.write("Deep Learning")

a  = st.number_input("Enter a Number")

img = cv2.imread("img.jpg")

st.image(img)
st.write("Resize it")
Height = st.slider("select the Height",100,500)
Width = st.slider("select the Width",100,500)

img1 = cv2.resize(img,(Width,Height))

# st.text(a)

st.image(img1)
