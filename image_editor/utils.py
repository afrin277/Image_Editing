from PIL import Image
import numpy as np
import cv2
import io


def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return np.array(image)


def to_bytes(img):

    # Convert RGB to BGR before saving
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode(".png", img_bgr)

    return buffer.tobytes()