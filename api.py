import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from PIL import Image
from flask import Flask, request, jsonify
import pycurl
import json

def download_image(url, save_as):
    with open(save_as, 'wb') as file:
        curl = pycurl.Curl()
        curl.setopt(curl.URL, url)
        curl.setopt(curl.WRITEDATA, file)
        curl.perform()
        curl.close()



class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
model = models.load_model('image_classifier.keras')

app = Flask(__name__)

@app.route("/get-classification", methods=["POST"])
def get_classification():
    data = request.get_json()
    url_data = data[0]["url"]
    if not url_data.endswith(".jpg"):
        return jsonify('Image not supporeted'), 415
    else:
        image_url = url_data
        save_as = 'image.jpg'

        download_image(image_url, save_as)

        image = Image.open('image.jpg')
        new_image = image.resize((32, 32))
        new_image.save('newimage.jpg')
        print('image rezised')

        img = cv.imread('newimage.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        plt.imshow(img, cmap=plt.cm.binary)

        prediction = model.predict(np.array([img]) / 255)
        index = np.argmax(prediction)
        return jsonify(f'Prediction is {class_names[index]}'), 200











if __name__ == "__main__":
    app.run(debug=True)