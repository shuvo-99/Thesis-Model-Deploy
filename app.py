# from tkinter.tix import InputOnly
# from flask import Flask, render_template, request, jsonify
# import os
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# from keras.layers import Input


# app = Flask(__name__)

# # Load your trained CNN model
# # model = tf.keras.models.load_model('D:/JS project/Thesis_Website/model/10_layer_Without_Image_Enhance_1_v2.h5')
# # model = load_model('D:/JS project/Thesis_Website/model/10_layer_Without_Image_Enhance_1_v2.h5')
# # print(model.summary())
# # print(model.layers[0].input_shape)
# # custom_model = load_model('D:/JS project/Thesis_Website/model/10_layer_Without_Image_Enhance_1_v2.h5', compile=False)
# # custom_model.build((None, 224, 224, 3))

# # Load your trained CNN model without the input layer
# model_path = 'D:/JS project/Thesis_Website/model/10_layer_Without_Image_Enhance_1_v2.h5'
# model = load_model(model_path, compile=False)

# # Manually add an Input layer with float64 dtype
# model_input = Input(shape=(224, 224, 3), dtype='float64')
# model_output = model(model_input)
# custom_model = tf.keras.Model(inputs=model_input, outputs=model_output)

# # Define the classes for your model
# classes = ["CNV", "DME", "DRUSEN", "NORMAL"]  # Replace with your actual class labels

# # Define the upload folder
# UPLOAD_FOLDER = 'D:/JS project/Thesis_Website/images/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Function to preprocess the image for the model
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((224, 224))  # Adjust the size based on your model requirements
#     img = np.array(img, dtype=np.float32)
#     img = img / 255.0  # Normalize the pixel values
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # Function to predict the class of the image using the model
# def predict_class(image_path):
#     processed_img = preprocess_image(image_path)
#     print(processed_img.shape)
#     print(processed_img.dtype)

#     prediction = model.predict(processed_img)
#     print(prediction.shape)

#     predicted_class = classes[np.argmax(prediction)]
#     return predicted_class

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     if file:
#         # Save the uploaded file
#         filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filename)

#         # Make predictions
#         predicted_class = predict_class(filename)

#         # Return the result to the webpage
#         return render_template('result.html', image_path=filename, predicted_class=predicted_class)

# if __name__ == '__main__':
#     app.run(debug=True)



# ==========================================================================================================

# from flask import Flask, render_template, request
# import tensorflow as tf
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# #from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50

# app = Flask(__name__)
# model =  tf.keras.models.load_model('D:/JS project/Thesis_Website/model/10_layer_Without_Image_Enhance_1_v2.h5')

# @app.route('/', methods=['GET'])
# def hello_word():
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
# def predict():
#     imagefile= request.files['imagefile']
#     image_path = "./images/" + imagefile.filename
#     imagefile.save(image_path)

#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#     yhat = model.predict(image)
#     label = decode_predictions(yhat)
#     label = label[0][0]

#     classification = '%s (%.2f%%)' % (label[1], label[2]*100)


#     return render_template('index.html', prediction=classification)


# if __name__ == '__main__':
#     app.run(port=3000, debug=True)

# ================================================================================================================

# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from keras.preprocessing.image import img_to_array, load_img
# from keras.models import load_model
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Load the pre-trained model
# model = load_model('D:/JS project/Thesis_Website/model/10_layer_Without_Image_Enhance_1_v2.h5')

# # Define the class labels
# class_labels = ["CNV", "DME", "DRUSEN", "NORMAL"]

# # Function to preprocess the image
# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Route for the home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Route for image prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'})

#     image = request.files['image']
    
#     # Use the original file name for saving
#     image_path = "D:/JS project/Thesis_Website/images/" + secure_filename(image.filename)

#     image.save(image_path)
#     processed_image = preprocess_image(image_path)

#     # Make prediction
#     predictions = model.predict(processed_image)
#     predicted_class = class_labels[np.argmax(predictions)]

#     return jsonify({'prediction': predicted_class})

# if __name__ == '__main__':
#     app.run(debug=True)

# ======================================== ALL GOOD ========================================================================

# from flask import Flask, render_template, request, jsonify, redirect, url_for
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array, load_img
# from werkzeug.utils import secure_filename
# import os
# import tensorflow as tf
# import numpy as np
# from flask import jsonify

# app = Flask(__name__)

# # Set the path for uploaded images
# UPLOAD_FOLDER = 'D:/JS project/Thesis_Website/images'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the pre-trained model
# model = load_model('D:/JS project/Thesis_Website/model/10_layer_Without_Image_Enhance_1_v2.h5')

# # Define the class labels
# class_labels = ["CNV", "DME", "DRUSEN", "NORMAL"]

# # Function to preprocess the image
# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Route for the home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Route for image prediction form
# # Route for image prediction
# # Route for image prediction
# # Update the /predict route
# @app.route('/predict', methods=['POST'])
# def predict():
#     image = request.files.get('image')

#     if not image:
#         return jsonify({'error': 'No image provided'})

#     # Use the original file name for saving
#     image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
#     image.save(image_path)
#     processed_image = preprocess_image(image_path)

#     # Make prediction
#     predictions = model.predict(processed_image)
#     predicted_class = class_labels[np.argmax(predictions)]

#     # Return JSON response
#     return jsonify({'image_path': image_path, 'predicted_class': predicted_class})

# # # Route for the result page
# # @app.route('/result')
# # def result():
# #     image_path = request.args.get('image_path', '')
# #     predicted_class = request.args.get('predicted_class', '')

# #     return render_template('templates/result.html', image_path=image_path, predicted_class=predicted_class)

# if __name__ == '__main__':
#     app.run(debug=True)

# ======================================== Further test ========================================================================


from flask import Flask, render_template, request, jsonify, redirect, url_for
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from flask import jsonify

app = Flask(__name__)

# Set the path for uploaded images
UPLOAD_FOLDER = 'D:/JS project/Thesis_Website/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('D:/JS project/Thesis_Website/model/10_layer_Without_Image_Enhance_1_v2.h5')

# Define the class labels
class_labels = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Set a confidence threshold (adjust as needed)
CONFIDENCE_THRESHOLD = 0.5

# Route for image prediction form
# Route for image prediction
# Route for image prediction
# Update the /predict route
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files.get('image')

    if not image:
        return jsonify({'error': 'No image provided'})

    # Use the original file name for saving
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
    image.save(image_path)
    processed_image = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0, predicted_class_index]

    # Check if the confidence is below the threshold
    if confidence < CONFIDENCE_THRESHOLD:
        print(confidence)
        predicted_class = "Unknown"
    else:
        print(confidence)
        predicted_class = class_labels[predicted_class_index]

    # Return JSON response
    return jsonify({'image_path': image_path, 'predicted_class': predicted_class})

# # Route for the result page
# @app.route('/result')
# def result():
#     image_path = request.args.get('image_path', '')
#     predicted_class = request.args.get('predicted_class', '')

#     return render_template('templates/result.html', image_path=image_path, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)

