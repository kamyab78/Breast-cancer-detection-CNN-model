import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define and train the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the model
model.save('breast_cancer_detection_model.h5')

saved_model_path = 'breast_cancer_detection_model.h5' 
model = tf.keras.models.load_model(saved_model_path)

# Function to predict whether the image has breast cancer or not
def predict_cancer_from_image(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the pixel values

    prediction = model.predict(img_array)
    print(prediction)
    if prediction[0][0] < 0.5:
        return "The image does not show signs of breast cancer."
    else:
        return "The image shows signs of breast cancer."

img_path_to_test = 'path_to_test_image'  # Replace with the actual path to the image you want to test
result = predict_cancer_from_image(img_path_to_test, model)
print(result)
