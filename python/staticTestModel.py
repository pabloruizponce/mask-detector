import PIL
import PIL.Image
import pathlib
import tensorflow as tf 
import numpy as np

labels = ['mask', 'no-mask']
input_size = (224, 224)
model = tf.keras.models.load_model('mask_detector.model')

data_dir = pathlib.Path('imgs')
faces_dir = list(data_dir.glob('*'))

faces = []

for face in faces_dir:
    image = PIL.Image.open(str(face)).resize(input_size)
    image = np.array(image)
    image = image[:, :, :3]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    faces.append(image)


faces = np.array(faces)
print(faces.shape)
prediction = model.predict(faces)

for i, face in enumerate(faces_dir):
    print(str(face) + " " + labels[np.argmax(prediction[i])])
