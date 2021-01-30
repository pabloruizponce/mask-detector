import tensorflow as tf 
import numpy as np
from cv2 import cv2

labels = ['mask', 'no-mask']
input_size = (224, 224)
model = tf.keras.models.load_model('mask_detector.model')

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
position = (5,15)
fontScale = 0.7
red = (0, 0, 255)
green = (0, 255, 0)

actual_pred = 0
diff_pred = 0

while True:
    ret, frame = cam.read()
    frame = frame[:, 80:560]

    im = cv2.resize(frame, input_size)
    im = np.array(im)
    im = tf.keras.applications.mobilenet_v2.preprocess_input(im)
    im = tf.expand_dims(im, axis=0)

    pred = model.predict(im)

    if np.argmax(pred[0]) != actual_pred:
        diff_pred += 1
        if diff_pred > 5:
            actual_pred = np.argmax(pred[0])
            diff_pred = 0
    elif diff_pred > 0:
        diff_pred -= 0


    cv2.putText(frame, labels[actual_pred], 
        position, 
        font, 
        fontScale,
        green if actual_pred == 0 else red,
        1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()

cv2.destroyAllWindows()