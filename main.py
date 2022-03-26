import cv2
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.models import  load_model
import numpy as np

model = load_model("best_model.h5")
face_haar_cascade = cv2.CascadeClassifier('wajah.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        sample = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        img_pixels = image.img_to_array(sample)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        max_index = np.argmax(predictions[0])

        emotions = ('marah', 'jijik', 'takut', 'senang', 'sedih', 'kaget', 'flat')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Emosimu lho ', resized_img)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows
