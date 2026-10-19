import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import sys

# --- FIX for old TensorFlow model formats ---
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

# --- Configuration ---
MODEL_PATH = "converted_keras/keras_model.h5"
LABELS_PATH = "converted_keras/labels.txt"

# --- Load Model and Labels ---
model = load_model(MODEL_PATH, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]
print("Model and labels loaded.")

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Standard Image Processing ---
    img_resized = cv2.resize(frame, (224, 224))
    img_array = np.asarray(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    normalized_img_array = (img_array_expanded.astype(np.float32) / 127.5) - 1

    # --- Live Model Prediction ---
    preds = model.predict(normalized_img_array, verbose=0)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    try:
        class_name = labels[class_idx].split(' ', 1)[1]
    except IndexError:
        class_name = labels[class_idx]
    label_text = f"{class_name}: {confidence:.2%}"

    # --- Display ---
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, "Press [q] to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
