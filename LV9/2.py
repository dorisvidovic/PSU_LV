import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

MODEL_PATH = "models/best_gtsrb.keras"
IMG_SIZE   = (48, 48)

CLASS_NAMES = {
     0: "Speed limit (20km/h)",
     1: "Speed limit (30km/h)",
     2: "Speed limit (50km/h)",
     3: "Speed limit (60km/h)",
     4: "Speed limit (70km/h)",
     5: "Speed limit (80km/h)",
     6: "End of speed limit (80km/h)",
     7: "Speed limit (100km/h)",
     8: "Speed limit (120km/h)",
     9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

def predict_image(image_path, top_k=3):
    
    img = keras_image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) 
    img_batch = np.expand_dims(img_array, axis=0)  # (1, 48, 48, 3)

   
    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(img_batch, verbose=0)[0]

   

    top_idx = probs.argsort()[-top_k:][::-1]

    print(f"\nSlika: {image_path}")
    print("Top ", top_k, "predikcija:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. klasa {idx:2d} — {CLASS_NAMES[idx]:45s} "
              f"(p = {probs[idx]:.4f})")

    
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Predikcija: {CLASS_NAMES[top_idx[0]]}\n"
              f"vjerojatnost = {probs[top_idx[0]]:.3f}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return top_idx[0], probs[top_idx[0]]


if __name__ == "__main__":
   
    if len(sys.argv) < 2:
        print("Korištenje: python 03_predict_single_image.py <putanja_do_slike>")
        sys.exit(1)
    predict_image(sys.argv[1], top_k=3)