import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import color
from tensorflow.keras import models
import numpy as np

filename = 'test.png'

img_original = mpimg.imread(filename)

img = color.rgb2gray(img_original)

img = resize(img, (28, 28), anti_aliasing=True)

if img.mean() > 0.5:
    img = 1.0 - img

plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.title('Ulaz u mrežu (28×28, invertirano)')
plt.axis('off')
plt.show()

img = img.reshape(1, 28, 28, 1).astype('float32')

model = models.load_model('best_model.keras')

predictions = model.predict(img, verbose=0)

predicted_class = np.argmax(predictions, axis=1)[0]
confidence = predictions[0][predicted_class] * 100

print(f"\n{'='*40}")
print(f"  Predviđena znamenka: {predicted_class}")
print(f"  Pouzdanost:          {confidence:.2f}%")
print(f"{'='*40}\n")

print("Vjerojatnosti po znamenkama:")
for digit, prob in enumerate(predictions[0]):
    bar = '█' * int(prob * 40)
    print(f"  {digit}: {prob*100:6.2f}%  {bar}")