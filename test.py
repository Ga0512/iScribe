import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from tqdm import tqdm_notebook 


# Get the image path from the command line arguments
img_path = ''

# Function to extract features from an image using a model
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Can't open image! Ensure that the image path and extension are correct")
    image = image.resize((299, 299))
    image = np.array(image)
    
    # For 4-channel images, convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

# Function to map an integer to its corresponding word in the tokenizer
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate a description for an image using a model and a tokenizer
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

max_length = 33
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
description = generate_desc(model, tokenizer, photo, max_length)

print("\n")
print(description)

# Display the image
plt.imshow(img)
plt.show()
