from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True
print("[+] Setup model")
base_model = VGG16(weights='imagenet', include_top=True)
out = base_model.get_layer("fc2").output
model = Model(inputs=base_model.input, outputs=out)


def save_features(save_path, features):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[+]Save extracted feature to file : ", save_path)
    np.save(save_path, features)

def extract_features(path):
    img = image.load_img(path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features

def extract_features_from_src(src):    
    files = os.listdir(src)
    for i,file in enumerate(files):
        img_path = os.path.join(src, file)
        print("[+] Read image  : ", img_path," id : ", i)
        if os.path.isfile(img_path) and img_path.find(".jpg") != -1:            
            save_path = img_path.replace("images", "features/vgg16_fc2").replace(".jpg", ".npy")                                  
            print("[+] Extract feature from image : ", img_path)
            features = extract_features(img_path)
            save_features(save_path, features)
        

if __name__=="__main__":
    src = sys.argv[1]
    print(src)
    extract_features_from_src(src)


