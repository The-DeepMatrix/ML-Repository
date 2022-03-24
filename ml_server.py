from flask import Flask,request,jsonify 
from PIL import Image 
import base64
from flask_cors import CORS
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import pickle
import requests
from googletrans import Translator
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
device = torch.device('cpu')
ml_model = pickle.load(open('xg_boost_model.pkl','rb'))
transforms = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases): 
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases),
                                       nn.LogSoftmax(dim=1))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    
dl_model = ResNet9(3,38)
dl_model.load_state_dict(torch.load('plantdiseaseclassification.pth',map_location=torch.device('cpu')))
dl_model = dl_model.to(device)
dl_model.eval()
image_dir = './images/'
IMG_SIZE = 128

num_to_categories = {0: 'rice',
 1: 'maize',
 2: 'chickpea',
 3: 'kidneybeans',
 4: 'pigeonpeas',
 5: 'mothbeans',
 6: 'mungbean',
 7: 'blackgram',
 8: 'lentil',
 9: 'pomegranate',
 10: 'banana',
 11: 'mango',
 12: 'grapes',
 13: 'watermelon',
 14: 'muskmelon',
 15: 'apple',
 16: 'orange',
 17: 'papaya',
 18: 'coconut',
 19: 'cotton',
 20: 'jute',
 21: 'coffee'}

dl_mapping_dict = {0: 'Apple___healthy',
 1: 'Cherry_(including_sour)___healthy',
 2: 'Tomato___Early_blight',
 3: 'Peach___Bacterial_spot',
 4: 'Corn_(maize)___Common_rust_',
 5: 'Strawberry___healthy',
 6: 'Tomato___Leaf_Mold',
 7: 'Strawberry___Leaf_scorch',
 8: 'Blueberry___healthy',
 9: 'Apple___Black_rot',
 10: 'Potato___Late_blight',
 11: 'Tomato___Late_blight',
 12: 'Soybean___healthy',
 13: 'Pepper,_bell___Bacterial_spot',
 14: 'Raspberry___healthy',
 15: 'Grape___Black_rot',
 16: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 17: 'Corn_(maize)___healthy',
 18: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 19: 'Tomato___Spider_mites Two-spotted_spider_mite',
 20: 'Cherry_(including_sour)___Powdery_mildew',
 21: 'Grape___Esca_(Black_Measles)',
 22: 'Tomato___healthy',
 23: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 24: 'Grape___healthy',
 25: 'Tomato___Septoria_leaf_spot',
 26: 'Squash___Powdery_mildew',
 27: 'Tomato___Target_Spot',
 28: 'Pepper,_bell___healthy',
 29: 'Apple___Cedar_apple_rust',
 30: 'Tomato___Bacterial_spot',
 31: 'Potato___Early_blight',
 32: 'Corn_(maize)___Northern_Leaf_Blight',
 33: 'Orange___Haunglongbing_(Citrus_greening)',
 34: 'Potato___healthy',
 35: 'Peach___healthy',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Apple___Apple_scab'}

class FeatureExtractor(nn.Module):
  def __init__(self):
    super(FeatureExtractor,self).__init__()
    self.feature_extractor = torchvision.models.vgg19(pretrained=True, progress=False) 
    for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
  def forward(self,input_image):
    features = self.feature_extractor(input_image)
    return features

normalization_mean = [0.485, 0.456, 0.406]
normalization_std = [0.229, 0.224, 0.225]
image_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(mean=normalization_mean,
                                                                              std=normalization_std)])
def load_image(image_path,image_transformer,search_img=False):
  if not search_img:
    image = Image.open(os.path.join(image_dir,image_path))
  else:
    image = Image.open(os.path.join(image_path))
  transformed_image = image_transformer(image).unsqueeze(0)
  return transformed_image

def calculate_euclidean_distance(feature_vector_1,feature_vector_2):
  return torch.sum(torch.square(torch.subtract(feature_vector_1,feature_vector_2)))**(0.5)

def calculate_k_similar_images(input_image,search_list,k,model):
  score_list = []
  input_image_tensor = load_image(input_image,image_transforms,search_img=True).to(device)
  input_image_feature = model(input_image_tensor)
  for image in search_list:
    search_image_tensor = load_image(image,image_transforms).to(device)
    search_image_feature = model(search_image_tensor)
    euclidean_distance = calculate_euclidean_distance(input_image_feature,
                                                     search_image_feature)
    score_list.append({'image_id':image,'score':euclidean_distance.item()})
  
  k_images = sorted(score_list,reverse=False, key=lambda x: x['score'])[:k]
  return k_images


@app.route('/best_crop',methods=['GET','POST'])
def best_crop():
    data = dict(request.get_json())
    N = float(data['N'])
    K = float(data['K'])
    P = float(data['P'])
    pH = float(data['pH'])
    rainfall = float(data['rainfall'])
    humidity = float(data['humidity'])
    temp = float(data['temperature'])
    prediction = get_ml_prediction(N,K,P,pH,rainfall,humidity,temp)
    return jsonify({'prediction':prediction})

@app.route('/disease_prediction',methods=['GET','POST'])
def disease_prediction():
    data = dict(request.get_json())
    image_link = data['image_link']
    download_image(image_link)
    dl_preds = test(image_link,dl_model,dl_mapping_dict)
    return jsonify({'prediction':dl_preds})

@app.route('/search_img/', methods=['POST','GET'])
def image_search(image_link):
    try:
        form_data = json.loads(request.data)
        image_link = data['image_link']
        download_image(image_link)
        image_name = image_link
        print("Starting Image Search")
        model = FeatureExtractor()
        model = model.to(device)
        images_to_search = os.listdir('./images/')
        similar_images_score = calculate_k_similar_images(image_name,images_to_search,5,model)
        similar_images = []
        for image in similar_images_score:
          similar_images.append(image['image_id'])
        print(similar_images)
        shutil.copy(image_name,os.path.join('./images/',image_name))
        os.remove(image_name)
        # return jsonify(similar_images=similar_images)
        return similar_images
    except Exception as e:
        print(str(e))
        # return jsonify(message=str(e))
        return str(e)
    # return jsonify(message='image saved to flask')

image_base_url = 'https://ipfs.infura.io/ipfs/'

def download_image(image_id):
    f = open('./'+str(image_id)+'.jpg','wb')
    f.write(requests.get(image_base_url+str(image_id)).content)
    f.close()

def get_ml_prediction(N,K,P,pH,rainfall,humidity,temp):
    arr = np.array([N,P,K,temp,humidity,pH,rainfall]).reshape(1,-1)
    prediction = ml_model.predict(arr)
    return num_to_categories[prediction[0]]

def get_broad_class(prediction):
    class_mapping = {
        'Bacterial Spot':['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Peach___Bacterial_spot','Pepper,_bell___Bacterial_spot','Tomato___Bacterial_spot','Tomato___Septoria_leaf_spot','Tomato___Target_Spot'],
        'Black Measles':['Grape___Esca_(Black_Measles)'],
        'Black Rot':['Apple___Black_rot','Grape___Black_rot',],
        'Blight':['Corn_(maize)___Northern_Leaf_Blight','Potato___Early_blight','Potato___Late_blight','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Tomato___Early_blight','Tomato___Late_blight'],
        'Citrus Greening':['Orange___Haunglongbing_(Citrus_greening)',],
        'Curl Virus':['Tomato___Tomato_Yellow_Leaf_Curl_Virus'],
        'Healthy':['Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___healthy','Corn_(maize)___healthy','Grape___healthy','Peach___healthy','Potato___healthy','Raspberry___healthy','Soybean___healthy','Strawberry___healthy','Tomato___healthy','Pepper,_bell___healthy'],
        'Leaf Scorch':['Strawberry___Leaf_scorch'],
        'Mold':['Tomato___Leaf_Mold'],
        'Mosaic Virus':['Tomato___Tomato_mosaic_virus'],
        'Powdery Mildew':['Cherry_(including_sour)___Powdery_mildew','Squash___Powdery_mildew'],
        'Rust':['Apple___Cedar_apple_rust','Corn_(maize)___Common_rust_'],
        'Scab':['Apple___Apple_scab'],
        'Spider Mite':['Tomato___Spider_mites Two-spotted_spider_mite']
    }

    for key, val in class_mapping.items():
        if prediction in val:
            return key
    
def test(image_link,model, mapping_dict):
    image = Image.open('./'+ image_link+ '.jpg')
    transformed_image = transforms(image)
    transformed_image = transformed_image.unsqueeze(0)
    transformed_image = transformed_image.to(device)
    pred = model(transformed_image)
    _, preds  = torch.max(pred, dim=1)
    broad_class = get_broad_class(mapping_dict[preds[0].item()])
    return broad_class

# x = image_search('QmQTXhtCdJzcCiXutzvzeop8Uekq21gixbpyKoVZxM3aAS')
# print(x)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)
# ml_preds = get_ml_prediction(10,10,10,10,10,10,10)
# print(ml_preds)   