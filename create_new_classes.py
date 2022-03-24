import os
import shutil

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

new_dataset_dir = './New Dataset/'
old_dataset_dir = './Dataset/'

for key, value in class_mapping.items():
    for old_folder in value:
        for file_name in os.listdir(os.path.join(old_dataset_dir,old_folder)):
            shutil.copyfile(os.path.join(old_dataset_dir,old_folder,file_name), os.path.join(new_dataset_dir,key,file_name))