import os
import pandas as pd 

dataset_dir = './Dataset/'

data_dictionary = {
    'file_name':[],
    'class_name':[]
}
for folder_name in os.listdir(dataset_dir):
    for file_name in os.listdir(os.path.join(dataset_dir, folder_name)):
        data_dictionary['file_name'].append(file_name)
        data_dictionary['class_name'].append(folder_name)

new_dataset = pd.DataFrame.from_dict(data_dictionary)
new_dataset.to_csv('./image_disease_dataset.csv',index=False)
