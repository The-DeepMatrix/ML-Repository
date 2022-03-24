import os
import shutil

train_dir = './train/'
valid_dir = './valid/'

for index, folder_name in enumerate(os.listdir(valid_dir)):
    print("on Folder: ", index+1)
    for file_name in os.listdir(os.path.join(valid_dir,folder_name)):
        shutil.copyfile(os.path.join(valid_dir,folder_name,file_name), os.path.join(train_dir,folder_name,file_name))

