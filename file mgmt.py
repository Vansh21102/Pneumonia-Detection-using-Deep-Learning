import os
import shutil

pneuma = "C:\\Users\\vansh\\OneDrive\\Desktop\\chest_xray\\chest_xray\\train\\PNEUMONIA"

files = os.listdir(pneuma)
bacteria_path = "C:\\Users\\vansh\\OneDrive\\Desktop\\programs\\Pneumonia diagnosis\\train\\bacterial"
virus_path = "C:\\Users\\vansh\\OneDrive\\Desktop\\programs\\Pneumonia diagnosis\\train\\viral"

for img in files:
    if 'bacteria' in img:
        path = os.path.join(pneuma, img)
        bacterial_path = os.path.join(bacteria_path ,img)
        shutil.move(path, bacterial_path)
    else:
        path = os.path.join(pneuma, img)
        viral_path = os.path.join(virus_path, img)
        shutil.move(path, viral_path) 