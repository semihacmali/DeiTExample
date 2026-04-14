# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:23:12 2024

@author: SemihAcmali
"""
import os
import numpy as np
from PIL import Image

# Görüntüyü yükleme
def imageCrop(image_path):
    img = Image.open(image_path)
    
    # Görüntü boyutlarını alma
    width, height = img.size
    
    # Orta noktayı hesaplama
    center_x, center_y = width // 2, height // 2
    
    # Kırpılacak alanın köşe koordinatlarını hesaplama
    crop_size = 224
    half_crop = crop_size // 2
    
    left = center_x - half_crop
    upper = center_y - half_crop
    right = center_x + half_crop
    lower = center_y + half_crop
    
    # Görüntüyü kırpma
    cropped_img = img.crop((left, upper, right, lower))
    return cropped_img





valid_images = [".jpg",".gif",".png",".tga"]
    
subfolders = [ f.path for f in os.scandir("E:\\VİT\\MangoLeafBD\\") if f.is_dir() ]

for subfolder in subfolders:
    newsubfolder = os.path.join("E:\\VİT\\MangoLeafBDCropped\\" ,subfolder.split("\\")[-1])
    if not os.path.exists(newsubfolder):
        os.mkdir(newsubfolder)
    for im in os.listdir(subfolder):
        ext = os.path.splitext(im)[1] # File Extention
        if ext.lower() in valid_images:
            if os.path.exists(os.path.join(newsubfolder,im)):
                continue
            print(im)
            croppedImage = imageCrop(os.path.join(subfolder,im))
            #os.mkdir(path2)
            if croppedImage is None:
                continue
            croppedImage.save(os.path.join(newsubfolder,im))