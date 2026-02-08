# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 11:17:41 2026

@author: swath
"""

from ultralytics import YOLO
from PIL import Image
model = YOLO('best.pt')


image_path = "TestComicPage2.jpg"
results = model.predict(image_path)

for result in results:
    im_array = result.plot()
    im = Image.fromarray(im_array[...,::-1])
    im.show()

for box in results[0].boxes:
    print("Class:", model.names[int(box.cls)])
    print("Confidence:", box.conf.item())
    print("Coordinates (xyxy):", box.xyxy[0].tolist())
    print("-" * 20)