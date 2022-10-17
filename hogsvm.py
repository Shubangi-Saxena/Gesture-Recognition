path=r"C:\Users\shuba\Desktop\CDSAML\hog+svm\output"

'''import splitfolders
splitfolders.ratio(path, output="output", seed=1337, ratio=(0.7, 0.3))'''

import os
from os import listdir
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import joblib
import cv2
#from google.colab.patches import cv2_imshow
#from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import skimage.feature
from sklearn.metrics import classification_report
import warnings
from PIL import Image,ImageOps
from sklearn import svm
from sklearn.model_selection import GridSearchCV
  

#im = Image.open(r"C:\Users\System-Pc\Desktop\ybear.jpg") 
#im.show() 

warnings.simplefilter(action='ignore', category=FutureWarning)

import skimage.io
from skimage import color
img_list=[]
lab_list=[]
j=0
path=r'C:\Users\shuba\Desktop\CDSAML\hog+svm\output\train'
for i in os.listdir(path):
  print(i,"\n")
  path1= os.path.join(path,i)
  #print(path1)
  path1+="\\"
  #print(path1)
  for images in os.listdir(path1):
    #img=cv2.imread(os.path.join(path1, images),0)
    #img =skimage.io.imread(os.path.join(path1, images))
    #img=color.rgb2gray(img)
    if(images.endswith(".jpg")):
      path2=os.path.join(path1, images)
      #print(path2)
      img=Image.open(path2)
      #img.show()
      dim=(128,256)
      img=img.resize(dim)
      #img=ImageOps.grayscale(img)
      feat=hog(img, orientations=8, pixels_per_cell=(12, 12), cells_per_block=(4, 4),visualize=False,multichannel=True)
      img_list.append(feat)
      lab_list.append(i)
      '''if(j==5):
          break
      j+=1
      print(len(img_list),len(lab_list))'''


#read image
#convert to grayscale

lab_list=np.array(lab_list)
img_list=np.array(img_list)

print('Model starting: ')
#from sklearn.svm import LinearSVC
#model = LinearSVC()
#param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly'],'max_iter':[10,10]}
#svc=svm.SVC(probability=True)
model = svm.SVC(kernel='poly', gamma=0.5, C=0.1,verbose=1,max_iter=10)
#model=GridSearchCV(svc,param_grid)

print('Fitting:')
model.fit(img_list, lab_list)

print('Fitting finished')

img_list1=[]
lab_list1=[]
j=0
path=r'C:\Users\shuba\Desktop\CDSAML\hog+svm\output\val'
for i in os.listdir(path):
  #print(i,"\n")
  path1= os.path.join(path,i)
  #print(path1)
  path1+="\\"
  #print(path1)
  for images in os.listdir(path1):
    #img=cv2.imread(os.path.join(path1, images),0)
    #img =skimage.io.imread(os.path.join(path1, images))
    #img=color.rgb2gray(img)
    if(images.endswith(".jpg")):
      path2=os.path.join(path1, images)
      #print(path2)
      img=Image.open(path2)
      #img.show()
      dim=(128,256)
      img=img.resize(dim)
      #img=ImageOps.grayscale(img)
      feat=hog(img, orientations=8, pixels_per_cell=(12, 12), cells_per_block=(4, 4),visualize=False,multichannel=True)
      img_list1.append(feat)
      lab_list1.append(i)

lab_list1=np.array(lab_list1)
img_list1=np.array(img_list1)

predictions = model.predict(img_list1)

print(predictions)


print(classification_report(lab_list1, predictions))
#model.save("my_model_hogsvm")

