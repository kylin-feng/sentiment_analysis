import pandas as pd#数据分析库
import cv2#图像识别库
import numpy as np#科学计算库


dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size=(48,48)
#预处理数据集
def load_fer2013():
        data = pd.read_csv(dataset_path)#读取数据集
        pixels = data['pixels'].tolist()#转化为列表
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:#遍历pixels
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]#分割
            face = np.asarray(face).reshape(width, height)#reshape成矩阵格式
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x