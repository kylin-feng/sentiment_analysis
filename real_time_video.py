from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw, ImageFont#字体转换库

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'#人脸识别器
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'#情绪分类器


# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["愤怒" ,"厌恶","害怕", "开心", "沮丧", "惊讶",
 "正常","未检测到人脸"]
#设置人脸检测器，和情绪分类器

#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)#c初始化摄像头
while True:
    frame = camera.read()[1]#打开摄像头
    #reading the frame
    frame = imutils.resize(frame,width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    #检测人脸，并把人脸数据化成二进制流

    canvas = np.zeros((250, 300, 3), dtype="uint8")#？？
    frameClone = frame.copy()
    if len(faces) > 0:
        #在有脸的情况下
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN

        #从灰度图像中提取人脸的ROI，将其调整为固定的28x28像素，然后准备
        # 通过CNN进行分类的ROI
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]#进行分类
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    else:continue


    def cv2AddChineseText(img, text, position, textColor=(255, 255, 255), textSize=22):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


    #把分析的结果可视化
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]
                # 在画布上绘制标签+概率条
                # emoji_face=情感_faces[np.argmax（preds）]

                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)

                canvas = cv2AddChineseText(canvas,text,(10, (i * 35) + 23))#右侧小窗显示百分比数字
                # cv2.putText(canvas, text, (10, (i * 35) + 23),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                #             (255, 255, 255), 2)
                frameClone = cv2AddChineseText(frameClone,label,(fX, fY - 10))#在图片上显示数字

                # cv2.putText(frameClone,label, (fX, fY - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)#都是一些前端样式的参数

                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),#矩形绘制
                              (0, 0, 255), 2)
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


    cv2.imshow('your_face', frameClone)#显示脸
    cv2.imshow("Probabilities", canvas)#显示概率
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
#关闭，结束