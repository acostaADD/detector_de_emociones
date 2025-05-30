import cv2
import os 
import numpy as np
import time

def obtenerModelo(method, facesData, Labels):
    if method == 'EigenFaces': emotion_recongnizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces': emotion_recongnizer = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH': emotion_recongnizer = cv2.face.LBPHFaceRecognizer_create()

    print('Entrenando('+method+')...')
    inicio = time.time()
    emotion_recongnizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time()-inicio
    print('Tiempo de entrenamiento('+method+'):',tiempoEntrenamiento)

    emotion_recongnizer.write('modelo'+method+'.xml')





dataPath = 'C:\\Users\\Alumno.F10KLAB103PC16\\Desktop\\emotions\\data'
emotionList = os.listdir(dataPath)
print('Lista de emociones: ', emotionList)

labels = []
facesData = []
label = 0

for nameDir in emotionList:
    emotionPath = dataPath + '/' + nameDir
    print('Leyendo las imagenes')

    for fileName in os.listdir(emotionPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(emotionPath+ '/' +fileName,0))


        label = label + 1

        obtenerModelo('EigenFaces',facesData, labels)
        obtenerModelo('FisherFaces',facesData, labels)
        obtenerModelo('LBPH',facesData, labels)