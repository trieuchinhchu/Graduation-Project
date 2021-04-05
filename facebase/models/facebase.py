# -*- coding: utf-8 -*-

from odoo import models, fields, api
import cv2
import os
import numpy as np
from PIL import Image
import time


class FaceBase(models.Model):
    _name = 'da.facebase'

    employee_id = fields.Many2one(comodel_name='hr.employee', string='Name')
    images = fields.Binary(string='Capture')
    path = 'dataSet'

    # def insert_data(self, id, name):
    #     face_base_obj = self.env['da.facebase']
    #     rec = face_base_obj.search([('id', '=', id)])
    #     if rec:
    #         rec.name = name
    #     else:
    #         face_base_obj.create({'id': id,
    #                               'name': name})

    def open_camera(self):
        num_img = 0
        cam = cv2.VideoCapture(0)
        # detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # a = cam.isOpened()
        # print(a)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(2)

        while cam.isOpened():
            ret, img = cam.read()
            cv2.imshow('frame', img)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # faces = detector.detectMultiScale(gray, 1.3, 5)
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #     # increment
            #     num_img += 1
            #     # save captured
            #     cv2.imwrite('dataset/%s.%s.%s.jpg' %(self.employee_id, self.id, num_img), gray[y:y+h, x:x+w])
            #
            #     cv2.imshow('frame', img)

            if (cv2.waitKey(100) & 0xFF == ord('q')) or num_img>20:
                break
        cam.release()
        cv2.destroyAllWindows()

    def get_images_labels(self):
        image_paths = [os.path.join(self.path, i) for i in os.listdir(self.path)]
        faces = []
        ids = []
        for image_path in image_paths:
            face_img = Image.open(image_path).convert('L')
            face_np = np.array(face_img, 'uint8')
            faces.append(face_np)
            id = int(os.path.split(image_path)[-1].split('.')[1])
            ids.append(id)
            cv2.imshow('traning', face_np)
            cv2.waitKey(10)
        return ids, faces

    def training_dataset(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        ids, faces = self.get_images_labels(self.path)
        recognizer.train(faces, np.array(ids))

