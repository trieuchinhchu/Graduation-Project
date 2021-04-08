# -*- coding: utf-8 -*-

from odoo import models, fields, api
import cv2
import os
import numpy as np
from PIL import Image


class FaceBase(models.Model):
    _name = 'da.facebase'

    employee_id = fields.Many2one(comodel_name='hr.employee', required=True, string='Name')
    path = os.getcwd()

    def get_path(self, path_str):
        path = os.path.join(self.path, path_str)
        if not os.path.isdir(path):
            os.mkdir(path)
        return path


    def open_camera(self):
        num_img = 0
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        dataset_path = self.get_path('dataset')
        while cam.isOpened():
            ret, img = cam.read()
            cv2.imshow('frame', img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (220, 135, 40), 1.5)
                # increment
                num_img += 1
                # save captured
                saved_img = cv2.imwrite('%s/%s.%s.%s.jpg' % (dataset_path, self.employee_id.user_id.login, self.employee_id.id, num_img), gray[y:y+h, x:x+w])
                if not saved_img:
                    raise ValueError("Could not write image")
                cv2.imshow('frame', img)
            if (cv2.waitKey(100) & 0xFF == ord('q')) or num_img>20:
                break

        cam.release()
        cv2.destroyAllWindows()

    def get_images_labels(self, path):
        image_paths = [os.path.join(path, i) for i in os.listdir(path)]
        faces = []
        ids = []
        for image_path in image_paths:
            face_img = Image.open(image_path).convert('L')
            face_np = np.array(face_img, 'uint8')
            faces.append(face_np)
            id = int(os.path.split(image_path)[-1].split('.')[-3])
            ids.append(id)
            cv2.imshow('traning', face_np)
            cv2.waitKey(10)
        return ids, faces

    def training_dataset(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        ids, faces = self.get_images_labels(self.get_path('dataset'))
        recognizer.train(faces, np.array(ids))
        recognizer_path = self.get_path('recognizer')
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        recognizer.save('%s/trainningData.yml' % recognizer_path)
        cv2.destroyAllWindows()

    def get_profile(self, employee_id):
        return self.env['hr.employee'].search([('id', '=', employee_id)])

    def detector(self):
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read('%s/trainningData.yml'%self.get_path('recognizer'))
        # set text style
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        fontcolor = (220, 140, 40)

        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        while cam.isOpened():
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (220, 140, 40), 1.5)
                employee_id, conf = rec.predict(gray[y:y+h, x:x+w])
                employee = self.get_profile(employee_id)
                if employee:
                    cv2.putText(img, "ID: %s" % employee.id,  (x, y+h+30), fontface, fontscale, fontcolor, 1)
                    cv2.putText(img, "Name: %s" % employee.name,  (x, y+h+60), fontface, fontscale, fontcolor, 1)
                else:
                    cv2.putText(img, 'Not Found', (x, y+h+30), fontface, fontscale, fontcolor, 1)
                cv2.imshow('Face', img)
            if cv2.waitKey(1) == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()

