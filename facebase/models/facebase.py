# -*- coding: utf-8 -*-
import base64
import glob
import os
import pickle
from datetime import datetime

import cv2
import numpy as np
from imutils import paths
import face_recognition
from odoo import models, fields, api


class FaceBase(models.Model):
    _name = 'da.facebase'

    employee_id = fields.Many2one(comodel_name='hr.employee', required=True, string='Name')
    file_path = fields.Char(string='File path')

    path = os.getcwd()
    data = []

    def get_path(self, path_str):
        path = os.path.join(self.path, path_str)
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    def capture_from_video(self):
        self.ensure_one()
        num_img = 0
        c = 1
        cam = cv2.VideoCapture(self.file_path)
        dataset_path = self.get_path('dataset3')
        while True:
            ret, img = cam.read()
            if c % 5 == 0:
                # increment
                num_img += 1
                # save captured
                saved_img = cv2.imwrite(
                    '%s/%s.%s.%s.jpg' % (dataset_path, self.employee_id.user_id.login, self.employee_id.id, num_img), img)
                if not saved_img:
                    raise ValueError("Could not write image")
                cv2.imshow('frame', img)
            c += 1
            if (cv2.waitKey(100) & 0xFF == ord('q')) or num_img >= 30:
                break

    @api.multi
    def encoding_data(self):
        self.ensure_one()
        print('Started encoding')
        count = 0
        for (i, image_path) in enumerate(self.image_paths):
            count += 1
            name = image_path.split('\\')[-1].split('.')[-5]

            image = cv2.imread(image_path)
            resize_img = cv2.resize(image, (720, 960), interpolation=cv2.INTER_NEAREST)
            rgb_image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
            face_frame = face_recognition.face_locations(rgb_image, model='hog')
            # If training image contains exactly one face
            if len(face_frame) == 1:
                face_encodings = face_recognition.face_encodings(rgb_image, face_frame)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                self.data_encoding.append(face_encodings)
                self.data_names.append(name)
            else:
                print(image_path + " was skipped and can't be used for training")

            print('%s %s' %(name, count))

        data = {'encoding': self.data_encoding, 'name': self.data_names}
        f = open(f'{self.get_path("recognizer")}/training_data', 'wb')
        f.write(pickle.dumps(data))
        f.close()

    def recognition(self):
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        data = pickle.loads(open(f'{self.get_path("recognizer")}/training_data', "rb").read())
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # set text style
        while True:
            ret, frame = capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(60, 60),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
            names = []
            rgb_image = small_frame[:, :, ::-1]
            encodings = face_recognition.face_encodings(rgb_image)
            for encoding in encodings:
                matches = face_recognition.compare_faces(data['encoding'], encoding)
                name = "Unknown"
                # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(data['encoding'], encoding)
                if min(face_distances) <= 0.5:
                    best_index = np.argmin(face_distances)
                    if matches[best_index]:
                        name = data['name'][best_index]
                names.append(name)

            for ((x, y, w, h), name) in zip(faces, names):
                # draw the predicted face name on the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
