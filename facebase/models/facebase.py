# -*- coding: utf-8 -*-
import base64
import os, time
import pickle
from dateutil.relativedelta import relativedelta
import cv2
import numpy as np
from imutils import paths
import face_recognition
from odoo import models, fields, api
from datetime import datetime, timezone


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
        image_paths = list(paths.list_images('D:\odoo12\server\dataset3'))
        count = 0
        data_encoding =[]
        data_names =[]
        data_ids = []
        for (i, image_path) in enumerate(image_paths):
            count += 1
            name = image_path.split('\\')[-1].split('.')[0:-3]
            id = image_path.split('\\')[-1].split('.')[-3]
            image = cv2.imread(image_path)
            resize_img = cv2.resize(image, (720, 960), interpolation=cv2.INTER_NEAREST)
            rgb_image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
            face_frame = face_recognition.face_locations(rgb_image, model='hog')
            # If training image contains exactly one face
            if len(face_frame) == 1:
                face_encodings = face_recognition.face_encodings(rgb_image, face_frame)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                data_encoding.append(face_encodings)
                data_names.append('.'.join(name))
                data_ids.append(int(id))
            else:
                print(image_path + " was skipped and can't be used for training")

            print('%s %s' %(name, count))

        data = {'encoding': data_encoding, 'name': data_names, 'id': data_ids}
        f = open(f'{self.get_path("recognizer")}/training_data', 'wb')
        f.write(pickle.dumps(data))
        f.close()

    @api.multi
    def recognition(self):
        self.ensure_one()
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        data = pickle.loads(open(f'{self.get_path("recognizer")}/training_data', "rb").read())
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        lst_index = []
        # set text style
        while True:
            a = time.time()
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
                    if len(lst_index) == 10:
                        best_index = max(lst_index, key=lst_index.count)

                        if matches[best_index]:
                            self.create_attendance_log(data['id'][best_index])
                        lst_index = []
                    else:
                        lst_index.append(best_index)
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

    def create_attendance_log(self, employee_id):
        '''
        Tạo attendance log mỗi 2 phút
        :param employee_id:
        :return: attendance_log_id
        '''
        now = datetime.now(tz=timezone.utc)
        att_log_obj = self.env['da.attendance.log']
        exist_log = att_log_obj.sudo().search([('employee_id', '=', employee_id),
                                               ('punch_time', '<=', now),
                                               ('punch_time', '>=', now + relativedelta(minutes=-2))])
        if not exist_log:
            att_log_obj.sudo().create({'employee_id': employee_id,
                                       'punch_time': now})
        return True

