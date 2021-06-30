# -*- coding: utf-8 -*-
import base64, pickle, cv2, face_recognition, io, os, time, math
from dateutil.relativedelta import relativedelta
from odoo import models, fields, api
from datetime import datetime, timezone
from odoo.exceptions import ValidationError
import numpy as np
from PIL import Image
from imutils import paths
import random


class FaceBase(models.Model):
    _name = 'da.facebase'

    data = fields.Binary(string='Trained Data')
    training_date = fields.Date(default=fields.Date.today(), string='Date')

    def get_attachments(self):
        query = 'SELECT * FROM hr_employee_facebase_images_rel'
        self._cr.execute(query)
        result = self._cr.fetchall()
        attach_ids = list(map(lambda x: x[1], result))
        emp_ids = list(map(lambda x: x[0], result))
        attachment_ids = self.env['ir.attachment'].sudo().browse(attach_ids)
        employee_ids = self.env['hr.employee'].sudo().browse(emp_ids)
        return employee_ids, attachment_ids

    @api.model
    def capture_from_video(self):
        num_img = 0
        c = 1
        cam = cv2.VideoCapture('D:\\odoo12\\server\\face_data\\ngat.MOV')
        dataset_path = 'D:\\odoo12\\server\\dataset3'
        employee_id = self.env['hr.employee'].search([('name', '=', 'Bùi Thị Ngát')])
        while True:
            ret, img = cam.read()
            if c % 2 == 0:
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_frame = face_recognition.face_locations(rgb_image, model='hog')

                # If training image contains exactly one face
                if len(face_frame) == 1 and face_recognition.face_encodings(rgb_image, face_frame):
                    # increment
                    num_img += 1
                    # save captured

                    saved_img = cv2.imwrite(
                        '%s/%s.%s.%s.jpg' % (dataset_path, employee_id.user_id.login, employee_id.id, num_img), img)
                    if not saved_img:
                        raise ValueError("Could not write image")
                    cv2.imshow('frame', img)
                    print(employee_id.user_id.login)
            c += 1
            if (cv2.waitKey(100) & 0xFF == ord('q')) or num_img >= 30:
                break

    @api.model
    def random_training_data(self):
        dataset_path = 'D:\\odoo12\\server\\dataset3'
        employee_ids = []
        list_img = list(paths.list_images(dataset_path))
        number_of_image = 3
        new_path = 'D:\\odoo12\\evaluate\\data_set_3'
        for i in list_img:
            employee_id = i.split('.')[-3]
            if employee_id not in employee_ids:
                employee_ids.append(employee_id)

        for i in employee_ids:
            random_list = self.get_random_list(number_of_image)
            new_list_img = list(filter(lambda r: r.split('.')[-3] == i and int(r.split('.')[-2]) in random_list, list_img))
            for j in new_list_img:
                img = cv2.imread(j)
                cv2.imwrite('%s\\%s' % (new_path, j.split('\\')[-1]), img)
            print()
            print(new_list_img)
            print('--------------------')
            # self.save_img(new_list_img)

    @api.model
    def five_fold(self):
        data_path = 'D:\\odoo12\\evaluate\\data_set_10'
        list_img = list(paths.list_images(data_path))
        for i in range(0, 5):
            X_test = self.choose_test(list_img)
            Y_test = self.get_y(X_test)
            X_train = list(set(list_img) - set(X_test))
            Y_train = self.get_y(X_train)

            data_training = self.training_2(X_train, Y_train)
            Y_predicted = self.recognition_2(data_training, X_test)
            print(Y_test)
            print(Y_predicted)

    def get_random_list(self, n):
        return random.sample(range(1, 31), n)

    def choose_test(self, list_img):
        imgs = []
        for i in range(15):
            imgs.append(random.choice(list_img))
        return imgs

    def get_y(self, x_test):
        result = []
        for i in x_test:
            result.append(i.split('\\')[-1].split('.')[-5])
        return result

    def training_2(self, X_train, Y_train):
        data_encoding = []
        data_names = []
        for img, name in zip(X_train, Y_train):
            img = cv2.imread(img)
            resize_img = cv2.resize(img, (720, 960), interpolation=cv2.INTER_NEAREST)
            rgb_image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
            face_frame = face_recognition.face_locations(rgb_image, model='hog')
            if len(face_frame) == 1:
                face_encodings = face_recognition.face_encodings(rgb_image, face_frame)[0]
                data_encoding.append(face_encodings)
                data_names.append(name)
        return {'encoding': data_encoding, 'name': data_names}

    def recognition_2(self, data, X_test):
        Y_predicted = []
        for img in X_test:
            img_r = cv2.imread(img)
            resize_img = cv2.resize(img_r, (720, 960), interpolation=cv2.INTER_NEAREST)
            rgb_image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
            # rgb_image = resize_img[:, :, ::-1]
            encodings = face_recognition.face_encodings(rgb_image)
            if not encodings:
                Y_predicted.append('%s --- %s' % (img.split('\\')[-1], 'not encoding'))
            for encoding in encodings:
                name = "Unknown"
                # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(data['encoding'], encoding)
                result = min(face_distances)
                if result <= 0.55:
                    best_index = np.argmin(face_distances)
                    name = data['name'][best_index]
                Y_predicted.append(name)
        return Y_predicted


    @api.model
    def training(self):
        print('Started encoding')
        count = 0
        data_encoding = []
        data_names = []
        data_ids = []
        employee_ids, attachment_ids = self.get_attachments()
        for employee_id, attachment_id in zip(employee_ids, attachment_ids):
            count += 1
            decode_img = base64.b64decode(attachment_id.datas)
            img = Image.open(io.BytesIO(decode_img))
            image = np.asarray(img)
            resize_img = cv2.resize(image, (720, 960), interpolation=cv2.INTER_NEAREST)
            rgb_image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
            face_frame = face_recognition.face_locations(rgb_image, model='hog')
            # If training image contains exactly one face
            if len(face_frame) == 1:
                face_encodings = face_recognition.face_encodings(rgb_image, face_frame)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                data_encoding.append(face_encodings)
                data_names.append(employee_id.account)
                data_ids.append(int(employee_id.id))
            else:
                print(attachment_id.datas_fname + " was skipped and can't be used for training")

            print('%s %s' % (employee_id.account, count))

        data = {'encoding': data_encoding, 'name': data_names, 'id': data_ids}
        facebase_id = self.env.ref('facebase.facebase_trained_data')
        if facebase_id:
            facebase_id.write({'data': base64.b64encode(pickle.dumps(data)),
                               'training_date': fields.Date.today()})

    @api.model
    def recognition(self):
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        facebase_id = self.env.ref('facebase.facebase_trained_data')
        if not facebase_id or not facebase_id.data:
            raise ValidationError('Không tìm thấy data được training!')
        data_decode = base64.b64decode(facebase_id.data)
        data = pickle.loads(data_decode)
        cam_1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cam_2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cam_1.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cam_2.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        lst_index_1 = []
        lst_index_2 = []
        count = 0
        while True:
            ret_1, frame_1 = cam_1.read()
            ret_2, frame_2 = cam_2.read()

            index_1 = self.detect(data, faceCascade, frame_1, 'Check In')

            index_2 = self.detect(data, faceCascade, frame_2, 'Check Out')
            lst_index_1.extend(index_1)
            lst_index_2.extend(index_2)
            count += 1
            if count == 10:
                for i in lst_index_1:
                    if lst_index_1.count(i) >= 6:
                        self.create_attendance_log(data['id'][i], 'check_in')
                for i in lst_index_2:
                    if lst_index_2.count(i) >= 6:
                        self.create_attendance_log(data['id'][i], 'check_out')
                count = 0
                lst_index_1 = lst_index_2 = []
            if cv2.waitKey(1) == ord('q'):
                break
        cam_1.release()
        cam_2.release()
        cv2.destroyAllWindows()

    def detect(self, data, faceCascade, frame, frame_name):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        # lst_index, names = self.recognizer(data, frame)
        lst_index, names, lst_accuracy = self.recognizer(data, frame)

        # for ((x, y, w, h), name) in zip(faces, names):
        for ((x, y, w, h), name, accuracy) in zip(faces, names, lst_accuracy):
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 142, 232), 2)
            # cv2.putText(frame, '%s'% name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, '%s(%s)' % (name, accuracy), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (51, 142, 232), 2)
        cv2.imshow(frame_name, frame)
        return lst_index

    def recognizer(self, data, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_image = small_frame[:, :, ::-1]
        encodings = face_recognition.face_encodings(rgb_image)
        names = []
        lst_index = []
        lst_accuracy = []
        for encoding in encodings:
            name = "Unknown"
            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(data['encoding'], encoding)
            result = min(face_distances)
            if result <= 0.55:
                best_index = np.argmin(face_distances)
                name = data['name'][best_index]
                lst_index.append(best_index)
                lst_accuracy.append(str(round(self.face_distance_to_conf(result) * 100)) + '%')
            names.append(name)
        return lst_index, names, lst_accuracy

    def face_distance_to_conf(self, face_distance, face_match_threshold=0.55):
        if face_distance > face_match_threshold:
            range = (1.0 - face_match_threshold)
            linear_val = (1.0 - face_distance) / (range * 2.0)
            return linear_val
        else:
            range = face_match_threshold
            linear_val = 1.0 - (face_distance / (range * 2.0))
            return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

    def create_attendance_log(self, employee_id, type):
        '''
        Tạo attendance log mỗi 2 phút
        :param employee_id:
        :return: attendance_log_id
        '''
        now = datetime.now(tz=timezone.utc)
        att_log_obj = self.env['da.attendance.log']
        exist_log = att_log_obj.sudo().search([('employee_id', '=', employee_id),
                                               ('check_type', '=', type),
                                               ('punch_time', '<=', now),
                                               ('punch_time', '>=', now + relativedelta(minutes=-2))])
        if not exist_log:
            att_log_obj.sudo().create({'employee_id': employee_id,
                                       'check_type': type,
                                       'punch_time': now})
        return True
