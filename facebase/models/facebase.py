# -*- coding: utf-8 -*-
import base64, pickle, cv2, face_recognition, io, os, time
from dateutil.relativedelta import relativedelta
from odoo import models, fields, api
from datetime import datetime, timezone
from odoo.exceptions import ValidationError
import numpy as np
from PIL import Image


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

    @api.model
    def training(self):
        print('Started encoding')
        count = 0
        data_encoding =[]
        data_names =[]
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

            print('%s %s' %(employee_id.account, count))

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
        # set text style
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
        lst_index, names = self.recognizer(data, frame)

        for ((x, y, w, h), name) in zip(faces, names):
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, '%s'%name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow(frame_name, frame)
        return lst_index

    def recognizer(self, data, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_image = small_frame[:, :, ::-1]
        encodings = face_recognition.face_encodings(rgb_image)
        names = []
        lst_index = []
        # lst_accuracy = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(data['encoding'], encoding)
            name = "Unknown"
            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(data['encoding'], encoding)
            result = min(face_distances)
            if result <= 0.5:
                best_index = np.argmin(face_distances)
                if matches[best_index]:
                    name = data['name'][best_index]
                lst_index.append(best_index)
            names.append(name)
                # lst_accuracy.append(round(100-result*100))
        return lst_index, names

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

