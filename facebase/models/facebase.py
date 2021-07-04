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
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from time import time
import logging


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from collections import Counter


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

    def get_random_list(self, n):
        return random.sample(range(1, 101), n)

    @api.model
    def k_fold(self):
        data_paths = ['D:\\odoo12\\training\\training_1', 'D:\\odoo12\\training\\training_2', 'D:\\odoo12\\training\\training_3']
        N = [5]
        N_set = [90, 150, 300]
        sets = []
        for n in N:
            for index, n_set in enumerate(N_set):
                list_img = list(paths.list_images(data_paths[index]))
                list_accuracy = []
                list_y_test = []
                list_y_predict = []
                list_lost = []
                for i in range(0, n):
                    X_test = self.choose_test(list_img, n_set)
                    Y_test = self.get_y(X_test)
                    X_train = list(set(list_img) - set(X_test))
                    Y_train = self.get_y(X_train)
                    Y_test = self.filter_y_test(Y_test, Y_train)
                    Y_predicted = self.training_2(X_train, Y_train, self.get_cv_number(Y_train), X_test)
                    # Y_predicted = self.recognition_2(data_training, X_test)
                    accuracy = self.get_evaluate(Y_test, Y_predicted)
                    list_accuracy.append(accuracy)
                    print('%s Fold - Set %s - Fold %s ------------------------' % (str(n), str(index+1), str(i+1)))
                    print('Accuracy %s' % str(accuracy))
                    list_y_test.append(Y_test)
                    list_y_predict.append(Y_predicted)

                best_accurate = max(list_accuracy)
                b_index = list_accuracy.index(best_accurate)
                z = len(list_y_test[b_index]) - len(list_y_predict[b_index])
                if z > 0:
                    a = set(list_y_test[b_index]) - set(list_y_predict[b_index])
                    rm = a[:z]
                    for i in rm:
                        list_y_test[b_index].remove(i)
                print('Best accurate %s'%best_accurate)
                accurate = round(sum(list_accuracy)/len(list_accuracy), 2)
                sets.append(accurate)
                print('================================')
                print(list_lost)
                print('AVG Accurate: %s' % str(accurate))
        #
        # # sets = {1: [97.33, 97.2, 96.9], 2: [98.22, 99.6, 99.2], 3: [99.11, 99.4, 99.33]}
        # set_1 = np.array(sets[1])
        # set_2 = np.array(sets[2])
        # set_3 = np.array(sets[3])

        self.show_report_chart(sets)
        # self.show_performance_chart(np.array(time_sets[0]))

    def show_confusion_matrix(self, cm, labels):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    @api.model
    def k_fold_svm(self):
        N = [3, 5, 10]
        sets = {1: [],
                2: [],
                3: []}

        for index,i in enumerate(N):
            # for index, j in enumerate(N):
            #     list_accuracy = []
            #     count = 0
            #     for i in range(0, j):
            Y_test, Y_predicted = self.svm(i)
            accuracy = self.get_evaluate(Y_test, Y_predicted)
            print('================================')
            print('Dataset %s' %str(index+1))
            print('Accurate: %s' % str(accuracy))
            sets[index+1].append(accuracy)

        set_1 = np.array(sets[1])
        set_2 = np.array(sets[2])
        set_3 = np.array(sets[3])
        self.show_report_chart(sets)
            # for index, n_set in enumerate(N_set):
            #     list_accuracy = []
            #
            #     for i in range(0, n):
            #         X_test = self.choose_test(list_img, n_set)
            #         Y_test = self.get_y(X_test)
            #         X_train = list(set(list_img) - set(X_test))
            #         Y_train = self.get_y(X_train)
            #         Y_test = self.filter_y_test(Y_test, Y_train)
            #         Y_predicted = self.training_2(X_train, Y_train, cv=self.get_cv_number(Y_train), X_test=X_test)
            #         # Y_predicted = self.recognition_svm(data_training, X_test)
            #         lost_indexes, accuracy = self.get_evaluate(X_test, Y_test, Y_predicted)
            #         list_accuracy.append(accuracy)
            #         print(lost_indexes)
            #
            #         print('%s Fold - Set %s - Fold %s ------------------------' % (str(n), str(index+1), str(i+1)))
            #         print('Accuracy %s' % str(accuracy))
            #     accurate = round(sum(list_accuracy)/len(list_accuracy), 2)
            #     sets[index + 1].append(accurate)
            #     print('================================')
            #     print('Accurate: %s' % str(accurate))

        # sets = {1: [97.33, 97.2, 96.9], 2: [98.22, 99.6, 99.2], 3: [99.11, 99.4, 99.33]}
        set_1 = np.array(sets[1])
        set_2 = np.array(sets[2])
        set_3 = np.array(sets[3])
        self.show_report_chart(set_1, set_2, set_3, sets)
        # self.show_performance_chart(np.array(time_sets[0]))

    def show_report_chart(self, sets):
        sets = np.array(sets)
        title_font = {'fontname': 'Arial', 'size': '10', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'bottom', 'horizontalalignment': 'center'}
        folds = np.array([1, 2, 3])
        bar_width = 0.2
        plt.bar(folds, sets, color=(40/255, 64/255, 112/255, 0.85), width=bar_width, label='Dataset')
        plt.legend(['Dataset'])
        plt.xticks(folds, ['Dataset 1', 'Dataset 2', 'Dataset 3'])
        plt.ylabel("Acurracy")
        plt.xlabel("5-Fold Cross Validation")
        plt.title("SVM Report")
        for x in range(3):
            x += 1
            plt.text(x, sets[x-1] - 5, str(sets[x-1]) + '%',  **title_font)
        plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.6)
        plt.ylim(0, 115)
        plt.tight_layout()
        plt.show()

    def show_performance_chart(self, lst_performance):
        datasets = np.array([1, 2, 3])

        plt.plot(datasets, lst_performance, color='red', marker='o')
        plt.xticks(datasets, ['Dataset 1', 'Dataset 2', 'Dataset 3'])
        plt.title('Model Performance', fontsize=14)
        plt.xlabel('Datasets', fontsize=14)
        plt.ylabel('Seconds', fontsize=14)
        plt.grid(True)
        plt.show()

    def get_cv_number(self, Y_train):
        a = dict(Counter(Y_train))
        return max(3, min(a.values()))
    def choose_test(self, list_img, n_set):
        imgs = []
        for i in range(n_set):
            imgs.append(random.choice(list_img))
        return imgs

    def get_y(self, x_test):
        result = []
        for i in x_test:
            result.append('_'.join(i.split('\\')[-1].split('_')[:-1]))
        return result

    def filter_y_test(self, y_test, y_train):
        y = y_test
        for index, i in enumerate(y):
            if i not in y_train:
                y[index] = 'Unknown'
        return y

    def get_evaluate(self, y_test, y_predict):
        num_lost = 0
        total = len(y_test)
        # lost_indexes = []
        for index, i in enumerate(zip(y_test, y_predict)):
            if i[0] != i[1]:
                num_lost += 1
                # lost_indexes.append({i[1]: x_test[index]})
        lost = round(float(num_lost/total*100), 2)
        accuracy = 100-lost
        return accuracy

    def svm(self, n):
        lfw_people = fetch_lfw_people(min_faces_per_person=n, resize=0.5)

        # for machine learning we use the 2 data directly (as relative pixel
        # positions info is ignored by this model)

        # for machine learning we use the 2 data directly (as relative pixel
        # positions info is ignored by this model)

        X = lfw_people.data
        # the label to predict is the id of the person
        y = lfw_people.target
        y_counter = dict(Counter(y.tolist()))
        y_counter = dict(sorted(y_counter.items(), reverse=True))
        count = 0
        l_y = []
        for key, value in y_counter.items():
            if count == 100:
                break
            count += 1
            l_y.append(key)
        new_y = []
        x_index = []
        count = {}
        for index, i in enumerate(y.tolist()):
            if i in l_y:
                if i in count.keys():
                    if count[i] == n:
                        continue
                    count[i] += 1
                else:
                    count[i] = 1
                new_y.append(i)
                x_index.append(index)

        new_x = []
        for i in x_index:
            new_x.append(X[i])

        new_y = np.array(new_y)
        new_x = np.array(new_x)

        target_names = lfw_people.target_names
        # split into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            new_x, new_y, test_size=0.333)
        n_components = 150

        pca = PCA(n_components=n_components, svd_solver='randomized',
                  whiten=True).fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        param_grid = {'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1, 5, 10], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=self.get_cv_number(new_y))
        clf = clf.fit(X_train_pca, y_train)
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        y_pred = clf.predict(X_test_pca)
        return y_test, y_pred


    @api.model
    def test(self):
        lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.4)

        # introspect the images arrays to find the shapes (for plotting)
        n_samples, h, w = lfw_people.images.shape

        # for machine learning we use the 2 data directly (as relative pixel
        # positions info is ignored by this model)
        X = lfw_people.data
        n_features = X.shape[1]

        # the label to predict is the id of the person
        y = lfw_people.target

        # split into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33)

        n_components = 66

        pca = PCA(n_components=n_components, svd_solver='randomized',
                  whiten=True).fit(X_train)


        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=3)
        clf = clf.fit(X_train_pca, y_train)
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        y_pred = clf.predict(X_test_pca)

        print(y_test)
        print(y_pred)

    def training_2(self, X_train, Y_train, cv=0, X_test=None):
        data_encoding = []
        data_names = []
        for img, name in zip(X_train, Y_train):
            img = cv2.imread(img)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_frame = face_recognition.face_locations(rgb_image, model='hog')
            if len(face_frame) == 1:
                face_encodings = face_recognition.face_encodings(rgb_image, face_frame)[0]
                data_encoding.append(face_encodings)
                data_names.append(name)
        if cv != 0:
            param_grid = {'C': [1000, 10000, 20000, 50000, 100000],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0,5, 1, 5, 10]}
            clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=cv)
            clf = clf.fit(data_encoding, data_names)
            print("Best estimator found by grid search:")
            print(clf.best_estimator_)
            a = []
            for img in X_test:
                img_r = cv2.imread(img)
                rgb_image = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_image)
                if encodings:
                    a.append(encodings[0])
            y_pred = clf.predict(a)
            return y_pred
        return {'encoding': data_encoding, 'name': data_names}

    def recognition_2(self, data, X_test):
        Y_predicted = []
        for img in X_test:
            img_r = cv2.imread(img)
            rgb_image = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            if not encodings:
                Y_predicted.append('%s --- %s' % (img.split('\\')[-1], 'not encoding'))
                continue
            name = "Unknown"
            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(data['encoding'], encodings[0])
            result = min(face_distances)
            if result <= 0.55:
                best_index = np.argmin(face_distances)
                name = data['name'][best_index]
            Y_predicted.append(name)
        return Y_predicted

    def recognition_svm(self, clf, X_test):
        y_pred = clf.predict(X_test)
        return y_pred

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
