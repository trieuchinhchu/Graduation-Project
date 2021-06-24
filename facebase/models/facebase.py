# -*- coding: utf-8 -*-
import base64, pickle, cv2, face_recognition, io, os, time, math
from dateutil.relativedelta import relativedelta
from odoo import models, fields, api
from datetime import datetime, timezone
from odoo.exceptions import ValidationError
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn import metrics
from imutils import paths
import joblib

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class FaceBase(models.Model):
    _name = 'da.facebase'

    data = fields.Binary(string='Trained Data')
    training_date = fields.Date(default=fields.Date.today(), string='Date')

    # @api.model
    # def filter_training_data(self):
    #     base_path = 'D:\odoo12\lfw\lfw'
    #     training_path = 'D:\odoo12\\training\\training_3'
    #     testing_path = 'D:\odoo12\\testing\\testing_3'
    #     count_base = 0
    #     test_left = 200
    #     for dir in os.listdir(base_path):
    #         image_paths = list(paths.list_images(base_path + '\%s' % dir))
    #         is_tested = 0
    #         if count_base == 100:
    #             break
    #         lst_training = []
    #         lst_test = []
    #         if len(image_paths) >= 11:
    #             for image_path in image_paths:
    #                 if len(lst_training) < 10:
    #                     if self.face_filtered(image_path):
    #                         lst_training.append(image_path)
    #                 elif len(lst_test) <= 1:
    #                     if test_left != 0:
    #                         if self.face_filtered(image_path):
    #                             lst_test.append(image_path)
    #                             is_tested += 1
    #                         if is_tested == 2:
    #                             test_left -= 1
    #                             break
    #                     else:
    #                         if self.face_filtered(image_path):
    #                             lst_test.append(image_path)
    #                             break
    #         if lst_training and lst_test:
    #             for i in lst_training:
    #                 cv2.imwrite(training_path + '\%s' % i.split('\\')[-1], cv2.imread(i))
    #             for i in lst_test:
    #                 cv2.imwrite(testing_path + '\%s' % i.split('\\')[-1], cv2.imread(i))
    #             count_base += 1
    #     print('%s people'%count_base)

    @api.model
    def filter_training_data(self):
        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

        # #############################################################################
        # Download the data, if not already on disk and load it as numpy arrays

        lfw_people = fetch_lfw_people(min_faces_per_person=10)

        # introspect the images arrays to find the shapes (for plotting)
        n_samples, h, w = lfw_people.images.shape

        # for machine learning we use the 2 data directly (as relative pixel
        # positions info is ignored by this model)
        X = lfw_people.data
        n_features = X.shape[1]

        # the label to predict is the id of the person
        y = lfw_people.target
        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]

        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)

        # #############################################################################
        # Split into a training set and a test set using a stratified k fold

        # split into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # #############################################################################
        # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
        # dataset): unsupervised feature extraction / dimensionality reduction

        n_components = 150
        #
        # print("Extracting the top %d eigenfaces from %d faces"
        #       % (n_components, X_train.shape[0]))
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized',
                  whiten=True).fit(X_train)
        # print("done in %0.3fs" % (time() - t0))
        #
        eigenfaces = pca.components_.reshape((n_components, h, w))
        #
        # print("Projecting the input data on the eigenfaces orthonormal basis")
        t0 = time()
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print("done in %0.3fs" % (time() - t0))

        # #############################################################################
        # Train a SVM classification model

        print("Fitting the classifier to the training set")
        t0 = time()
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(
            SVC(kernel='rbf', class_weight='balanced'), param_grid
        )
        clf = clf.fit(X_train_pca, y_train)
        print("done in %0.3fs" % (time() - t0))
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        # #############################################################################
        # Quantitative evaluation of the model quality on the test set

        print("Predicting people's names on the test set")
        t0 = time()
        y_pred = clf.predict(X_test_pca)
        print("done in %0.3fs" % (time() - t0))

        print(classification_report(y_test, y_pred, target_names=target_names))
        print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

        # #############################################################################
        # Qualitative evaluation of the predictions using matplotlib

        def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
            """Helper function to plot a gallery of portraits"""
            plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
            plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
            for i in range(n_row * n_col):
                plt.subplot(n_row, n_col, i + 1)
                plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
                plt.title(titles[i], size=12)
                plt.xticks(())
                plt.yticks(())

        # plot the result of the prediction on a portion of the test set

        def title(y_pred, y_test, target_names, i):
            pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
            true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
            return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

        prediction_titles = [title(y_pred, y_test, target_names, i)
                             for i in range(y_pred.shape[0])]

        plot_gallery(X_test, prediction_titles, h, w)

        # plot the gallery of the most significative eigenfaces

        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        plot_gallery(eigenfaces, eigenface_titles, h, w)

        plt.show()



    def face_filtered(self, image_path):
        rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        face_frame = face_recognition.face_locations(rgb_image, model='hog')
        if len(face_frame) != 1:
            return False
        return True

    def get_path(self, path_str):
        if not os.path.isdir(path_str):
            os.mkdir(path_str)
        return path_str


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
                    '%s/%s.%s.%s.jpg' % (dataset_path, self.employee_id.user_id.login, self.employee_id.id, num_img),
                    img)
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
        data_encoding = []
        data_names = []
        training_path = 'D:\odoo12\\training\\training_1'
        data_path = 'D:\odoo12\data\data_1'
        images = list(paths.list_images(training_path))
        for image in images:
            name = '_'.join(image.split('\\')[-1].split('_')[:-1])
            count += 1
            rgb_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            face_frame = face_recognition.face_locations(rgb_image, model='hog')
            # If training image contains exactly one face
            if len(face_frame) == 1:
                face_encodings = face_recognition.face_encodings(rgb_image, face_frame)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                data_encoding.append(face_encodings)
                data_names.append(np.array(name))
            else:
                print(image + " was skipped and can't be used for training")

            print('%s' % image)

        # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        #               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        # clf = GridSearchCV(
        #     SVC(kernel='rbf', class_weight='balanced'), param_grid
        # )
        # print("Best estimator found by grid search:")
        # print(clf.best_estimator_)
        clf = svm.SVC(gamma='scale', probability=True)
        clf.fit(data_encoding, data_names)
        joblib.dump(clf, data_path+'\\training_data_svm.pkl')

        # data = {'encoding': data_encoding, 'name': data_names}
        # f = open(data_path+'\\training_data_svm', 'wb')
        # f.write(pickle.dumps(data))
        # f.close()

    def plot_desicion_boundary(self, X, y, clf, title=None):
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1


        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

        if title is not None:
            plt.title(title)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        plt.show()

    @api.model
    def recognition(self):
        testing_path = 'D:\odoo12\\testing\\testing_1'
        data_path = 'D:\odoo12\data\data_1\\training_data_svm.pkl'
        clf = joblib.load(data_path)
        y_test = []
        x_test = []
        for image in paths.list_images(testing_path):
            img_name = '_'.join(image.split('\\')[-1].split('_')[:-1])
            img = cv2.imread(image)
            # resized = cv2.resize(img, (250, 250), interpolation=cv2.INTER_AREA)
            rgb_image = img[:, :, ::-1]
            encodings = face_recognition.face_encodings(rgb_image)
            for encoding in encodings:
                x_test.append(encoding)
            y_test.append(img_name)
        # y_test = np.array(y_test)

        y_predicted = clf.predict(x_test).tolist()
        a = metrics.accuracy_score(y_test, y_predicted)
        print(a)
        print(metrics.classification_report(y_test, y_predicted))
        print(metrics.confusion_matrix(y_test, y_predicted))

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

    def recognizer(self, image):
        img_name = '_'.join(image.split('\\')[-1].split('_')[:-1])
        img = cv2.imread(image)
        resized = cv2.resize(img, (250, 250), interpolation=cv2.INTER_AREA)
        rgb_image = resized[:, :, ::-1]
        encodings = face_recognition.face_encodings(rgb_image)
        if not encodings:
            print(image)
        # for encoding in encodings:
        #     # name = "Unknown"
        #     # use the known face with the smallest distance to the new face
        #     # face_distances = face_recognition.face_distance(data['encoding'], encoding)
        #
        #
        #     # result = min(face_distances)
        #     # # if result <= 0.5:
        #     # best_index = np.argmin(face_distances)
        #     # name = data['name'][best_index]
        #     # return [img_name, name]

        return [img_name, False]

    def calculate_percentage(self, lst):
        lst_false = list(filter(lambda r: r[0] != r[1], lst))
        print(lst_false)
        num_false = len(lst_false)

        num_true = len(lst) - num_false

        return num_true/len(lst)*100


    def face_distance_to_conf(self, face_distance, face_match_threshold=0.6):
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
