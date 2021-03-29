# -*- coding: utf-8 -*-

from odoo import models, fields, api
import cv2

class FaceBase(models.Model):
    _name = 'da.face.base'

    name = fields.Char(string='Name')
    res_id = fields.Interger(string='res_id')

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default')

    def insert_data(self, id, name):
        face_base_obj = self.env['da.face.base']
        rec = face_base_obj.search([('id', '=', id)])
        if rec:
            rec.name = name
        else:
            face_base_obj.create({'id': id,
                                  'name': name})
    
    # def open_camera(self):
