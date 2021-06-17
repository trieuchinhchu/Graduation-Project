# -*- coding: utf-8 -*


from odoo import models, api, fields, _
from odoo.exceptions import ValidationError
import base64, io, cv2, face_recognition, numpy as np
from PIL import Image


class HrEmployeeFacebaseInherit(models.Model):
    _inherit = 'hr.employee'
    _description = 'HR Employee Facebase Inherit'

    images = fields.Many2many(string='Images',
                              comodel_name='ir.attachment',
                              required=True,
                              relation='hr_employee_facebase_images_rel')

    view_images = fields.Many2many(string='View Images',
                                   comodel_name='ir.attachment',
                                   compute='_compute_view_images')

    @api.depends('images')
    def _compute_view_images(self):
        for r in self:
            if r.images:
                r.view_images = [(6, 0, r.images.ids)]

    @api.constrains('images')
    def validate_images(self):
        for r in self:
            if not 5 <= len(r.images) <= 10:
                raise ValidationError('Yêu cầu cung cấp từ 5 đến 10 ảnh chụp chân dung!')
            lst = []
            for image in r.images:
                decode_img = base64.b64decode(image.datas)
                img = Image.open(io.BytesIO(decode_img))
                resize_img = cv2.resize(np.asarray(img), (720, 960), interpolation=cv2.INTER_NEAREST)
                rgb_image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
                face_frame = face_recognition.face_locations(rgb_image, model='hog')
                # If training image contains exactly one face
                if len(face_frame) != 1:
                    lst.append(image.datas_fname)
            if lst:
                raise ValidationError('Ảnh không đạt yêu cầu: \n%s' % '\n'.join(lst))

