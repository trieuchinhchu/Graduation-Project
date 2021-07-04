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
            if len(r.images) != 10:
                raise ValidationError('Yêu cầu cung cấp đủ 10 ảnh chụp rõ khuôn mặt!')
            lst = []
            for image in r.images:

                decode_img = base64.b64decode(image.datas)
                img = Image.open(io.BytesIO(decode_img))
                img_r = np.asarray(img)
                size = self.calculate_resize_image(img_r)
                resize_img = cv2.resize(img_r, size, interpolation=cv2.INTER_NEAREST)
                rgb_image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
                face_frame = face_recognition.face_locations(rgb_image, model='hog')
                face_encodings = face_recognition.face_encodings(rgb_image, face_frame)
                # If training image contains exactly one face
                if not face_encodings:
                    lst.append(str(image.datas_fname))
                else:
                    img = Image.fromarray(resize_img, 'RGB')
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    rz_img = buffer.getvalue()
                    image.datas = base64.b64encode(rz_img)
            if lst:
                raise ValidationError('Ảnh không đạt yêu cầu: \n%s' % '\n'.join(lst))

    def calculate_resize_image(self, img):
        d = 250
        h, w, c = img.shape
        p = h/w
        return (d, round(d*p))