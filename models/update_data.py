# -*- coding: utf-8 -*-

from odoo import models, fields, api

class UpdateData(models.Model):
    _name = 'da.update.data'

    name = fields.Char(string='Name')

    def action_update(self, id, name):
        face_base_obj = self.env['da.face.base']
        rec = face_base_obj.search([('id', '=', id)])
        if rec:
            rec.name = name
        else:
            face_base_obj.create({'id': id,
                                  'name': name})
