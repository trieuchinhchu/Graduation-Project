# -*- coding: utf-8 -*-

from odoo import models, fields, api

class FaceBase(models.Model):
    _name = 'da.face.base'

    name = fields.Char(string='Name')