# -*- coding: utf-8 -*

from datetime import datetime, timedelta
from odoo import models, api, fields, _
from odoo.exceptions import ValidationError


class HrEmployeeFacebaseInherit(models.Model):
    _inherit = 'hr.employee'
    _description = 'HR Employee Facebase Inherit'

    images = fields.Many2many(string='Images', comodel_name='ir.attachment',
                              relation='hr_employee_facebase_images_rel')
