# -*- coding: utf-8 -*

from datetime import datetime, timedelta
from odoo import models, api, fields, _
from odoo.exceptions import ValidationError


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
