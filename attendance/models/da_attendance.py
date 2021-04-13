# -*- coding: utf-8 -*-

from odoo import models, fields, api


class Attendance(models.Model):
    _inherit = 'hr.attendance'

    date = fields.Date(string='Date', compute='compute_date', store=True)

    @api.depends('check_in')
    def compute_date(self):
        for r in self:
            if r.check_in:
                r.date = r.check_in.date()