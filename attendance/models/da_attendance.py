# -*- coding: utf-8 -*-

from odoo import models, fields, api


class Attendance(models.Model):
    _inherit = 'hr.attendance'

    attendance_log_ids = fields.One2many(comodel_name='da.attendance.log',
                                         inverse_name='attendance_id',
                                         compute='compute_attendance_log',
                                         store=True,
                                         string='Attendance log ids')
    date = fields.Date(string='Date', compute='compute_date', store=True)

    @api.depends('attendance_log_ids')
    def compute_attendance_log(self):
        for r in self:
            if r.attendance_log_ids:
                r.check_in = min(r.attendance_log_ids.filtered(lambda x: x.check_type == 'check_in').mapped('punch_time'))
                r.check_out = max(r.attendance_log_ids.filtered(lambda x: x.check_type == 'check_out').mapped('punch_time'))

    @api.depends('check_in')
    def compute_date(self):
        for r in self:
            if r.check_in:
                r.date = r.check_in.date()