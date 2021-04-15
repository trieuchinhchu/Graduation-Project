# -*- coding: utf-8 -*-

from odoo import models, fields, api
from datetime import datetime

_CHECK_TYPE = [('check_in', 'Check In'),
               ('check_out', 'Check Out')]


class AttendanceLog(models.Model):
    _name = 'da.attendance.log'

    attendance_id = fields.Many2one(comodel_name='hr.attendance', string='Attendance ID')
    employee_id = fields.Many2one(comodel_name='hr.employee', string='Employee', required=True)
    punch_time = fields.Datetime(string='Punch time', required=True)
    date = fields.Date(string='Date', compute='compute_date', store=True)
    check_type = fields.Selection(selection=_CHECK_TYPE, default='check_in', string='Type', required=True)

    @api.depends('punch_time')
    def compute_date(self):
        for r in self:
            if r.punch_time:
                r.date = r.punch_time.date()

    @api.model
    def create(self, values):
        res = super(AttendanceLog, self).create(values)
        attendance_id = res.create_attendance()
        if attendance_id:
            res.attendance_id = attendance_id.id
        return res

    @api.multi
    def write(self, values):
        if 'punch_time' in values:
            punch_time = datetime.strptime(values.get('punch_time', False), '%Y-%m-%d %H:%M:%S')
            self.write_attendance(punch_time)
        return super(AttendanceLog, self).write(values)

    def create_attendance(self):
        attendance_id = self.env['hr.attendance'].search([
            ('employee_id', '=', self.employee_id.id),
            ('date', '=', self.punch_time.date())
        ], limit=1)
        if not attendance_id:
            attendance_record = {'employee_id': self.employee_id.id,
                                 'check_in': self.punch_time,
                                 'check_out': self.punch_time}
            attendance_record = self.env['hr.attendance'].create(attendance_record)
            return attendance_record

        log_ids = self.search([('attendance_id', '=', attendance_id.id)])
        if log_ids:
            if self.check_type == 'check_in':
                punch_time = min(log_ids.filtered(lambda r: r.check_type == 'check_in').mapped('punch_time') + [self.punch_time])
            else:
                punch_time = max(log_ids.filtered(lambda r: r.check_type == 'check_out').mapped('punch_time') + [self.punch_time])
            attendance_id.write({self.check_type: punch_time})
        return attendance_id

    def write_attendance(self, punch_time):
        if self.attendance_id and punch_time:
            log_ids = self.search([('attendance_id', '=', self.attendance_id.id),
                                   ('check_type', '=', self.check_type)]).mapped('punch_time')
            punch_time = min(log_ids + [punch_time])
            self.attendance_id.write({self.check_type: punch_time})
