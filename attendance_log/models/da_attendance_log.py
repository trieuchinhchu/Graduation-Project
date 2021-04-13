# -*- coding: utf-8 -*-

from odoo import models, fields, api

_CHECK_TYPE = [('check_in', 'Check In'),
               ('check_out', 'Check Out')]


class AttendanceLog(models.Model):
    _name = 'da.attendance.log'

    attendance_id = fields.Many2one(comodel_name='hr.attendance', string='Attendance ID')
    employee_id = fields.Many2one(comodel_name='hr.employee', string='Employee', required=True)
    punch_time = fields.Datetime(string='Punch time', required=True)
    date = fields.Date(string='Date', compute='compute_date', store=True)
    check_type = fields.Selection(selection=_CHECK_TYPE, string='Type', required=True)

    @api.depends('punch_time')
    def compute_date(self):
        for r in self:
            if r.punch_time:
                r.date = r.punch_time.date()

    @api.model
    def create(self, values):
        res = super(AttendanceLog, self).create(values)
        attendance_id, attendance_log_id = res.create_attendance()
        if attendance_id and attendance_log_id:
            res.attendance_id = attendance_id.id
            attendance_log_id.attendance_id = attendance_id.id
        return res

    def create_attendance(self):
        attendance_id = self.env['hr.attendance'].search([
            ('employee_id', '=', self.employee_id.id),
            ('date', '=', self.punch_time.date())
        ], limit=1)
        log_obj = self.env['da.attendance.log']
        if not attendance_id:
            log_obj = log_obj.search([('date', '=', self.date)])
            if not log_obj:
                return
            attendance_record = {'employee_id': self.employee_id.id}
            if log_obj.check_type == 'check_in':
                attendance_record.update({'check_in': log_obj.punch_time,
                                        'check_out': self.punch_time})
            else:
                attendance_record.update({'check_out': log_obj.punch_time,
                                        'check_in': self.punch_time})

            attendance_id = self.env['hr.attendance'].create(attendance_record)
        return attendance_id, log_obj