# -*- coding: utf-8 -*-

from odoo import models, fields, api
from datetime import datetime, time
from dateutil import tz


class Attendance(models.Model):
    _inherit = 'hr.attendance'

    date = fields.Date(string='Date', compute='compute_date', store=True)
    worked_hours_nft = fields.Float(string='Worked hours', compute='compute_worked_hours_nft', store=False)
    late_cm = fields.Boolean(string='Late Comming', default=False, compute='compute_late_cm', store=False)
    early_leave = fields.Boolean(string='Early Leave', default=False, compute='compute_early_leave', store=False)

    @api.depends('check_in', 'employee_id')
    def compute_late_cm(self):
        for r in self:
            print(r.employee_id.resource_calendar_id.work_from_datetime.time())
            if r.utc_to_local(r.check_in).time() >= r.employee_id.resource_calendar_id.work_from_datetime.time():
                r.late_cm = True

    @api.depends('check_out', 'employee_id')
    def compute_early_leave(self):
        for r in self:
            print(r.employee_id.resource_calendar_id.work_from_datetime.time())
            if r.utc_to_local(r.check_out).time() <= r.employee_id.resource_calendar_id.work_to_datetime.time():
                r.early_leave = True

    @api.depends('worked_hours')
    def compute_worked_hours_nft(self):
        for r in self:
            if not r.worked_hours:
                continue
            if r.worked_hours >= 10:
                r.worked_hours_nft = r.worked_hours - 2
            elif r.worked_hours >= 9:
                r.worked_hours_nft = 8
            elif r.worked_hours >= 5:
                r.worked_hours_nft = r.worked_hours - 1
            else:
                r.worked_hours_nft = r.worked_hours

    @api.depends('check_in')
    def compute_date(self):
        for r in self:
            if r.check_in:
                r.date = r.check_in.date()

    def utc_to_local(self, utc_dt):
        return utc_dt.replace(tzinfo=tz.gettz('UTC')).astimezone(tz.gettz(self.env.context.get('tz')) or tz.gettz(self.employee_id.tz))