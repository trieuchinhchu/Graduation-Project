# -*- coding: utf-8 -*-

from odoo import models, fields, api
from datetime import datetime, time
from dateutil import tz


class DAResourceCalendar(models.Model):
    _inherit = 'resource.calendar'

    work_from = fields.Float(string='Work From', default=8.5)
    work_to = fields.Float(string='Work To', default=17.5)
    work_from_datetime = fields.Datetime(compute='compute_work_time', store=True)
    work_to_datetime = fields.Datetime(compute='compute_work_time', store=True)

    def _get_default_attendance_ids(self):
        return [
            (0, 0, {'name': 'Monday Morning', 'dayofweek': '0', 'hour_from': self.work_from, 'hour_to': 12, 'day_period': 'morning'}),
            (0, 0, {'name': 'Monday Evening', 'dayofweek': '0', 'hour_from': 13, 'hour_to': self.work_to, 'day_period': 'afternoon'}),
            (0, 0, {'name': 'Tuesday Morning', 'dayofweek': '1', 'hour_from': self.work_from, 'hour_to': 12, 'day_period': 'morning'}),
            (0, 0, {'name': 'Tuesday Evening', 'dayofweek': '1', 'hour_from': 13, 'hour_to': self.work_to, 'day_period': 'afternoon'}),
            (0, 0, {'name': 'Wednesday Morning', 'dayofweek': '2', 'hour_from': self.work_from, 'hour_to': 12, 'day_period': 'morning'}),
            (0, 0, {'name': 'Wednesday Evening', 'dayofweek': '2', 'hour_from': 13, 'hour_to': self.work_to, 'day_period': 'afternoon'}),
            (0, 0, {'name': 'Thursday Morning', 'dayofweek': '3', 'hour_from': self.work_from, 'hour_to': 12, 'day_period': 'morning'}),
            (0, 0, {'name': 'Thursday Evening', 'dayofweek': '3', 'hour_from': 13, 'hour_to': self.work_to, 'day_period': 'afternoon'}),
            (0, 0, {'name': 'Friday Morning', 'dayofweek': '4', 'hour_from': self.work_from, 'hour_to': 12, 'day_period': 'morning'}),
            (0, 0, {'name': 'Friday Evening', 'dayofweek': '4', 'hour_from': 13, 'hour_to': self.work_to, 'day_period': 'afternoon'})
        ]

    attendance_ids = fields.One2many(
        'resource.calendar.attendance', 'calendar_id', 'Working Time',
        copy=True, readonly=True, default=_get_default_attendance_ids, compute='compute_attendance_ids', store=True)

    @api.depends('work_from', 'work_to')
    def compute_attendance_ids(self):
        for r in self:
            if r.work_from and r.work_to:
                r.attendance_ids = r._get_default_attendance_ids()

    @api.depends('work_from', 'work_to')
    def compute_work_time(self):
        for r in self:
            if r.work_from:
                r.work_from_datetime = datetime.today().replace(hour=int(r.work_from),
                                                                minute=round((r.work_from - int(r.work_from))*60),
                                                                second=0)
            if r.work_to:
                r.work_to_datetime = datetime.today().replace(hour=int(r.work_to),
                                                              minute=round((r.work_to - int(r.work_to))*60),
                                                              second=0)