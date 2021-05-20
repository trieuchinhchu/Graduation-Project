# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import timedelta
import datetime
from pytz import timezone, utc
from dateutil.relativedelta import relativedelta

from odoo import models, fields, api, _
from odoo.tools import float_utils
from odoo.exceptions import ValidationError


class DAResCompanyHoliday(models.Model):
    _inherit = "res.company"
    _description = "DA Company"

    deadline_request_day = fields.Integer(string='Last day create request', default=1,
                                          help='After this day member can not approve request!')
    deadline_approve_day = fields.Integer(string='Last day approve request', default=15,
                                          help='After this day pm, dl can not approve request!')
    approve_day = fields.Integer(string='Approve day', default=2,
                                 help='After this day, pm, dl can approve request at late status!')
    hr_email = fields.Char("HR Email")
    rec_email = fields.Char("Rec Email")
    it_email = fields.Char("IT Email", default='it@da.com.vn')

    _sql_constraints = [
        ('deadline_request_day_check', "CHECK ((deadline_request_day >= 0 ))",
         "The Last day create request can not be lower than 0."),
        ('deadline_approve_day', "CHECK ((deadline_approve_day >= 0 ))",
         "The Last day approve request can not be lower than 0."),
        ('approve_day', "CHECK ((approve_day >= 0 ))", "The Approve day can not be lower than 0."),
    ]

    def check_da_holiday(self, date):
        global_leave_ids = self.resource_calendar_id.global_leave_ids
        tz = timezone((self.resource_calendar_id or utc).tz)
        check_overlap_date = global_leave_ids.filtered(
            lambda r: r.date_from.astimezone(tz).date() <= date <= r.date_to.astimezone(tz).date())
        if check_overlap_date:
            return True
        return False

    def get_request_deadline_date(self, request_date, is_approve=False):
        self.ensure_one()
        request_day = self.deadline_approve_day if is_approve else self.deadline_request_day
        allow_date = request_date + relativedelta(day=1, months=1, days=-1)
        if request_day >= 1:
            allow_date = request_date + relativedelta(day=request_day, months=1)
            while self.check_da_holiday(allow_date) or allow_date.weekday() in (5, 6):
                allow_date += relativedelta(days=1)
        if fields.Date.today() > allow_date:
            return allow_date + relativedelta(days=1)
        return False

    def is_late_approve_record(self, request_date):
        self.ensure_one()
        allow_date = request_date + relativedelta(day=1, months=1, days=-1)
        if self.approve_day >= 1:
            allow_date = request_date + relativedelta(day=self.approve_day, months=1)
            while self.check_da_holiday(allow_date) or allow_date.weekday() in (5, 6):
                allow_date += relativedelta(days=1)
        if fields.Date.today() > allow_date:
            return True
        return False

    def get_working_day_date(self, date):
        self.ensure_one()
        if not isinstance(date, datetime.date):
            date = date.date()
        while self.check_da_holiday(date) or date.weekday() in (5, 6):
            date += relativedelta(days=1)
        return date