# -*- coding: utf-8 -*-

from odoo import models, fields, api
from datetime import datetime, time


class Attendance(models.Model):
    _inherit = 'hr.attendance'

    date = fields.Date(string='Date', compute='compute_date', store=True)
    worked_hours_nft = fields.Float(string='Worked hours', compute='compute_worked_hours_nft', store=False)
    late_cm = fields.Boolean(string='Late Comming', default=False, compute='compute_late_cm', store=False)
    early_leave = fields.Boolean(string='Early Leave', default=False, compute='compute_early_leave', store=False)

    @api.depends('check_in')
    def compute_late_cm(self):
        for r in self:
            if r.check_in.time() >= time(1, 30):
                r.late_cm = True

    @api.depends('check_out')
    def compute_early_leave(self):
        for r in self:
            if r.check_in.time() <= time(10, 30):
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

    # def utc_to_local(self, utc_dt):
    #     return arrow.get(utc_dt).to('local').datetime