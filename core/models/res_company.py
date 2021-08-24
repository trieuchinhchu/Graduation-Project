# -*- coding: utf-8 -*-

import datetime
from pytz import timezone, utc
from dateutil.relativedelta import relativedelta
from odoo import models, fields, api, _


class DAResCompanyHoliday(models.Model):
    _inherit = "res.company"
    _description = "DA Company"

    hr_email = fields.Char("HR Email", default='hr@da.com.vn')
    rec_email = fields.Char("Rec Email", default='rec@da.com.vn')
    it_email = fields.Char("IT Email", default='it@da.com.vn')