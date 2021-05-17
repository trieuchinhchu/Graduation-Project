# -*- coding: utf-8 -*-

from odoo import models, fields, api
from datetime import datetime, time, timezone
import pytz

# put POSIX 'Etc/*' entries at the end to avoid confusing users - see bug 1086728
_tzs = [(tz, tz) for tz in sorted(pytz.all_timezones, key=lambda tz: tz if not tz.startswith('Etc/') else '_')]
def _tz_get(self):
    return _tzs


class HrEmployeeInherit(models.Model):
    _inherit = 'hr.employee'

    tz = fields.Selection(_tz_get, string='Timezone', default='Asia/Ho_Chi_Minh')
    account = fields.Char(string='Account', compute='compute_account', store=True)

    @api.depends('user_id')
    def compute_account(self):
        for r in self:
            if r.user_id:
                r.account = r.user_id.name