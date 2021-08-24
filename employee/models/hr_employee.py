# -*- coding: utf-8 -*

from datetime import datetime, timedelta
import logging
from werkzeug import urls

from odoo import models, api, fields, _
from odoo.exceptions import ValidationError
from odoo.http import request
import secrets
import pytz

_tzs = [(tz, tz) for tz in sorted(pytz.all_timezones, key=lambda tz: tz if not tz.startswith('Etc/') else '_')]

def _tz_get(self):
    return _tzs


class HrEmployeeWebsiteInherit(models.Model):
    _inherit = 'hr.employee'
    _description = 'HR employee'

    tz = fields.Selection(_tz_get, string='Timezone', default='Asia/Ho_Chi_Minh')
    personal_email = fields.Char('Personal Email', copy=False)
    internship = fields.Boolean("Internship", help="Thực tập sinh gửi đến từ công ty khác.")
    code = fields.Char(string='Employee Code', default=_('New'))
    email_update_date = fields.Datetime(string='Latest email sent', readonly=True)
    emp_info_state = fields.Selection(string='Updating Status',
                                      selection=[('employee_update', 'Employee updating'),
                                                 ('wait_to_hr_confirm', 'Waiting for HR confirm'),
                                                 ('hr_confirm', 'Hr Confirmed')],
                                      readonly=True, default='employee_update')
    password_new_employee = fields.Char(string="Password New Employee")
    old_emp_type = fields.Selection(selection=[('within_3_months', 'Within 3 Months'),
                                               ('re_recruiting', 'Re-recruiting')], string='Recruiting type')

    def set_password(self):
        for r in self:
            data = secrets.token_hex(nbytes=2)
            if r.user_id:
                r.user_id.sudo().write({
                    'new_password': data
                })
                r.user_id._set_new_password()
                r.password_new_employee = data

    def action_url(self, controller='/'):
        base_url = request.env['ir.config_parameter'].sudo().get_param('web.base.url')
        self.ensure_one()
        action_url = self._notify_get_action_link('controller', controller=controller)
        return urls.url_join(base_url, action_url)

    def get_emp_url(self):
        self.ensure_one()
        base_url = request.env['ir.config_parameter'].sudo().get_param('web.base.url')
        return urls.url_join(base_url, f'/employee/{self.id}')

    def get_emp_update_url(self):
        controller = '/new_employee/info/' + str(self.id)
        return controller

    def approve_employee_info_update(self):
        self.update({'emp_info_state': 'hr_confirm'})

    def send_request_update_employee_info_email(self):
        template = self.env.ref('hr_employee.email_to_new_emp_template')
        for r in self:
            r.email_update_date = fields.Datetime.now()
            r.emp_info_state = 'employee_update'
            self.env['mail.template'].browse(template.id).sudo().send_mail(r.id)

    def get_new_employee_code(self):
        code = self.env['ir.sequence'].with_context(force_company=self.company_id.id). \
            next_by_code('employee.code') or self.code or 'New'
        return code

    @api.model
    def name_search(self, name, args=None, operator='ilike', limit=200):
        res = super(HrEmployeeWebsiteInherit, self).name_search(name, args=args, operator=operator, limit=limit)
        if self._context.get('old_employee', False) or self._context.get('internship_employee', False):
            domain = ['|', '|', ('name', operator, name), ('work_email', operator, name),
                      ('code', operator, name)]
            if self._context.get('old_employee', False):
                domain.extend([('active', '=', False), ('resignation', '=', True)])
            elif self._context.get('internship_employee', False):
                domain.extend(['|', ('active', '=', False), ('active', '=', True), ('internship', '=', True)])
            res = self.search(domain, limit=limit)
            return res.name_get()
        return res

    @api.model
    def search_read(self, domain=None, fields=None, offset=0, limit=None, order=None):
        if self._context.get('old_employee', False):
            domain.append(('active', '=', False))
        elif self._context.get('internship_employee', False):
            domain.append(('internship', '=', True))
        return super(HrEmployeeWebsiteInherit, self).search_read(domain, fields, offset, limit, order)

    @api.model
    def read_group(self, domain, fields, groupby, offset=0, limit=None, orderby=False, lazy=True):
        if self._context.get('old_employee', False):
            domain.append(('active', '=', False))
        elif self._context.get('internship_employee', False):
            domain.append(('internship', '=', True))
        return super(HrEmployeeWebsiteInherit, self).read_group(domain, fields, groupby, offset=offset, limit=limit,
                                                                orderby=orderby, lazy=lazy)

    @api.model
    def create(self, vals):
        if vals.get('code', _('New')) == _('New'):
            if vals.get('internship', False):
                vals['code'] = self.env.ref('hr_core.sequence_employee_internship_code').next_by_id() or _('New')
            elif 'company_id' in vals:
                vals['code'] = self.env['ir.sequence'].with_context(force_company=vals['company_id']). \
                                   next_by_code('employee.code') or _('New')
            else:
                raise ValidationError("Company is required!")

        return super(HrEmployeeWebsiteInherit, self).create(vals)

    def open_infor(self):
        action = self.env.ref('hr_employee.da_skill_employee_action').read()[0]
        action['res_id'] = self.id
        return action