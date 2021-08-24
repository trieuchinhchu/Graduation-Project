# -*- coding: utf-8 -*-

import re
import logging
from werkzeug import urls
from datetime import datetime

from odoo import models, api, fields, _
from odoo.exceptions import ValidationError
from odoo.http import request

_logger = logging.getLogger(__name__)


class HrEmployeeNewLines(models.Model):
    _name = 'hr.employee.new'
    _description = 'Request create account'
    _inherit = ['mail.thread']
    _order = 'create_date desc, company_id, department_id, account'
    _rec_name = 'full_name'

    def _default_work_location(self):
        return self.env['da.location'].search([], limit=1).id or False


    employee_id = fields.Many2one('hr.employee', ondelete='set null', string='Employee',
                                  help='Employee-related data of the employee')
    code = fields.Char(related='employee_id.code', string='Employee Code', default=_('New'), store=True)
    start_work_date = fields.Date(string='Hire Date', required=False, track_visibility='onchange',
                                  help="start work date is normally the date when someone begins working ... ")
    barcode = fields.Char(string='Badge ID', required=False, track_visibility='onchange')
    work_email = fields.Char(string='Work Email', required=False, track_visibility='onchange')
    full_name = fields.Char(string='Full Name', required=True)
    first_name = fields.Char(string='First Name', required=False)
    name = fields.Char(string='Last Name', required=False)
    job_id = fields.Many2one('hr.job', 'Job Position', track_visibility='onchange', required=True)
    department_id = fields.Many2one('hr.department', string='Department',
                                    track_visibility='onchange', required=True)
    personal_email = fields.Char('Personal Email', copy=True, track_visibility='onchange', required=True)
    social_facebook = fields.Char('Personal Email', copy=True, track_visibility='onchange')
    account = fields.Char(string='Account', track_visibility='onchange', required=True)
    expected_date = fields.Date(string='Expected date', track_visibility='onchange', required=True, copy=False,
                                default=fields.Date.today())
    old_expected_date = fields.Date(string='Old Expected date', track_visibility='onchange', required=False)
    account_type = fields.Selection([('account', 'Account')], string='Account type', default='account', required=True)
    work_location_id = fields.Many2one('da.location', string='Work location', default=_default_work_location)
    note = fields.Char(string='Note')
    state = fields.Selection(selection=[('draft', 'Draft'), ('submit', 'Submitted'),
                                        ('onboard', 'Onboard'), ('rejected', 'Rejected')],
                             string='State', default='draft', track_visibility='onchange')
    company_id = fields.Many2one('res.company', string='Company', required=True,
                                 default=lambda self: self.env.user.company_id.id)
    request_type = fields.Selection([('new', 'New Employee'),
                                     ('old', 'Old employee'),
                                     ('internship', 'Internship')
                                     ], string='Request type', default='new', required=True)
    old_emp = fields.Boolean("Old employee", help="Nhân viên cũ quay lại làm việc.")
    internship_emp = fields.Boolean("Is Internship", help="Nhân viên là internship!")
    internship = fields.Boolean("Internship", help="Thực tập sinh gửi đến từ công ty khác.", default=False)
    body_shop = fields.Boolean("Bodyshop", help="Bodyshop.", default=False)
    old_emp_type = fields.Selection(selection=[('within_3_months', 'Within 3 Months'),
                                               ('re_recruiting', 'Re-recruiting')], default='re_recruiting',
                                    string='Recruiting type',track_visibility='onchange')

    source_id = fields.Many2one('utm.source', "Source", ondelete='cascade', track_visibility='onchange')
    follower = fields.Many2one('hr.employee', 'Introduce', help='Tên người phụ trách ứng viên/giới thiệu ứng viên')
    user_id = fields.Many2one("res.users", "Follower", default=lambda self: self.env.uid)
    is_import = fields.Boolean("Import", default=False)

    @api.onchange('request_type')
    def _onchange_request_type(self):
        if self.request_type == 'new':
            self.old_emp = False
            self.internship_emp = False
        elif self.request_type == 'old':
            self.old_emp = True
            self.internship_emp = False
            return {'domain': {'employee_id': [('active', '=', False)]}}
        elif self.request_type == 'internship':
            self.old_emp = False
            self.internship = False
            self.internship_emp = True
            return {'domain': {'employee_id': [('internship', '=', True)]}}

    def get_employee_by_domain(self, request_type):
        domain = []
        if request_type == 'old':
            domain = [('active', '=', False)]
        elif request_type == 'internship':
            domain = [('internship', '=', True)]
        return self.env['hr.employee'].sudo().search(domain)

    @api.onchange('expected_date')
    def _onchange_expected_date(self):
        if self.expected_date:
            self.start_work_date = self.expected_date

    @api.onchange('request_type', 'employee_id')
    def _onchange_old_emp(self):
        if self.request_type == 'new':
            self.employee_id = False
            self.account = False
            self.personal_email = False
            self.code = False
            self.barcode = False
            if not self.id:
                self.code = 'New'
        elif self.request_type != 'new' and self.employee_id:
            self.account = self.employee_id.account
            self.full_name = self.employee_id.name
            self.personal_email = self.employee_id.personal_email
            self.social_facebook = self.employee_id.social_facebook
            self.code = self.employee_id.code
            self.barcode = self.employee_id.barcode
            self.department_id = self.employee_id.department_id
            self.job_id = self.employee_id.job_id

    def submit(self):
        if not self.env.user.has_group('hr.group_hr_manager') and not self.env.user.has_group('hr.group_hr_user'):
            raise ValidationError(_('Permission denied! only HR Manager or HR Officer'))
        for r in self:
            r.create_employee()
        # send email to IT HR (1 loai)
        send_request = self.search([('id', 'in', self.ids),
                                    ('state', '=', 'draft')], order='company_id, department_id, account')
        send_request.send_request_create_account_email()

        # send email to employee yeu cau dien thong tin ca nhan
        employee_ids = send_request.filtered(lambda x: not x.old_emp).mapped('employee_id')
        employee_ids and employee_ids.send_request_update_employee_info_email()
        self.write({'state': 'submit'})

    def format_address(self, name, email):
        if not name:
            return email
        return f"{name} <{email}>"

    def send_request_create_account_email(self):
        self._check_company_configure()
        base_url = request.env['ir.config_parameter'].sudo().get_param('web.base.url')
        mails = self.env['mail.mail'].sudo()
        dl_emails = ', '.join(
            [department_id.manager_id.work_email for department_id in self.mapped('department_id') if
             department_id.manager_id.work_email])
        t_body = ""
        for emp in self:
            account_type = dict(self._fields['account_type'].selection).get(emp.account_type) or ''
            t_body += f"""
                <tr>
                    <td>{emp.company_id.name}</td>
                    <td>{emp.code}</td>
                    <td>{emp.full_name}</td>
                    <td>{emp.account}</td>
                    <td>{emp.department_id.name}</td>
                    <td>{emp.job_id.name}</td>
                    <td>{datetime.strftime(emp.expected_date, '%d/%m/%Y')}</td>
                    <td>{emp.work_location_id.name}</td>
                    <td>{account_type}</td>
                    <td>{emp.internship or ''}</td>
                    <td>{emp.personal_email}</td>
                    <td>{emp.note or ''}</td>
                    <td><a href='{base_url}/employee_new/{emp.id}' target='_blank'>link</a></td>
                </tr>
            """

            mess_to_it = f"""
                Bên dưới là danh sách các nhân viên mới chuẩn bị đi làm. <br />
                Nhờ bộ phận IT {f"tạo {account_type} và " if account_type else ''} chuẩn bị thiết bị cũng như chỗ ngồi cho các bạn.<br />
                Danh sách nhân viên mới:
            """

            if emp.request_type == 'internship':
                mess_to_it = f"""
                    Nhân viên {emp.full_name} được chuyển hợp đồng từ internship. 
                    Nhờ bộ phận IT {f"tạo {account_type} và " if account_type else ''} chuẩn bị thiết bị cũng như chỗ ngồi cho các bạn.<br /> 
                """

            mail_values = {
                'email_from': "DA  <no-reply@DA.com.vn>",
                'email_to': f"{emp.company_id.it_email}",
                'email_cc': f"{emp._get_email_cc()},{dl_emails}",
                'reply_to': f"{emp.company_id.hr_email or ''}",
                'subject': f"[DA New Member] Request tạo account và setup máy tính cho nhân viên mới",
                'body_html': f"""
                    <div style="margin: 0px; padding: 0px; overflow-x: scroll;" >
                        <p style="margin: 0px; padding: 0px; font-size: 13px;">
                            Dear DA IT,<br/>
                            Cc DA HR, DL<br/><br/>
                            {mess_to_it}
                            <table border="1" cellpadding="5" cellspacing="0" style="font-size: 13px;">
                                <tr>
                                    <th style="min-width: 60px;">Công ty</th>
                                    <th style="min-width: 60px;">Mã NV</th>
                                    <th style="max-width: 200px;">Họ và Tên</th>
                                    <th style="min-width: 160px;">Account</th>
                                    <th style="min-width: 70px;max-width: 100px;">Đơn vị</th>
                                    <th style="min-width: 70px;max-width: 100px;">Vị trí</th>
                                    <th style="min-width: 70px;max-width: 100px;">Ngày đi làm dự kiến</th>
                                    <th style="min-width: 100px;max-width: 120px;">Địa điểm làm việc</th>
                                    <th style="min-width: 70px;max-width: 100px;">Loại account</th>
                                    <th style="min-width: 70px;max-width: 100px;">Group mail</th>
                                    <th style="min-width: 70px;max-width: 100px;">Internship</th>
                                    <th style="max-width: 200px;">Email cá nhân</th>
                                    <th style="max-width: 200px;">Ghi chú</th>
                                    <th style="max-width: 200px;">Request url</th>
                                </tr>
                                <tbody>
                                    {t_body}
                                </tbody>
                            </table>
                            <br/>
                            Regards,
                        </p>
                        </div>
                """,
                'notification': True,
                'auto_delete': False,
            }
            mail = self.env['mail.mail'].sudo().create(mail_values)
            mails |= mail
            mails.sudo().send()

    def reject_work_email_action(self):
        self.ensure_one()
        mails = self.env['mail.mail'].sudo()
        mail_values = {
            'email_from': "DA  <no-reply@DA.com.vn>",
            'email_to': f"{self.company_id.it_email or ''}",
            'email_cc': f"{self._get_email_cc()}, {self.department_id.manager_id.work_email or ''}",
            'reply_to': f"{self.company_id.hr_email or ''}",
            'subject': f"[DA New Member] Nhân viên mới: {self.full_name} đã từ chối đi làm.",
            'body_html': f"""
                <div style="margin: 0px; padding: 0px;" >
                    <p style="margin: 0px; padding: 0px; font-size: 13px;">
                        Dear DA IT,<br/>
                        Cc DA HR, DL<br/><br/>
                        
                        Nhân viên {self.full_name} ({self.department_id.name}) đã từ chối đi làm.
                        Nhờ bộ phận IT xóa các tài khoản và thu hồi các thiết bị liên quan tới nhân viên này.
                        <br/>
                        <br/>
                        Regards,
                    </p>
                    </div>
            """,
            'notification': True,
            'auto_delete': False,
        }
        mail = self.env['mail.mail'].sudo().create(mail_values)
        mails |= mail
        mails.sudo().send()

    def update_expected_date_action(self):
        self.ensure_one()
        action = self.env.ref('hr_employee.action_hr_employee_update_expected_date_request').read()[0]
        action['res_id'] = self.id
        return action

    def _check_company_configure(self):
        if not self.sudo().company_id.sudo().hr_email:
            raise ValidationError(f'HR email is require for company {self.company_id.name}')
        if not self.company_id.it_email:
            raise ValidationError(f'IT email is require for company {self.company_id.name}')

    def update_expected_date_email_action(self):
        self.ensure_one()
        mails = self.env['mail.mail'].sudo()
        self._check_company_configure()
        mail_values = {
            'email_from': "DA  <no-reply@DA.com.vn>",
            'email_to': f"{self.company_id.hr_email or ''}",
            'email_cc': f"{self._get_email_cc()}, "
                        f"{self.department_id.manager_id.work_email or ''},"
                        f"{self.company_id.it_email or ''}",
            'reply_to': f"{self.company_id.hr_email or ''}",
            'subject': f"[DA New Member] Nhân viên mới: {self.full_name} thay đổi ngày dự kiến đi làm.",
            'body_html': f"""
                <div style="margin: 0px; padding: 0px;" >
                    <p style="margin: 0px; padding: 0px; font-size: 13px;">
                        Dear DA HR,<br/>
                        Cc All<br/><br/>
                        Nhân viên {self.full_name} ({self.department_id.name}) đã thay đổi ngày dự kiến đi làm từ ngày {self.old_expected_date} sang ngày {self.expected_date}.
                        <br/>
                        Người thay đổi: {self.env.user.name}   
                        <br/>
                        Ghi chú: {self.note or ''}
                        <br/>
                        <br/>
                        Regards,
                    </p>
                    </div>
            """,
            'notification': True,
            'auto_delete': False,
        }
        mail = self.env['mail.mail'].sudo().create(mail_values)
        mails |= mail
        mails.sudo().send()

    def action_hr_employee_to_onboard_action(self):
        self.ensure_one()
        action = self.env.ref('hr_employee.action_hr_employee_to_onboard_request').read()[0]
        action['res_id'] = self.id
        return action

    def set_to_onboard(self):
        if not self.env.user.has_group('hr.group_hr_manager'):
            raise ValidationError(_('Permission denied! Only HR Manager.'))
        for r in self:
            r.employee_id.write({'active': True,
                                 'barcode': r.barcode,
                                 'work_email': r.work_email,
                                 })
            user_id = r.employee_id.user_id
            user_id.sudo().write({'active': True,
                                  'groups_id': [(6, 0, [self.env.ref('base.group_user').id])]})
            user_id.sudo().mapped('partner_id').write({'active': True})
            r.state = 'onboard'
            update_info = {'resignation': False, 'resignation_date': False, 'active': True}
            if not r.employee_id.start_work_date and r.start_work_date:
                update_info.update({'start_work_date': r.start_work_date})
            r.employee_id.sudo().write(update_info)

        return True

    def reject(self):
        for r in self:
            related_user_id = r.employee_id.user_id
            related_partner_id = related_user_id.partner_id
            if r.old_emp:
                r.employee_id.sudo().active = False
                related_user_id.sudo().active = False
                related_partner_id.sudo().active = False
            else:
                r.employee_id.unlink()
                related_user_id.sudo().unlink()
                related_partner_id.sudo().unlink()
                r.code = False
                r.barcode = False
            # send email to IT, cc HR de xoa account va loai bo cac giay to
            r.reject_work_email_action()
            r.state = 'rejected'
        return True

    def to_draft(self):
        if self._context.get('allow_admin', False) and not self.env.user.has_group('base.group_system'):
            raise ValidationError(_('Administrator access is required!'))
        for r in self:
            related_user_id = r.employee_id.user_id
            related_partner_id = related_user_id.partner_id
            if r.old_emp:
                r.employee_id.active = False
                related_user_id.sudo().active = False
                related_partner_id.sudo().active = False
            else:
                r.employee_id.unlink()
                related_user_id.sudo().unlink()
                related_partner_id.sudo().unlink()
                r.code = False
                r.barcode = False
            r.start_work_date = False
            r.work_email = False
            r.email_count = 0
            r.state = 'draft'

    @api.constrains('account', 'old_emp', 'is_import')
    def validate_account(self):
        for r in self:
            account = r.account.lower()
            if not r.is_import and self.request_type not in ['old', 'internship'] and \
                    self.env['hr.employee'].search(['|', ('active', '=', False), ('active', '=', True),
                                                    ('account', '=', account)], limit=1):
                accounts = self.env['hr.employee'].search(['|', ('active', '=', False), ('active', '=', True)
                                                              , ('account', 'ilike', account + '%')]).mapped('account')
                mess = "The same exited account is: \n"
                for account in accounts:
                    mess += f"{account} \n"
                raise ValidationError(f"Account {account} already exits!\n "
                                      f" {mess} ")

            if self.search_count([('account', '=', account),
                                  ('state', 'in', ['submit', 'onboard']),
                                  ('expected_date', '=', r.expected_date),
                                  ('id', '!=', r.id)]) > 0:
                raise ValidationError(_('Error! Overlap account logs in the same day.'))

    @api.constrains('account')
    def validate_account_length(self):
        for r in self:
            if r.account and len(r.account) > 20:
                raise ValidationError("The maximum number of characters supported in Active Directory (AD) "
                                      "for user account is 20.")

    @api.constrains('personal_email')
    def validate_mail(self):
        for r in self:
            if r.personal_email and not r.old_emp:
                personal_email = r.personal_email.lower()
                if not re.match('^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$',
                                personal_email):
                    raise ValidationError(f'Email: {personal_email} are invalid E-mail')


    def create_employee(self):
        employee_obj = self.env['hr.employee']
        company_email = self.company_id.email
        work_email = False
        if company_email and company_email.find('@') >= 0:
            work_email = self._get_email_by_company()

        if self.account_type not in ['email_account', 'email']:
            work_email = self.personal_email
        self.work_email = work_email

        vals = {
            'active': False,
            'name': self.full_name,
            'personal_email': self.personal_email,
            'internship': self.internship,
            'account': self.account,
            'job_id': self.job_id.id,
            'department_id': self.department_id.id,
            'manager_id': self.department_id.manager_id.id or False,
            'company_id': self.company_id.id,
            'work_location_id': self.work_location_id.id,
            'old_emp_type': self.old_emp_type,

        }
        if self.request_type == 'internship':
            new_internship_code = self.employee_id.get_new_employee_code()
            vals.update({'code': new_internship_code})
        if self.employee_id and (self.old_emp or self.request_type == 'internship'):
            self.employee_id.sudo().write(vals)
        else:
            user_id = self.sudo().create_user(work_email)
            vals.update({
                'user_id': user_id.id,
                'work_email': work_email,
                'address_home_id': user_id.partner_id.id,
            })
            self.employee_id = employee_obj.create(vals)
        self.employee_id.set_password()

    def create_user(self, email):
        hr_employee = self.env.ref('hr_core.group_hr_client_user') or False
        return self.env['res.users'].with_context({
            'default_customer': False,
            'no_reset_password': True
        }).create({
            'active': True,
            'company_id': self.company_id.id,
            'company_ids': [(6, 0, self.company_id.ids)],
            'groups_id': [(6, 0, [hr_employee.id])],
            'lang': 'en_US',
            'tz': 'Asia/Ho_Chi_Minh',
            'name': self.full_name,
            'login': self.account,
            'email': email
        })

    @api.onchange('company_id')
    def _onchange_company_id(self):
        if self.company_id:
            company = self.company_id
        else:
            company = self.env.user.company_id
        self.department_id = False
        self.job_id = False
        return {
            'domain': {
                'department_id': [('company_id', 'in', [company.id, False])],
                'job_id': [('company_id', 'in', [company.id, False])]
            }
        }

    def _get_email_cc(self):
        self.ensure_one()
        return ', '.join([self.user_id.partner_id.email or '',
                          self.company_id.hr_email or '',
                          self.company_id.rec_email or ''])

    def _get_email_by_company(self):
        company_email = self.company_id.email
        if not company_email or company_email.find('@') == -1:
            raise ValidationError(f"Company email mismatch! "
                                  f"please config email for company {self.company_id.name} first!")
        return self.account.lower() + company_email[company_email.find('@'):]

    @api.multi
    def write(self, vals):
        new_expected_date = vals.get('expected_date')
        if new_expected_date:
            for r in self:
                r.old_expected_date = r.expected_date
                r.start_work_date = new_expected_date
        return super(HrEmployeeNewLines, self).write(vals)

    @api.model
    def create(self, vals):
        vals.update({'start_work_date': vals.get('expected_date', False)})
        res = super(HrEmployeeNewLines, self).create(vals)
        if vals.get('old_emp', False):
            res.barcode = res.employee_id.barcode
            res.code = res.employee_id.code
        return res

    @api.multi
    def unlink(self):
        not_draft_ids = self.filtered(lambda r: r.state != 'draft')
        if not_draft_ids:
            mess = ""
            for not_draft_id in not_draft_ids:
                mess += f"Employee: {not_draft_id.full_name}, record id: {not_draft_id.id} \n"
            raise ValidationError(f"Cannot delete record are not in draft state!\n "
                                  f" {mess} ")
        return super(HrEmployeeNewLines, self).unlink()

    def action_url(self, controller='/'):
        base_url = request.env['ir.config_parameter'].sudo().get_param('web.base.url')
        self.ensure_one()
        action_url = self._notify_get_action_link('controller', controller=controller)
        return urls.url_join(base_url, action_url)

    def it_checked_url(self):
        return self.action_url(controller='/employee_new/checked')
