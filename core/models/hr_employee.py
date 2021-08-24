# -*- coding: utf-8 -*-

from unidecode import unidecode
from datetime import datetime
from odoo import models, api, fields, _
from odoo.exceptions import ValidationError
from odoo.addons.base.models.ir_mail_server import MailDeliveryException

_CERTIFICATE = [('master', 'Trên đại học'),
                ('university', 'Đại học'),
                ('colleges', 'Cao đẳng'),
                ('vocational', 'Trung cấp'),
                ('high_school', 'THPT'),
                ('other', 'Khác')]

# class DAEmployeeDoc(models.Model):
#     _name = 'da.employee.docs'
#     _description = 'HR employee documents'
#
#     name = fields.Char('Name')


class HrEmployeeInherit(models.Model):
    _inherit = 'hr.employee'
    _description = 'HR employee inherit base'

    def _default_work_location(self):
        try:
            default_location = self.env['da.location'].search([], limit=1).id
        except Exception:
            return False
        return default_location

    display_name = fields.Char(compute='_compute_display_name', store=False)
    code = fields.Char(string="Employee Code", track_visibility='onchange', default='New')
    job_id = fields.Many2one('hr.job', 'Job Position', track_visibility='onchange')
    home_address = fields.Char('Home Address')
    work_email = fields.Char('Work Email', copy=False)
    account = fields.Char('Account', copy=False, readonly=False)
    manager = fields.Boolean(string='Is a Manager')
    department_id = fields.Many2one('hr.department', string='Department', track_visibility='onchange')
    start_work_date = fields.Date(string='Hire Date', required=False, track_visibility='onchange',
                                  help="start work date is normally the date when someone begins working ... ")
    hire_date = fields.Date(string='Hire Date 2', required=False, track_visibility='onchange',
                            help="Hire Date 2 is normally the date when someone begins working ... ")
    company_id = fields.Many2one('res.company', string='Company', default=lambda self: self.env.user.company_id)
    identification_id = fields.Char(string='Identification No',
                                    groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    tax_code = fields.Char(string='Tax Code')
    id_tax = fields.Char(string='ID Registration Tax')
    passport_id = fields.Char('Passport No', groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    place_for_id = fields.Char('Place for ID', help="Place for identity cards")
    place_for_passport = fields.Char('Place for Passport')
    id_date = fields.Date('ID Date')
    passport_date = fields.Date('Passport Date')
    sinid = fields.Char('SIN No', help='Social Insurance Number',
                        groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    vat = fields.Char(string='Tax ID', help="The Tax Identification Number. "
                                            "Complete it if the contact is subjected to government taxes. "
                                            "Used in some legal statements.")
    address_home_id = fields.Many2one(
        'res.partner', 'Private Address',
        help='Enter here the private address of the employee, not the one linked to your company.',
        groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    street = fields.Char(related='address_home_id.street', readonly=False)
    street2 = fields.Char(related='address_home_id.street2', readonly=False)
    place_of_birth = fields.Char('Place of Birth',
                                 groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    zip = fields.Char(related='address_home_id.zip', readonly=False, change_default=True)
    city = fields.Char(related='address_home_id.city', readonly=False)
    country_id = fields.Many2one('res.country', 'Nationality (Country)',
                                 groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    country = fields.Many2one(related='address_home_id.country_id', string='Country', ondelete='restrict',
                              readonly=False, store=True)
    state_id = fields.Many2one(related='address_home_id.state_id', readonly=False, string='State', ondelete='restrict',
                               domain="[('country_id', '=?', country_id)]")

    personal_email = fields.Char('Personal Email', copy=False)
    certificate = fields.Selection(_CERTIFICATE, 'Certificate Level',
                                   default='university',
                                   groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    graduation_year = fields.Char('Graduation Year')
    emergency_relation = fields.Char('Emergency Relation')
    account_number = fields.Char(string='Bank Account Number',
                                 placeholder="Bank Account Number",
                                 help="Technical field used to store the bank account number before its creation, "
                                      "upon the line's processing")
    # document_ids = fields.Many2many('da.employee.docs', 'employee_document_rel', 'doc_id', 'employee_id',
    #                                 string='Employee Documents')
    work_location_id = fields.Many2one('da.location', default=_default_work_location)
    work_location = fields.Selection([('dp', 'Tầng 6 tòa nhà Đại Phát'),
                                      ('other', 'Địa điểm khác')], required=False, default='dp')
    contract_id = fields.Many2one('hr.contract', compute='_compute_contract_id', string='Current Contract',
                                  help='Latest contract of the employee',
                                  groups="hr_contract.group_hr_contract_manager")
    resignation = fields.Boolean(string="Resignation")
    resignation_date = fields.Date(string="Resignation Date",
                                   help="Resignation Date means the date specified in the Resignation Notice, or the actual date the Executive terminates employment with the Company as the result of a resignation as provided in whichever occurs earlier.")
    resignation_reason = fields.Char("Resignation Reason")
    resignation_note = fields.Text("Resignation Notes")
    social_facebook = fields.Char(string="Socical Facebook")
    home_country = fields.Many2one(string='Home Country', comodel_name='res.country', ondelete='set null')
    home_state = fields.Many2one(string='Home State', comodel_name='res.country.state', ondelete='set null')
    home_city = fields.Char(string="Home City")
    home_disctrict = fields.Many2one(string='Home District', comodel_name='res.district', ondelete='set null')
    current_district = fields.Many2one(string='Current District', comodel_name='res.district', ondelete='set null')
    place_disctrict = fields.Many2one(string='Place District', comodel_name='res.district', ondelete='set null')
    place_country = fields.Many2one(string='Place Country', comodel_name='res.country', ondelete='set null')
    place_state = fields.Many2one(string='Place State', comodel_name='res.country.state', ondelete='set null')
    school_id = fields.Many2one(string='School', comodel_name='res.school', ondelete='set null')
    gender = fields.Selection([
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other')], default="male", groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    birthday = fields.Date('Date of Birth', groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    permission_view = fields.Boolean(string="Permission", compute='_compute_view_record')
    study_field = fields.Char("Field of Study", placeholder='Computer Science',
                              groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    study_school = fields.Char("School",
                               groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    emergency_contact = fields.Char("Emergency Contact",
                                    groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user")
    emergency_phone = fields.Char("Emergency Phone",
                                  groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user",
                                  default="")
    marital = fields.Selection([
        ('single', 'Single'),
        ('married', 'Married'),
        ('cohabitant', 'Legal Cohabitant'),
        ('widower', 'Widower'),
        ('divorced', 'Divorced')
    ], string='Marital Status',
        groups="hr.group_hr_user,base.group_user,hr_core.group_hr_client_user",
        default='single')
    _sql_constraints = [
        ('work_email_uniq', 'unique(work_email)', 'Work email must be unique.')
    ]

    @api.multi
    def _compute_view_record(self):
        for r in self:
            if not self.env.user.has_group('base.group_system') and \
                    not self.env.user.has_group('hr.group_hr_user') and \
                    not self.env.user.has_group('hr.group_hr_manager') and \
                    self.env.user.id != r.user_id.id:
                raise ValidationError(_("You don't have permission view Infor"))
            else:
                r.permission_view = True

    @api.constrains('graduation_year')
    def _validate_graduation_year(self):
        for record in self:
            try:
                val = int(record.graduation_year)
            except ValueError:
                raise ValidationError(_("Graduated year must be a number"))

    @api.constrains('identification_id')
    def _validate_identification_id(self):
        for record in self:
            if record.identification_id and not (9 <= len(record.identification_id) <= 12):
                raise ValidationError(_("Identification must be greater than 9 and less than 12"))

    @api.constrains('id_date')
    def _validate_id_date(self):
        for record in self:
            now = datetime.today().date()
            if record.id_date and record.id_date > now:
                raise ValidationError(_("ID date cannot be greater than the current date"))

    @api.constrains('passport_date')
    def _validate_passport_date(self):
        for record in self:
            now = datetime.today().date()
            if record.passport_date and record.passport_date > now:
                raise ValidationError(_("Passport date cannot be greater than the current date"))

    @api.constrains('birthday')
    def _validate_birthday(self):
        for record in self:
            now = datetime.today().date()
            if record.birthday and record.birthday > now:
                raise ValidationError(_("Birthday cannot be greater than the current date"))

    @api.multi
    def name_get(self):
        res = []
        for r in self:
            name = r.name
            department_id = r.sudo().department_id
            if department_id and department_id.name:
                name += f" ({department_id.name})"
            if r.work_email:
                name += f" <{r.work_email}>"
            res.append((r.id, name))
        return res

    @api.model
    def name_search(self, name, args=None, operator='ilike', limit=100):
        args = args or []
        name_decoded = unidecode(name.strip())
        args.extend(['|', '|',
                     ('account', operator, name_decoded),
                     ('work_email', operator, name_decoded),
                     ('name', operator, name),
                     ])
        return self.search(args, limit=limit).name_get()

    @api.onchange('country_id')
    def _onchange_country_id(self):
        self.country = self.country_id

    @api.onchange('resignation')
    def _onchange_resignation(self):
        if not self.resignation:
            self.resignation_date = False
        else:
            self.resignation_date = fields.Date.today()

    @api.depends('name', 'department_id', 'department_id.name')
    def _compute_display_name(self):
        for r in self:
            if r.name and r.department_id and r.department_id.sudo().name and r.name.find('(') == -1:
                r.display_name = '%s (%s)' % (r.name, r.department_id.name)
            else:
                r.display_name = r.name

    def check_valid_input(self, vals):
        hr_manager_ids = self.env.ref('hr.group_hr_manager').users.ids
        system_user_ids = self.env.ref('base.group_system').users.ids
        if self._uid not in hr_manager_ids + system_user_ids and not self._context.get('update_old_emp', False):
            mess = ''
            if vals.get('code', False):
                mess = mess + 'code \n'
            if vals.get('department_id', False):
                mess = mess + 'Department \n'
            if vals.get('job_id', False):
                mess = mess + 'Job Position \n'
            if vals.get('parent_id', False):
                mess = mess + 'Manager \n'
            if vals.get('coach_id', False):
                mess = mess + 'Coach \n'
            if vals.get('resource_calendar_id', False):
                mess = mess + 'Working Hours \n'
            if vals.get('work_email', False):
                mess = mess + 'Work Email \n'
            if mess != '':
                raise ValidationError(_(f'Permission denied! Only HR can update these value: \n {mess}'))

    @api.constrains('code')
    def check_duplicate_code(self):
        for record in self:
            domain = [
                ('code', '=', record.code),
                ('code', '!=', 'New'),
                ('id', '!=', record.id),
            ]
            overlaps = self.search_count(domain)
            if overlaps:
                raise ValidationError(_('The employee code must be unique per company!'))

    @api.multi
    def write(self, vals):
        self.check_valid_input(vals)
        return super(HrEmployeeInherit, self).write(vals)

    @api.multi
    def create_employee_user(self):
        user_group = self.env.ref('base.group_user') or False
        users_res = self.env['res.users']
        for record in self:
            if not self.env.user.has_group('hr.group_hr_manager'):
                raise ValidationError(_("Only Hr Manager can create users from employee!"))
            if not record.work_email:
                raise ValidationError(_(f"Employee {record.name} work email not found! "
                                        f"\nWork email is require to create an User. "))

            if not record.user_id:
                user_id = users_res.sudo().with_context(default_customer=False).create({
                    'name': record.name,
                    'login': record.work_email,
                    'active': True,
                    'company_id': record.company_id.id,
                    'company_ids': [(6, 0, [record.company_id.id])],
                    'groups_id': [(6, 0, [user_group.id])],
                    'lang': 'en_US',
                    'tz': 'Asia/Ho_Chi_Minh',
                    'email': record.work_email,
                })
                record.user_id = user_id
        return True

    @api.onchange('job_id')
    def _onchange_job_id(self):
        if self.job_id:
            self.job_title = self.job_id.job_title

    def action_reset_password(self):
        if not self.env.user.has_group('hr.group_hr_manager'):
            raise ValidationError(_('Permission denied! only HR Manager can do this action.'))
        for r in self:
            if not r.user_id:
                raise ValidationError(_('Data miss match! Employee do not have user to login.'))
            try:
                r.user_id.sudo().action_reset_password()
            except MailDeliveryException:
                raise ValidationError(_('Mail delivery false! Cannot send reset password to employee.'))


class HrJobInherit(models.Model):
    _inherit = 'hr.job'
    _description = 'DA Job'
    is_approve = fields.Boolean('Is Approve', default=False)
    job_title = fields.Char("Job Title")
    require_rank = fields.Boolean('Require Rank', default=False)

    @api.constrains('name', 'company_id', 'department_id')
    def check_duplicate_job(self):
        for record in self:
            domain = [
                ('name', '=', record.name),
                ('company_id', '=', record.company_id.id),
                ('department_id', '=', record.department_id.id),
                ('id', '!=', record.id),
            ]
            overlaps = self.search_count(domain)
            if overlaps:
                raise ValidationError(_('The name of the job position must be unique per department in company!'))

    @api.model
    def create(self, vals):
        if not self.env.user.has_group('hr.group_hr_manager'):
            raise ValidationError(_("Only Hr Manager can create job position!"
                                    "Contact admin for help!"))
        return super(HrJobInherit, self).create(vals)
