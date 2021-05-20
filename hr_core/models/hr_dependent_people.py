# -*- coding: utf-8 -*-

from odoo import models, api, fields, _
from odoo.exceptions import ValidationError


class HrDependentPeople(models.Model):
    _name = 'hr.dependent.people'
    _description = 'Employee Dependent People'

    name = fields.Char('Name', required=True)
    date_of_birth = fields.Date("Date Of Birth", required=True)
    parent_id = fields.Many2one('hr.employee', 'Employee', required=True)
    study_school = fields.Char("School")
    country_id = fields.Many2one('res.country', string='Country', ondelete='restrict', required=False)
    state_id = fields.Many2one("res.country.state", string='State', ondelete='restrict',
                               domain="[('country_id', '=?', country_id)]",
                               required=False)
    city = fields.Char('City', readonly=False)
    zip = fields.Char('Zip', change_default=True)
    street2 = fields.Char(string='Street2', readonly=False, required=False)
    street = fields.Char(string='Street', readonly=False, required=False)
    active = fields.Boolean('Active', default=True)
    is_dependent = fields.Boolean('người phụ thuộc', default=False)

    has_id = fields.Boolean('Người phụ thuộc đã có MST hoặc chưa có mã số thuế nhưng có CMND/CCCD/Hộ chiếu')
    sinid = fields.Char(string='SIN No', help='Social Insurance Number')
    identification_id = fields.Char(string='Identification No')
    passport_id = fields.Char('Passport No')
    birth_no = fields.Char('Số (giấy khai sinh)')
    book_no = fields.Char('Quyển số (giấy khai sinh)')

    relation = fields.Char('Relationship', required=False)
    register_date = fields.Date('Register Date', help="Thời điểm bắt đầu tính giảm trừ (tháng/năm)", required=False)
    end_date = fields.Date('End Date', help="Thời điểm kết thúc tính giảm trừ (tháng/năm)", required=False)

    @api.constrains('register_date', 'end_date')
    def _check_valid_time(self):
        for line in self:
            if line.register_date and line.end_date and line.register_date >= line.end_date:
                raise ValidationError(_('Error! End date must be greater than register date.'))

    @api.onchange('register_date')
    def onchange_register_date(self):
        if self.register_date:
            self.register_date = self.register_date.replace(day=1)

    @api.onchange('end_date')
    def onchange_end_date(self):
        if self.end_date:
            self.end_date = self.end_date.replace(day=1)
