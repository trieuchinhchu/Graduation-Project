# -*- coding: utf-8 -*-

from odoo import models, fields, api


class WizardDAEmployee(models.TransientModel):
    _name = 'wizard.da.employee'
    _description = "Create User for selected Employee(s)"

    def _get_default_emp(self):
        if self.env.context and self.env.context.get('active_ids'):
            return self.env.context.get('active_ids')
        return []

    employee_ids = fields.Many2many('hr.employee', default=_get_default_emp, string='Employee(s)')

    @api.multi
    def create_user(self):
        active_ids = self.env.context.get('active_ids', []) or []
        records = self.env['hr.employee'].browse(active_ids)
        records.create_employee_user()
