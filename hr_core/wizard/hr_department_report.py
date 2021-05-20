# -*- coding: utf-8 -*-

from odoo import fields, models, api, tools


class HrDepartmentChangeReport(models.Model):
    _name = 'hr.department.change.report'
    _description = 'Hr department changed Report'
    _rec_name = 'employee_id'
    _auto = False
    _order = 'date_update desc'

    author_id = fields.Many2one('res.partner', 'Update By')
    employee_id = fields.Many2one('hr.employee', 'Employee')
    old_department_id = fields.Many2one('hr.department', 'Old Department')
    new_department_id = fields.Many2one('hr.department', 'New Department')
    date_update = fields.Datetime('Update Time')

    @api.model_cr
    def init(self):
        tools.drop_view_if_exists(self.env.cr, viewname=self._table)
        self.env.cr.execute("""
           create or replace view hr_department_change_report as (
               select row_number() over (ORDER BY mm.res_id)  as id,
                    mm.author_id             as author_id,
                    mm.res_id             as employee_id,
                    mtv.old_value_integer as old_department_id,
                    mtv.new_value_integer as new_department_id,
                    mm.create_date        as date_update
                    from mail_message mm
                        inner join mail_tracking_value mtv on mm.id = mtv.mail_message_id
                    where mm.model = 'hr.employee' and field = 'department_id'
           ) 
        """)
