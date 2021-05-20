# -*- coding: utf-8 -*-

from unidecode import unidecode

from odoo import models, fields, api, _


class DACompanyConfig(models.Model):
    _inherit = "res.company"
    _description = "DA Company Config"

    welcome_template_id = fields.Many2one("mail.template", "Welcome mail template")


class RecEmployeeInherit(models.Model):
    _inherit = 'res.users'

    @api.model
    def name_search(self, name, args=None, operator='ilike', limit=100):
        recs = super(RecEmployeeInherit, self).name_search(name, args, operator, limit)
        if self._context.get('hr_rec', False):
            user_ids = self.env.ref('hr_core.group_hr_rec').users.ids + \
                       self.env.ref('hr_recruitment.group_hr_recruitment_manager').users.ids
            args += ['|', ('login', operator, unidecode(name.strip())), ('name', operator, unidecode(name.strip())),
                     ('id', 'in', user_ids)]
            return self.search(args, limit=limit).name_get()
        return recs

    @api.model
    def search_read(self, domain=None, fields=None, offset=0, limit=None, order=None):
        if self._context.get('hr_rec', False):
            user_ids = self.env.ref('hr_core.group_hr_rec').users.ids + \
                       self.env.ref('hr_recruitment.group_hr_recruitment_manager').users.ids
            domain.extend([('id', 'in', user_ids)])
        return super(RecEmployeeInherit, self).search_read(domain, fields, offset, limit, order)

    @api.model
    def read_group(self, domain, fields, groupby, offset=0, limit=None, orderby=False, lazy=True):
        if self._context.get('hr_rec', False):
            user_ids = self.env.ref('hr_core.group_hr_rec').users.ids + \
                       self.env.ref('hr_recruitment.group_hr_recruitment_manager').users.ids
            domain.extend([('id', 'in', user_ids)])
        return super(RecEmployeeInherit, self).read_group(domain, fields, groupby, offset=offset, limit=limit,
                                                             orderby=orderby, lazy=lazy)