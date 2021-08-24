# -*- coding: utf-8 -*-

from odoo import models, fields, api, _


class HRDepartmentInherit(models.Model):
    _inherit = 'hr.department'
    _description = 'Department'

    @api.model
    def name_search(self, name, args=None, operator='ilike', limit=100):
        if self._context.get('view_all', False):
            self = self.sudo()
        return super(HRDepartmentInherit, self).name_search(name, args, operator, limit)

    @api.model
    def search_read(self, domain=None, fields=None, offset=0, limit=None, order=None):
        if self._context.get('view_all', False):
            self = self.sudo()
        return super(HRDepartmentInherit, self).search_read(domain, fields, offset, limit, order)

    @api.model
    def read_group(self, domain, fields, groupby, offset=0, limit=None, orderby=False, lazy=True):
        if self._context.get('view_all', False):
            self = self.sudo()
        return super(HRDepartmentInherit, self).read_group(domain, fields, groupby, offset=offset, limit=limit,
                                                            orderby=orderby, lazy=lazy)