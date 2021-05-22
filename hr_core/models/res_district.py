# -*- coding: utf-8 -*-
# Part of Odoo. See LICENSE file for full copyright and licensing details.

import logging
from odoo import api, fields, models

_logger = logging.getLogger(__name__)


class CountryState(models.Model):
    _description = "District"
    _name = 'res.district'
    _order = 'code'

    state_id = fields.Many2one('res.country.state', string='State', required=True)
    name = fields.Char(string='District Name', required=True,
               help='Administrative divisions of a State. E.g. Fed. District, Departement, Canton')
    code = fields.Char(string='District Code', help='The district code.', required=True)

    _sql_constraints = [
        ('name_code_uniq', 'unique(state_id, code)', 'The code of the District must be unique by state !')
    ]

    @api.multi
    def name_get(self):
        result = []
        for record in self:
            result.append((record.id, "{} ({})".format(record.name, record.state_id.code)))
        return result


class Manufacturer(models.Model):
    _description = "Manufacturer"
    _name = 'res.manufacturer'

    name = fields.Char(string='Manufacturer Name', required=True)