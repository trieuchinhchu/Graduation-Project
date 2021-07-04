# -*- coding: utf-8 -*-
# Part of Odoo. See LICENSE file for full copyright and licensing details.

import logging
from odoo import api, fields, models


class SchoolName(models.Model):
    _description = "School"
    _name = 'res.school'
    _order = 'name'

    name = fields.Char(string='Schoool Name', required=True)
    code = fields.Char(string='Schoool Code', help="Schoool's code.")