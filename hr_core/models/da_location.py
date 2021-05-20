# -*- coding: utf-8 -*-

from werkzeug import urls

from odoo import models, api, fields, _
from odoo.exceptions import ValidationError
from odoo.http import request


class DAEmployeeLocation(models.Model):
    _name = "da.location"
    _description = "Employee location"
    _order = "sequence, name"

    name = fields.Char('Location', required=True)
    detail = fields.Text('Location Detail')
    sequence = fields.Integer('Sequence', default=1)
    company_id = fields.Many2one('res.company', string='Company')

    _sql_constraints = [
        ('da_location_sequence', 'CHECK(sequence >= 0)', 'Sequence number MUST be a natural')
    ]