# -*- coding: utf-8 -*-
# Part of Odoo. See LICENSE file for full copyright and licensing details.

import re
import logging
from odoo import api, fields, models
from odoo.osv import expression
from psycopg2 import IntegrityError
from odoo.tools.translate import _
_logger = logging.getLogger(__name__)

class SchoolName(models.Model):
    _description = "School"
    _name = 'res.school'
    _order = 'name'

    name = fields.Char(string='Schoool Name', required=True)
    code = fields.Char(string='Schoool Code', help='The schoool code.')