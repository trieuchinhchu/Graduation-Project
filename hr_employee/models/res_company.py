# -*- coding: utf-8 -*-

from unidecode import unidecode

from odoo import models, fields, api, _


class DACompanyConfig(models.Model):
    _inherit = "res.company"
    _description = "DA Company Config"

    welcome_template_id = fields.Many2one("mail.template", "Welcome mail template")