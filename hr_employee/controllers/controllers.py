# -*- coding: utf-8 -*-
from odoo import http
from odoo.http import request
import json
import logging

_logger = logging.getLogger(__name__)

class WebsiteSearch(http.Controller):
    @http.route('/new_employee/info/<int:id>/', type='http', auth="user", website=True)
    def get_new_employee_skill_infor(self, id):
        menu_id = request.env.ref('hr_core.menu_open_view_employee_list_my').id
        action_id = request.env.ref('hr_employee.da_skill_employee_action').id
        link_format = "/web#id={}&action={}&model=hr.employee&view_type=form&menu_id={}"
        link = link_format.format(id, action_id, menu_id)
        return request.redirect(link)

    @http.route('/employee/skill', csrf=False, auth='public', website=True, type='http', methods=['GET'])
    def suggest_search(self, keywords, **params):
        """
        Search products and categories that match keywords
        :param keywords: search query
        :return: json
        """
        if not keywords:
            return json.dumps([])

        Product = request.env['hr.skill'].with_context(bin_size=True)
        
        if not keywords or keywords == '':
            domain = [('id', '!=', False)]
        else:
            domain = [('name_display', 'ilike', keywords)]
        products = Product.sudo().search(domain, limit=10)
        products = [dict(id=p.id, name=p.name_display) for p in products]
        _logger.debug(products)
        return json.dumps(products)

    @http.route('/employee/school', csrf=False, auth='public', website=True, type='http', methods=['GET'])
    def suggest_search_school(self, keywords, **params):
        """
        Search products and categories that match keywords
        :param keywords: search query
        :return: json
        """
        if not keywords:
            return json.dumps([])

        Product = request.env['res.school'].with_context(bin_size=True)
        
        if not keywords or keywords == '':
            domain = [('id', '!=', False)]
        else:
            domain = [('name', 'ilike', keywords)]
        products = Product.sudo().search(domain, limit=10)
        products = [dict(id=p.id, name=p.name) for p in products]
        _logger.debug(products)
        return json.dumps(products)

    @http.route('/employee/country', csrf=False, auth='public', website=True, type='http', methods=['GET'])
    def suggest_search_country(self, keywords, **params):
        """
        Search products and categories that match keywords
        :param keywords: search query
        :return: json
        """
        if not keywords:
            return json.dumps([])

        Product = request.env['res.country'].with_context(bin_size=True)
        
        if not keywords or keywords == '':
            domain = [('id', '!=', False)]
        else:
            domain = [('name', 'ilike', keywords)]
        products = Product.sudo().search(domain, limit=10)
        products = [dict(id=p.id, name=p.name) for p in products]
        _logger.debug(products)
        return json.dumps(products)

    @http.route('/employee/state', csrf=False, auth='public', website=True, type='http', methods=['GET'])
    def suggest_search_state(self, keywords, **params):
        """
        Search products and categories that match keywords
        :param keywords: search query
        :return: json
        """
        if not keywords:
            return json.dumps([])

        Product = request.env['res.country.state'].with_context(bin_size=True)
        
        if not keywords or keywords == '':
            domain = [('id', '!=', False)]
        else:
            domain = [('name', 'ilike', keywords)]
        products = Product.sudo().search(domain, limit=10)
        products = [dict(id=p.id, name=p.name) for p in products]
        _logger.debug(products)
        return json.dumps(products)

    @http.route('/employee/district', csrf=False, auth='public', website=True, type='http', methods=['GET'])
    def suggest_search_district(self, keywords, **params):
        """
        Search products and categories that match keywords
        :param keywords: search query
        :return: json
        """
        if not keywords:
            return json.dumps([])

        Product = request.env['res.district'].with_context(bin_size=True)
        
        if not keywords or keywords == '':
            domain = [('id', '!=', False)]
        else:
            domain = [('name', 'ilike', keywords)]
        products = Product.sudo().search(domain, limit=10)
        products = [dict(id=p.id, name=p.name) for p in products]
        _logger.debug(products)
        return json.dumps(products)