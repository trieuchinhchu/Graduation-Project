# -*- coding: utf-8 -*-
from odoo import http

# class Da2021(http.Controller):
#     @http.route('/da2021/da2021/', auth='public')
#     def index(self, **kw):
#         return "Hello, world"

#     @http.route('/da2021/da2021/objects/', auth='public')
#     def list(self, **kw):
#         return http.request.render('da2021.listing', {
#             'root': '/da2021/da2021',
#             'objects': http.request.env['da2021.da2021'].search([]),
#         })

#     @http.route('/da2021/da2021/objects/<model("da2021.da2021"):obj>/', auth='public')
#     def object(self, obj, **kw):
#         return http.request.render('da2021.object', {
#             'object': obj
#         })