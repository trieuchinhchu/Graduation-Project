# -*- coding: utf-8 -*-
from odoo import http

# class ResourceCalendar(http.Controller):
#     @http.route('/resource_calendar/resource_calendar/', auth='public')
#     def index(self, **kw):
#         return "Hello, world"

#     @http.route('/resource_calendar/resource_calendar/objects/', auth='public')
#     def list(self, **kw):
#         return http.request.render('resource_calendar.listing', {
#             'root': '/resource_calendar/resource_calendar',
#             'objects': http.request.env['resource_calendar.resource_calendar'].search([]),
#         })

#     @http.route('/resource_calendar/resource_calendar/objects/<model("resource_calendar.resource_calendar"):obj>/', auth='public')
#     def object(self, obj, **kw):
#         return http.request.render('resource_calendar.object', {
#             'object': obj
#         })