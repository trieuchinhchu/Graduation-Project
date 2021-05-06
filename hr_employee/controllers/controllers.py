# -*- coding: utf-8 -*-
from odoo import http

# class Attendance(http.Controller):
#     @http.route('/attendance/attendance/', auth='public')
#     def index(self, **kw):
#         return "Hello, world"

#     @http.route('/attendance/attendance/objects/', auth='public')
#     def list(self, **kw):
#         return http.request.render('attendance.listing', {
#             'root': '/attendance/attendance',
#             'objects': http.request.env['attendance.attendance'].search([]),
#         })

#     @http.route('/attendance/attendance/objects/<model("attendance.attendance"):obj>/', auth='public')
#     def object(self, obj, **kw):
#         return http.request.render('attendance.object', {
#             'object': obj
#         })