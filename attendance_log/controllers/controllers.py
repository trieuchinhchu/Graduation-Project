# -*- coding: utf-8 -*-
from odoo import http

# class AttendanceLog(http.Controller):
#     @http.route('/attendance_log/attendance_log/', auth='public')
#     def index(self, **kw):
#         return "Hello, world"

#     @http.route('/attendance_log/attendance_log/objects/', auth='public')
#     def list(self, **kw):
#         return http.request.render('attendance_log.listing', {
#             'root': '/attendance_log/attendance_log',
#             'objects': http.request.env['attendance_log.attendance_log'].search([]),
#         })

#     @http.route('/attendance_log/attendance_log/objects/<model("attendance_log.attendance_log"):obj>/', auth='public')
#     def object(self, obj, **kw):
#         return http.request.render('attendance_log.object', {
#             'object': obj
#         })