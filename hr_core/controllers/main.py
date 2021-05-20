# -*- coding: utf-8 -*-
from odoo import http
from odoo.http import request


class Controller(http.Controller):
    @http.route('/employee/<int:id>/avatar/', type='http', auth="public", website=True)
    def get_employee_image(self, id):
        link_format = "/web/image?model=hr.employee&id={}&field=image_medium"
        link = link_format.format(id)
        return request.redirect(link)

    @http.route('/employee/<int:id>/', type='http', auth="user", website=True)
    def get_employee_id(self, id):
        menu_id = request.env.ref('hr_core.menu_open_view_employee_list_my').id
        action_id = request.env.ref('hr.open_view_employee_list_my').id
        link_format = "/web?#id={}&view_type=form&model=hr.employee&menu_id={}&action={}"
        link = link_format.format(id, menu_id, action_id)
        return request.redirect(link)