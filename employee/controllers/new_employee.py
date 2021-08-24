# -*- coding: utf-8 -*-
import logging

from odoo.addons.mail.controllers.main import MailController
from odoo import http
from odoo.http import request

_logger = logging.getLogger(__name__)


class NewEmployeeController(http.Controller):

    @http.route('/employee_new/<int:id>/', type='http', auth="user", website=True)
    def get_employee_new_url(self, id):
        menu_id = request.env.ref('hr_employee.menu_hr_employee_new_request').id
        action_id = request.env.ref('hr_employee.action_hr_employee_new_request').id
        link_format = "/web?#id={}&view_type=form&model=hr.employee.new&menu_id={}&action={}"
        link = link_format.format(id, menu_id, action_id)
        return request.redirect(link)

    @http.route('/employee_new/checked', type='http', auth='user', methods=['GET'])
    def da_employee_new_it_checked(self, res_id, token):
        comparison, record, redirect = MailController._check_token_and_record_or_redirect('hr.employee.new',
                                                                                          int(res_id), token)
        if comparison and record:
            try:
                record.action_it_checked()
            except Exception:
                return MailController._redirect_to_messaging()
        return redirect

