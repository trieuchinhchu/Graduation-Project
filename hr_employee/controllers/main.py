# -*- coding: utf-8 -*-
import logging
import json, logging
import werkzeug
from datetime import datetime

_logger = logging.getLogger(__name__)
from odoo import fields, http, tools, _
from odoo.http import Controller, Response, request
from odoo.addons.mail.controllers.main import MailController
from odoo.tools import consteq, pycompat
from odoo.exceptions import UserError, ValidationError


class EmployeeFormController(http.Controller):
    MANDATORY_FIELDS = ["name", "place_of_birth", "home_address",
                        "street", "country_id", "state_id",
                        "work_phone", "personal_email","school_id",
                        "identification_id", "place_for_id",
                        "emergency_contact", "emergency_relation", "emergency_phone",
                        ]
    OPTIONAL_FIELDS = ["gender", "passport_id", "street2", "city","account","id","place_for_passport",
                       "account_number", "sinid", "vat", "zipcode","social_facebook","skill_id","home_disctrict",
                       "current_district","home_state","home_country",
                       "certificate", "study_field","vehicle_color",
                       "marital", "spouse_complete_name", "spouse_job","spouse_name","spouse_year",
                       "children","manufacturer", "vehicle_name", "vehicle_license", "vehicle_color", "graduation_year",
                       "password_new_employee","skill_list", "birthday", "id_date", "passport_date", "spouse_birthdate"]
    DATE_FIELDS = []

    # DEPENDENT_PEOPLE_FIELDS = ["name", "date_of_birth", "study_school"]

    @classmethod
    def _check_token(cls, base_link, token, res_id):
        if not res_id:
            res_id = request.params.get('res_id', False)
        if not base_link:
            base_link = request.httprequest.path
        params = {'res_id': res_id}
        valid_token = request.env['mail.thread']._notify_encode_link(base_link, params)
        return consteq(valid_token, str(token))

    @http.route('/my/info/update', type='http', auth="public", methods=['POST'], website=True)
    def employee_update_info(self, **post):
        values = {}
        keys = post.keys()
        childs_id =  []
        if 'childs_id[]' in keys:
            childs_name = post['childs[]'].split(',')
            childs_id = post['childs_id[]'].split(',')
            childs_study = post['childs_study[]'].split(',')
            childs_date = post['childs_date[]'].split(',')
        myname = []
        if 'myname[]' in keys:
            myname = post['myname[]'].split(',')
            myschool = post['myschool[]'].split(',')
            mytext = post['mytext[]'].split(',')
        token = request.session.get('token', False)
        res_id = request.session.get('res_id', False)
        employee_obj = request.env['hr.employee']
        comparison = self._check_token('/my/info', token, res_id)
        date_format = request.env['res.lang']._lang_get(request.env.lang).date_format
        if not comparison and not request.session.uid:
            return request.redirect('/web/login?redirect=/my/info/')

        if comparison and res_id:
            record = employee_obj.sudo().browse(int(res_id))
        elif request.session.uid:
            user_id = request.env.user
            record = employee_obj.sudo().search([('user_id', '=', user_id.id)], limit=1)
        if not record:
            raise werkzeug.exceptions.NotFound()
        i = 0
        for item in childs_id:
            data = record.dependent_ids.filtered(lambda x: x.id == int(item))
            if data:
                data[0].write({'name': childs_name[i],
                                'study_school': childs_study[i],
                                'date_of_birth': childs_date[i]})
            i = i + 1
        j = 0
        j = 0
        for item in myname:
            record.dependent_ids = [(0, 0, {
                'name': myname[j],
                'study_school': mytext[j],
                'date_of_birth': myschool[j]
            })]
            j = j + 1
        if 'childs_id[]' in keys:
            del post['childs[]']
            del post['childs_id[]']
            del post['childs_study[]']
            del post['childs_date[]']
        if 'myname[]' in keys:
            del post['myname[]']
            del post['myschool[]']
            del post['mytext[]']
        values.update(post)
        values = {key: post[key] for key in self.MANDATORY_FIELDS}
        values.update({key: post[key] for key in self.OPTIONAL_FIELDS if post.get(key, False)})
        for key in self.DATE_FIELDS:
            if post.get(key, False):
                values.update({key: datetime.strptime(post[key], date_format).date()})
        values.update({'zip': values.pop('zipcode', '')})
        values.update({'state_id': int(values.pop('state_id', 1))})
        values.update({'country_id': int(values.pop('country_id', 1))})
        values.update({'emp_info_state': 'wait_to_hr_confirm'})
        record.sudo().write(values)
        return json.dumps({'id': record.id})

    @http.route(['/my/info',
                 '/my/info?#res_id={}&<string:token>'],
                auth='public', method='GET', type='http', website=True, csrf=True)
    def get_current_employee_user(self, res_id=False, token=None, redirect=None, **kwargs):
        comparison, record, redirect = MailController._check_token_and_record_or_redirect('hr.employee', int(res_id),
                                                                                          token)
        request.session['res_id'] = res_id
        request.session['token'] = token
        values = {}
        employee_obj = request.env['hr.employee']
        date_format = request.env['res.lang']._lang_get(request.env.lang).date_format
        # time_format = request.env['res.lang']._lang_get(request.env.user.lang).time_format
        # dependent_people_obj = request.env['hr.dependent.people']
        if not comparison or not record:
            if not request.session.uid:
                raise werkzeug.exceptions.NotFound()
                return request.redirect('/web/login?redirect=/my/info/')
                # return http.local_redirect('/web/login?redirect=/my/info/')
                # return werkzeug.utils.redirect('/web/login', 303)
            user_id = request.env.user
            record = employee_obj.sudo().search([('user_id', '=', user_id.id)], limit=1)
            values.update({
                'default_country_id': user_id.partner_id.country_id.id,
                'default_state_id': user_id.partner_id.state_id.id,
                'error': {},
                'error_message': [],
            })

        countries = request.env['res.country'].sudo().search([])
        states = request.env['res.country.state'].sudo().search([])
        districts = request.env['res.district'].sudo().search([])
        values.update({key: record.sudo()[f'{key}'] for key in self.MANDATORY_FIELDS + self.OPTIONAL_FIELDS if
                       record and key in record})
        values.update(
            {key: record.sudo()[f'{key}'] and record.sudo()[f'{key}'].strftime(date_format) for key in self.DATE_FIELDS
             if record and key in record})
        values.update({
            'countries': countries,
            'states': states,
            'partner': request.env.user.partner_id,
            'districts': districts
            # 'allow_update': False,
        })
        values.update({'zipcode': record.sudo().zip})
        response = request.render("hr_employee.employee_input_form_template", values)
        response.headers['X-Frame-Options'] = 'DENY'
        return response
