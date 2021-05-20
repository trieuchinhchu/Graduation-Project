# -*- coding: utf-8 -*-

import base64, logging

from xlrd import open_workbook
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from odoo import fields, models, api, _
from odoo.exceptions import ValidationError

_LOGGER = logging.getLogger(__name__)


class HrEmployeeUpdate(models.TransientModel):
    _name = 'hr.employee.update'
    _description = 'HR Employee Update Info'

    type = fields.Selection(selection=[('info', 'Thông tin nhân viên'),
                                       ('resignation', 'Thông tin nhân viên đã nghỉ việc'),
                                       ('dependent', 'Con cái / Người phụ thuộc'),
                                       ('contract', 'Thông tin hợp đồng'), ],
                            required=True, string='Upload type', default='info')

    file_save = fields.Binary(string='Attachment File', attachment=True, help='Update your file.', required=True)
    file_name = fields.Char(stirng='File Name')

    EMPLOYEE_FIELDS = ["code", "account", "name", "job_title", "gender", "work_phone", "identification_id",
                       "place_for_id", "place_for_passport", "account_number", "sinid", "vat", "place_of_birth","passport_id",
                       "home_address", "street", "certificate", "study_school", "study_field", "graduation_year",
                       "personal_email","social_facebook", "emergency_contact", "emergency_phone",
                       "spouse_complete_name", "spouse_job","spouse_name","spouse_year",
                       "work_location", "resignation_reason", "resignation_note"]

    DATE_FIELDS = ["birthday", "id_date", "passport_date", "start_work_date", "spouse_birthdate", "hire_date", "resignation_date"]

    EMPLOYEE_CONTRACT_FIELDS = ["account", "name", "duration", "date_start"]

    @api.one
    def update_employee_info(self):
        if self.type not in ['info', 'resignation']:
            return False
        all_fields = self.EMPLOYEE_FIELDS + self.DATE_FIELDS
        # Load data from xls file
        wb = open_workbook(file_contents=base64.decodestring(self.file_save))
        sheet_name = 'employee_info'
        if self.type == 'resignation':
            sheet_name = 'employee_resignation_info'
        try:
            s = wb.sheet_by_name(sheet_name)
        except Exception as e:
            raise ValidationError(f"Không tồn tại sheet: {sheet_name}. Xin hãy kiểm tra lại!\n Detail: {e}")

        if not s.nrows:
            raise ValidationError(_('No data found!'))
        if not set(all_fields).issubset(set(s.row_values(0))):
            raise ValidationError(_(f"Your file must contain at least these columns: "
                                    f"{', '.join(all_fields)}!"))
        datas = {}
        for col, col_num in zip(s.row_values(0), range(0, s.ncols)):
            if col in all_fields:
                col_value = []
                for row_num in range(2, s.nrows):
                    value = s.cell(row_num, col_num).value
                    try:
                        value = str(value)
                    except:
                        pass
                    col_value.append(value)
                    datas.update({col: col_value})
        self.employee_update_info(datas)

        return True

    def employee_update_info(self, datas):
        employee_ids = self.env['hr.employee'].search([('work_email', '!=', False),
                                                       '|', ('active', '=', False), ('active', '=', True)])
        employee_dict = {f'{employee_id.account}': employee_id for employee_id in employee_ids}
        i = -1
        certificate = { 'Trên đại học': 'master',
                        'Đại học': 'university',
                        'Cao đẳng': 'colleges',
                        'Trung cấp': 'vocational',
                        'THPT': 'high_school',
                        'Khác': 'other'}

        work_location = {'Tầng 6 tòa nhà AC': 'ac6',
                         'Tầng 8 tòa nhà AC': 'ac8',
                         'Tầng 9 tòa nhà HL': 'hl9',
                         'Tòa nhà 60 Bạch Mai': '60bm',
                         'Địa điểm khác': 'other'}

        for account in datas.get(f'account'):
            i += 1
            if not account or account is None:
                raise ValueError(f"Account must be required at line {i + 3}")
            values = {f'{key}': datas.get(f'{key}')[i] for key in self.EMPLOYEE_FIELDS}
            gender = values.get('gender', '')
            if gender.strip().lower() == "nữ":
                values.update({'gender': 'female'})
            elif gender.strip().lower() == "name":
                values.update({'gender': 'male'})
            else:
                values.update({'gender': 'other'})
            values.update({'certificate': certificate.get(values.get('certificate', 'Khác').capitalize(), 'other')})
            values.update({'work_location': work_location.get(
                values.get('work_location', 'Địa điểm khác').capitalize(), 'other')})
            for key in self.DATE_FIELDS:
                if datas.get(f'{key}', False)[i]:
                    try:
                        convert_date = datetime.strptime(datas.get(f'{key}')[i], "%d/%m/%Y").date()
                    except Exception as e:
                        raise ValueError(f"Convert date fail at line {i + 3}! \n{e}")
                    values.update({f'{key}':convert_date})
            # if not values.get('hire_date', False):
            #     values.update({'hire_date': values.get('start_work_date', False)})
            if values.get('spouse_complete_name', False):
                values.update({'marital': 'married'})
            if self.type == 'resignation':
                values.update({'resignation': True, 'active': False})

            try:
                employee_id = employee_dict.get(f'{account}', False)
                if employee_id:
                    employee_id.write(values)
                elif self.type == 'info':
                    raise ValidationError(f"Không tồn tại nhân viên có account: {account}. Xin hãy kiểm tra lại!")
                elif self.type == 'resignation':
                    values.update({'work_email': account})
                    self.env['hr.employee'].create(values)
            except Exception as e:
                wrong_line = f"Cannot upload value at lines: {i + 3}, account: {account}."
                raise ValidationError(_(f'{wrong_line}. Detail: \n {e}!'))
        return True

    @api.one
    def upload_contract_info(self):
        if self.type != 'contract':
            return False

        wb = open_workbook(file_contents=base64.decodestring(self.file_save))
        sheet_name = 'employee_contract'
        try:
            s = wb.sheet_by_name(sheet_name)
        except Exception as e:
            raise ValidationError(f"Không tồn tại sheet: {sheet_name}. Xin hãy kiểm tra lại!\n Detail: {e}")

        if not s.nrows:
            raise ValidationError(_('No data found!'))
        if not set(self.EMPLOYEE_CONTRACT_FIELDS).issubset(set(s.row_values(0))):
            raise ValidationError(_(f"Your file must contain at least these columns: "
                                    f"{', '.join(self.EMPLOYEE_CONTRACT_FIELDS)}!"))
        datas = {}
        for col, col_num in zip(s.row_values(0), range(0, s.ncols)):
            if col in self.EMPLOYEE_CONTRACT_FIELDS:
                col_value = []
                for row_num in range(2, s.nrows):
                    value = s.cell(row_num, col_num).value
                    try:
                        value = str(value)
                    except:
                        pass
                    col_value.append(value)
                    datas.update({col: col_value})
        self.employee_upload_contract_info(datas)

        return True

    def employee_upload_contract_info(self, datas):
        employee_ids = self.env['hr.employee'].search([('work_email', '!=', False),
                                                       '|', ('active', '=', False), ('active', '=', True)])
        employee_dict = {f'{employee_id.account}': employee_id for employee_id in employee_ids}
        i = -1

        for account in datas.get(f'account'):
            i += 1
            values = {}
            if not account or account is None:
                continue
            employee_id = employee_dict.get(f'{account}', False)
            if not employee_id:
                raise ValidationError(f"Không tồn tại nhân viên có account: {account}. Xin hãy kiểm tra lại!")

            name = datas.get('name')[i]
            try:
                date_start = datetime.strptime(datas.get('date_start')[i], "%d/%m/%Y").date()
            except Exception as e:
                raise ValidationError(f"Error at line {i + 3}. \n"
                                      f"Ngày hiệu lực hợp đồng phải là date string định dạng '%d/%m/%Y'")

            date_end = None
            type_id = False
            if name == 'HĐLĐ':
                try:
                    duration = int(datas.get('duration')[i])
                except Exception as e:
                    raise ValidationError(e)
                if duration == 0:
                    type_id = self.env.ref('hr_contract.hr_contract_type_emp')
                elif duration == 1:
                    type_id = self.env.ref('hr_core.hr_contract_type_1_year_emp')
                    date_end = date_start + relativedelta(years=1, days=-1)
                elif duration == 2:
                    type_id = self.env.ref('hr_core.hr_contract_type_2_year_emp')
                    date_end = date_start + relativedelta(years=2, days=-1)
                elif duration == 3:
                    type_id = self.env.ref('hr_core.hr_contract_type_3_year_emp')
                    date_end = date_start + relativedelta(years=3, days=-1)

            elif name == 'HĐTV':
                type_id = self.env.ref('hr_core.hr_contract_type_trial')
                date_end = date_start + relativedelta(days=59)

            elif name == 'HD CTV':
                type_id = self.env.ref('hr_core.hr_contract_type_collaborator')
            elif name == 'HD ĐT':
                type_id = self.env.ref('hr_core.hr_contract_type_training')
            elif name == 'HD DV':
                type_id = self.env.ref('hr_core.hr_contract_type_service')
            else:
                type_id = self.env.ref('hr_core.hr_contract_type_other')
            if not date_end and name != 'HĐLĐ':
                try:
                    date_end = datetime.strptime(datas.get('duration')[i], "%d/%m/%Y").date()
                except Exception as e:
                    raise ValidationError(f"Error at line {i + 3}. \n"
                                          f"Duration phải là date string định dạng '%d/%m/%Y'")

            try:
                self.env['hr.contract'].create({
                    'name': f"{type_id.name} for {employee_id.name} from {date.strftime(date_start, '%Y-%m-%d')}"
                            f"{' to ' + date.strftime(date_end, '%Y-%m-%d') if date_end else ''}",
                    'employee_id': employee_id.id,
                    'department_id': employee_id.department_id.id,
                    'job_id': employee_id.job_id.id,
                    'company_id': employee_id.company_id.id,
                    'resource_calendar_id': employee_id.resource_calendar_id.id,
                    'type_id': type_id.id,
                    'date_start': date_start,
                    'date_end': date_end,
                    'wage': 0
                })
            except Exception as e:
                wrong_line = f"Cannot create contract at lines: {i + 3}, account: {account}."
                raise ValidationError(_(f'{wrong_line}. Detail: \n {e}!'))
        return True
