# -*- coding: utf-8 -*-

import base64, logging

from xlrd import open_workbook
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from odoo import fields, models, api, _
from odoo.exceptions import ValidationError

_LOGGER = logging.getLogger(__name__)


class DAUpdateSkillPoint(models.TransientModel):
    _name = 'migrate.profile.employee'

    file_save = fields.Binary(string='Attachment File', attachment=True, help='Update your file.')
    file_name = fields.Char(stirng='File Name')

    FIELDS = ["ACCOUNT", "SYLL", "GKSK", "SHK", "CMT_HC", "GKS", "ANH_3_4", "VB"]

    def profile_update_ids(self, data):
        job_ids = self.env['hr.employee'].search([])
        job_dict = {f'{job_id.account}': job_id for job_id in job_ids}
        i = -1
        for name in data.get(f'ACCOUNT'):
            params_ids = []
            i += 1
            if not name or name is None:
                raise ValidationError(_(f"Account must be required at line {i + 3}"))
            values = {f'{key}': data.get(f'{key}')[i] for key in self.FIELDS}
            if values.get('SYLL', False) == 'x':
                params_ids.append(self.env.ref('hr_core.da_doc_syll').id)
            if values.get('GKSK', False) == 'x':
                params_ids.append(self.env.ref('hr_core.da_doc_gksk').id)
            if values.get('SHK', False) == 'x':
                params_ids.append(self.env.ref('hr_core.da_doc_shk').id)
            if values.get('CMT_HC', False) == 'x':
                params_ids.append(self.env.ref('hr_core.da_doc_cmt').id)
            if values.get('GKS', False) == 'x':
                params_ids.append(self.env.ref('hr_core.da_doc_gks').id)
            if values.get('ANH_3_4', False) == 'x':
                params_ids.append(self.env.ref('hr_core.da_doc_img').id)
            if values.get('VB', False) == 'x':
                params_ids.append(self.env.ref('hr_core.da_doc_other').id)
            try:
                job_id = job_dict.get(f'{name}', False)
                if job_id:
                    values.update({'document_ids': [(6, 0, params_ids)] or None})
                    job_id.write(values)
            except Exception as e:
                wrong_line = f"Cannot upload value at lines: {i + 3}, ID: {name}."
                raise ValidationError(_(f'{wrong_line}. Detail: \n {e}!'))
        return True

    @api.one
    def info_update(self):
        all_fields = self.FIELDS
        # Load data from xls file
        wb = open_workbook(file_contents=base64.decodestring(self.file_save))
        sheet_name = 'Sheet1'
        s = wb.sheet_by_name(sheet_name)
        if not s.nrows:
            raise ValidationError(_('No data found!'))
        if not set(all_fields).issubset(set(s.row_values(0))):
            raise ValidationError(_(f"Your file must contain at least these columns: "
                                    f"{', '.join(all_fields)}!"))
        data = {}
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
                    data.update({col: col_value})
        self.profile_update_ids(data)
