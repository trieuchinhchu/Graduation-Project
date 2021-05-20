# -*- coding: utf-8 -*-
{
    'name': 'DA Hr Core',
    'version': '0.1',
    'summary': 'This module overrides Odoo hr core',
    'description': 'This module allow you can overrides odoo hr core.',
    'category': 'Human Resources',
    'author': 'DA (chinh.chutrieu)',

    'depends': ['base', 'contacts', 'hr', 'hr_payroll', 'mail', 'hr_holidays', "da_client_user",
                'swr_datepicker'],
    'data': [
        'security/hr_core_security.xml',
        'security/ir.model.access.csv',
        'wizard/hr_department_change_report_view.xml',
        'wizard/hr_employee_update_view.xml',
        'wizard/employee_create_user_wizard_view.xml',
        'data/employee_sequence.xml',
        'data/hr_employee_job_default.xml',
        'data/hr_contract_data.xml',
        'data/da_location_data.xml',
        'views/hr_employee_view.xml',
        'views/hr_dependent_people_view.xml',
        'views/hr_contract_view.xml',
        'views/res_company.xml',
        'views/da_location_view.xml',
        'views/res_distrct.xml',
        'views/res_school.xml',
        'views/employee_private.xml',
        'wizard/migrate_data.xml'
    ],
    'license': 'AGPL-3',
    'installable': True,
    'auto_install': False,
    'application': True
}
