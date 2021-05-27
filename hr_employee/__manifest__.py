# -*- coding: utf-8 -*-
{
    'name': 'HR Employee',
    'version': '0.1',
    'summary': 'This module add employee form info',
    'description': 'This module allow you can overrides odoo hr core.',
    'category': 'Human Resources',
    'author': 'DA (chinh.chutrieu)',
    'depends': ['base', 'website', 'mail', 'contacts', 'hr', 'hr_core',
                'hr_attendance', 'swr_datepicker', 'da_client_user'],
    'data': [
        'data/website_data.xml',
        'data/mail_template.xml',
        'security/hr_employee_new_security.xml',
        'security/ir.model.access.csv',
        'views/assets.xml',
        'views/hr_employee_new_views.xml',
        'views/hr_employee_views.xml',
        'views/res_company.xml',
        'views/menu_view.xml',
        'views/employee_skill.xml'
    ],
    'license': 'AGPL-3',
    'installable': True,
    'auto_install': False,
    'application': True
}
