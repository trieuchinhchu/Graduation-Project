# -*- coding: utf-8 -*-
{
    'name': "DA Client User Connect",
    'summary': "DA User Connect Summary",
    'description': "DA User Connect Description",
    'author': "DA",
    'version': '0.1',
    'installable': True,
    'application': True,
    'auto_install': False,
    'depends': ['base'],
    'data': [
        'security/da_access_security.xml',
        'security/ir.model.access.csv',
    ]
}