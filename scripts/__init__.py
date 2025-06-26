# scripts/__init__.py
"""
GreenRec Admin Scripts
======================
Utility scriptek gyűjteménye a GreenRec rendszerhez.
"""

import os

# Közös környezeti változók
DATABASE_URL = os.environ.get('DATABASE_URL')
SECRET_KEY = os.environ.get('SECRET_KEY')

# Export beállítások
EXPORT_DIR = 'exports'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

__all__ = ['DATABASE_URL', 'EXPORT_DIR', 'TIMESTAMP_FORMAT']
