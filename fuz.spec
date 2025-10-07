# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for fuz CLI application
To build: pyinstaller fuz.spec --log-level WARN
To build silently: pyinstaller fuz.spec --log-level ERROR
"""

import os
import warnings
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Suppress warnings during build
warnings.filterwarnings('ignore')

block_cipher = None

# Collect only essential Django files (exclude unused apps to reduce warnings)
django_datas = collect_data_files('django.core')
django_datas += collect_data_files('django.db')
django_datas += collect_data_files('django.contrib.contenttypes')

# Only include Django modules we actually use
django_hiddenimports = [
    'django.core.management',
    'django.core.management.commands',
    'django.db.backends.postgresql',
    'django.contrib.contenttypes',
    'django.contrib.postgres',
]

# Collect psycopg
psycopg_hiddenimports = collect_submodules('psycopg')
psycopg_datas = collect_data_files('psycopg')
psycopg_binary_datas = collect_data_files('psycopg_binary')

a = Analysis(
    ['src/fuz/cli.py'],
    pathex=[],
    binaries=[],
    datas=django_datas + psycopg_datas + psycopg_binary_datas,
    hiddenimports=[
        'fuz',
        'fuz.models',
        'fuz.settings',
        'click',
    ] + django_hiddenimports + psycopg_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude Django components we don't need
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'django.contrib.humanize',
        'django.contrib.redirects',
        'django.contrib.sitemaps',
        'django.contrib.syndication',
        'django.core.servers',
        'django.views',
        'django.template',
        'django.templatetags',
        'django.test',
        # Test and development tools
        'test',
        'tests',
        'unittest',
        'pytest',
        # Unnecessary standard library modules
        'tkinter',
        'turtle',
        'pydoc',
        'doctest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='fuz',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
