# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for T.A.C.O. Twosday: Python Edition"""
import sys
import os

block_cipher = None

a = Analysis(
    ['taco/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('taco/resources', 'taco/resources'),
    ],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.QtOpenGL',
        'PyQt6.QtOpenGLWidgets',
        'PyQt6.QtMultimedia',
        'OpenGL.GL',
        'OpenGL.GL.shaders',
        'OpenGL.GLU',
        'numpy',
        'pyrr',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TACO_Twosday',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='taco/resources/textures/AngryTaco.ico' if sys.platform == 'win32' else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TACO_Twosday',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='TACO_Twosday.app',
        icon=None,
        bundle_identifier='com.taco.twosday',
        info_plist={
            'CFBundleShortVersionString': '2.1.0',
            'NSHighResolutionCapable': True,
        },
    )
