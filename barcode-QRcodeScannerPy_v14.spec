# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from pyzbar import pyzbar   # dylibs not detected because they are loaded by ctypes


block_cipher = None


a = Analysis(['barcode-QRcodeScannerPy_v14.py'],
             pathex=['C:\\Users\\z3099851\\OneDrive - UNSW\\Code\\QR_image_renamer'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

a.binaries += TOC([ (Path(dep._name).name, dep._name, 'BINARY') for dep in pyzbar.EXTERNAL_DEPENDENCIES ])

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='barcode-QRcodeScannerPy_v14',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
