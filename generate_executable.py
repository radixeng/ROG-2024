import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--onefile',
    '--windowed',
    '--hidden-import=ultralytics',
    '--hidden-import=pygame',
    '--collect-data=ultralytics',
    '--collect-data=pygame',
    '--collect-data=best.pt',
    '--name=run'
])