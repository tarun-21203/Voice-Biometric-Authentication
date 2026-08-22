packages = [
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("sklearn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
]

if __name__ == '__main__':
    for name, mod in packages:
        try:
            m = __import__(mod)
            ver = getattr(m, '__version__', '')
            print(f"{name}: OK {ver}")
        except Exception as e:
            print(f"{name}: IMPORT ERROR: {e}")
