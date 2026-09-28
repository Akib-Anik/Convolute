from __future__ import annotations

import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("customtkinter", "customtkinter"),
]

OPTIONAL = [
    ("sounddevice", "sounddevice", "playback and the level meters"),
    ("soundfile", "soundfile", "MP3 / FLAC / OGG / AIFF import and export"),
]


def _check_dependencies() -> list[str]:
    missing = []
    for module, package in REQUIRED:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def main() -> int:
    missing = _check_dependencies()
    if missing:
        print("Mini Audio Editor cannot start — missing required packages:\n")
        for package in missing:
            print(f"  - {package}")
        print("\nInstall them with:\n")
        print(f"    pip install {' '.join(missing)}\n")
        print("Or install everything at once:\n")
        print("    pip install -r requirements.txt\n")
        return 1

    try:
        import tkinter
    except ImportError:
        print("Mini Audio Editor needs Tkinter, which is missing from this "
              "Python installation.\n")
        print("  macOS    : brew install python-tk")
        print("  Ubuntu   : sudo apt install python3-tk")
        print("  Windows  : re-run the Python installer and enable 'tcl/tk'\n")
        return 1

    for module, package, what in OPTIONAL:
        try:
            __import__(module)
        except ImportError:
            print(f"Note: '{package}' is not installed, so {what} "
                  f"will be unavailable.  (pip install {package})")

    from gui import MiniAudioEditorApp

    app = MiniAudioEditorApp()
    if len(sys.argv) > 1:
        app.after(120, lambda: app.open_path(sys.argv[1]))
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
