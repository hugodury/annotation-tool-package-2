"""Native folder/file dialogs — Linux, macOS, Windows."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

_user_cancelled = False


def _gui_available() -> bool:
    if platform.system() == "Windows":
        return True
    if platform.system() == "Darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _pick_folder_tk(initial_dir: str) -> str | None:
    if not _gui_available():
        return None
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        root.update()
        path = filedialog.askdirectory(
            initialdir=initial_dir,
            title="Choose storage folder",
            mustexist=True,
            parent=root,
        )
    finally:
        root.destroy()
    return path or None


def _pick_file_tk(initial_dir: str) -> str | None:
    if not _gui_available():
        return None
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        root.update()
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Choose JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=root,
        )
    finally:
        root.destroy()
    return path or None


def _pick_folder_linux(initial_dir: str) -> str | None:
    commands = [
        ["zenity", "--file-selection", "--directory", f"--filename={initial_dir}/"],
        ["kdialog", "--getexistingdirectory", initial_dir, "--title", "Choose storage folder"],
        ["yad", "--file", "--directory", f"--filename={initial_dir}"],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 1:
            return False
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def _pick_file_linux(initial_dir: str) -> str | None:
    commands = [
        [
            "zenity",
            "--file-selection",
            f"--filename={initial_dir}/",
            "--file-filter=JSON files | *.json",
        ],
        [
            "kdialog",
            "--getopenfilename",
            initial_dir,
            "JSON (*.json)|*.json",
            "--title",
            "Choose JSON file",
        ],
        [
            "yad",
            "--file",
            f"--filename={initial_dir}/",
            "--file-filter=*.json",
        ],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 1:
            return False
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def _pick_folder_macos(initial_dir: str) -> str | None:
    script = 'POSIX path of (choose folder with prompt "Choose storage folder")'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return False
    if result.stdout.strip():
        return result.stdout.strip()
    return None


def _pick_file_macos(initial_dir: str) -> str | None:
    script = (
        'POSIX path of (choose file of type {"json", "public.json"} '
        'with prompt "Choose JSON file")'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return False
    if result.stdout.strip():
        return result.stdout.strip()
    return None


def _powershell_bins() -> list[str]:
    bins = []
    for name in ("pwsh", "powershell"):
        if shutil_which(name):
            bins.append(name)
    return bins


def shutil_which(cmd: str) -> str | None:
    from shutil import which
    return which(cmd)


def _run_powershell(script: str) -> str | None:
    for shell in _powershell_bins():
        try:
            result = subprocess.run(
                [shell, "-NoProfile", "-STA", "-Command", script],
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def _ps_escape(path: str) -> str:
    return path.replace("'", "''")


def _pick_folder_windows(initial_dir: str) -> str | None:
    escaped = _ps_escape(initial_dir)
    script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = 'Choose storage folder'
if (Test-Path -LiteralPath '{escaped}') {{ $dlg.SelectedPath = '{escaped}' }}
if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    $dlg.SelectedPath
}}
""".strip()
    return _run_powershell(script)


def _pick_file_windows(initial_dir: str) -> str | None:
    escaped = _ps_escape(initial_dir)
    script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dlg = New-Object System.Windows.Forms.OpenFileDialog
$dlg.Title = 'Choose JSON file'
$dlg.Filter = 'JSON files (*.json)|*.json|All files (*.*)|*.*'
if (Test-Path -LiteralPath '{escaped}') {{ $dlg.InitialDirectory = '{escaped}' }}
if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    $dlg.FileName
}}
""".strip()
    return _run_powershell(script)


def _initial_dir(path: str | None) -> str:
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            return str(candidate.resolve())
        if candidate.parent.is_dir():
            return str(candidate.parent.resolve())
    return str(Path.home().resolve())


def _run_pickers(pickers: list, initial_dir: str) -> str | None:
    global _user_cancelled
    _user_cancelled = False
    last_error: str | None = None
    for picker in pickers:
        try:
            raw = picker(initial_dir)
        except Exception as exc:
            last_error = str(exc)
            continue
        if raw is False or raw == '':
            _user_cancelled = True
            continue
        if not raw:
            continue
        try:
            resolved = Path(raw).expanduser().resolve()
        except OSError:
            continue
        return str(resolved)
    if last_error:
        print(f"[native_dialogs] picker failed: {last_error}", file=sys.stderr)
    return None


def _linux_tool_available() -> list[str]:
    tools = []
    for cmd in ("zenity", "kdialog", "yad"):
        if shutil_which(cmd):
            tools.append(cmd)
    return tools


def dialog_capabilities() -> dict:
    system = platform.system()
    folder_pickers: list[str] = []
    file_pickers: list[str] = []
    if system == "Linux":
        folder_pickers.extend(_linux_tool_available())
        file_pickers.extend(_linux_tool_available())
        if _gui_available():
            folder_pickers.append("tkinter")
            file_pickers.append("tkinter")
    elif system == "Darwin":
        folder_pickers.extend(["osascript", "tkinter"])
        file_pickers.extend(["osascript", "tkinter"])
    elif system == "Windows":
        shells = _powershell_bins()
        folder_pickers.extend(shells or ["powershell"])
        file_pickers.extend(shells or ["powershell"])
        if _gui_available():
            folder_pickers.append("tkinter")
            file_pickers.append("tkinter")
    else:
        if _gui_available():
            folder_pickers.append("tkinter")
            file_pickers.append("tkinter")
    return {
        "platform": system,
        "gui_available": _gui_available(),
        "folder_pickers": folder_pickers,
        "file_pickers": file_pickers,
        "ready": bool(folder_pickers) and bool(file_pickers),
    }


def picker_unavailable_message() -> str | None:
    if _user_cancelled:
        return None
    caps = dialog_capabilities()
    if caps["ready"]:
        return None
    system = caps["platform"]
    if system == "Linux":
        return (
            "Impossible d'ouvrir le sélecteur système. Sur Linux : installez zenity, kdialog ou yad, "
            "ou vérifiez que DISPLAY/WAYLAND_DISPLAY est défini (session bureau)."
        )
    if system == "Windows":
        return (
            "Impossible d'ouvrir le sélecteur système. Sur Windows : installez PowerShell ou pwsh, "
            "ou Python avec tkinter."
        )
    if system == "Darwin":
        return (
            "Impossible d'ouvrir le sélecteur système. Sur macOS : osascript (Finder) ou tkinter requis."
        )
    return "Impossible d'ouvrir le sélecteur système sur cette machine."


def pick_folder(initial_dir: str | None = None) -> str | None:
    """Open the OS folder picker. Returns an absolute path or None if cancelled."""
    initial = _initial_dir(initial_dir)
    system = platform.system()
    if system == "Linux":
        pickers = [_pick_folder_linux, _pick_folder_tk]
    elif system == "Darwin":
        pickers = [_pick_folder_macos, _pick_folder_tk]
    elif system == "Windows":
        pickers = [_pick_folder_windows, _pick_folder_tk]
    else:
        pickers = [_pick_folder_tk]
    path = _run_pickers(pickers, initial)
    if path and Path(path).is_dir():
        return path
    return None


def pick_json_file(initial_dir: str | None = None) -> str | None:
    """Open the OS file picker for a JSON file. Returns an absolute path or None."""
    initial = _initial_dir(initial_dir)
    system = platform.system()
    if system == "Linux":
        pickers = [_pick_file_linux, _pick_file_tk]
    elif system == "Darwin":
        pickers = [_pick_file_macos, _pick_file_tk]
    elif system == "Windows":
        pickers = [_pick_file_windows, _pick_file_tk]
    else:
        pickers = [_pick_file_tk]
    path = _run_pickers(pickers, initial)
    if path and Path(path).is_file():
        return path
    return None
