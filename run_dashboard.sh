#!/usr/bin/env bash
# ONE script. Run this and that's it:
#   bash run_dashboard.sh
#
# Fixes the Hailo NPU driver package if it's wrong for your chip, sets up
# (or repairs) the venv, installs dependencies, self-tests the NPU, then
# launches the dashboard. Every step is idempotent -- safe to re-run any
# time, including after it exits early asking for a reboot.
#
# To run in the background and watch progress separately:
#   nohup bash run_dashboard.sh > dashboard.log 2>&1 &
#   tail -f dashboard.log
# (this is exactly what launch_remote.ps1 does for you over SSH)
#
# Override the venv name/path without editing this file:
#   VENV_NAME=/path/to/your/venv bash run_dashboard.sh
VENV_NAME="${VENV_NAME:-myenv_combined_dashboard}"
# ---------------------------------------------------------------------

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=========================================="
echo " Combined Dashboard -- Startup"
echo "=========================================="

# --- 1. Hailo NPU driver package must match the chip actually plugged in --
# Two apt metapackages exist (hailo-all for Hailo-8, hailo-h10-all for
# Hailo-10H) and they Conflict/Replace each other -- if the wrong one is
# installed, hailo_platform.VDevice() fails with HAILO_NOT_SUPPORTED no
# matter what the Python code does. Detected once here so nobody has to
# know or remember which one their hardware needs.
CHIP_LINE=$(lspci 2>/dev/null | grep -i "Hailo" || true)
if echo "$CHIP_LINE" | grep -qi "Hailo-10H"; then
    NEEDED_META="hailo-h10-all"
elif echo "$CHIP_LINE" | grep -qi "Hailo-8"; then
    NEEDED_META="hailo-all"
elif [ -z "$CHIP_LINE" ]; then
    echo "[ERROR] No Hailo device found on the PCIe/M.2 bus (lspci shows nothing)."
    echo "        Is the AI HAT / M.2 module properly seated? Aborting."
    exit 1
else
    echo "[WARN] Unrecognized Hailo device, defaulting to hailo-h10-all:"
    echo "       $CHIP_LINE"
    NEEDED_META="hailo-h10-all"
fi

if dpkg -s "$NEEDED_META" >/dev/null 2>&1; then
    echo "[INFO] Hailo driver package OK ($NEEDED_META) for: $CHIP_LINE"
else
    echo "[INFO] Detected: $CHIP_LINE"
    echo "[INFO] Wrong (or no) Hailo driver package installed -- installing $NEEDED_META."
    if ! sudo -n true 2>/dev/null && [ ! -t 0 ]; then
        echo "[ERROR] This needs sudo, but there's no terminal attached here to ask"
        echo "        for a password (e.g. running detached via launch_remote.ps1)."
        echo "        Run this script directly in a real terminal once --"
        echo "        SSH in and run 'bash run_dashboard.sh' yourself, or run it"
        echo "        locally on the machine -- so it can prompt you. After that,"
        echo "        this branch never runs again and launch_remote.ps1 works fine."
        exit 1
    fi
    echo "[INFO] This needs sudo -- enter your password if prompted."
    sudo apt-get update
    sudo apt-get install -y "$NEEDED_META"
    echo
    echo "=========================================="
    echo " REBOOT REQUIRED -- one time only"
    echo "=========================================="
    echo "The Hailo driver package just changed, so the PCIe kernel driver"
    echo "needs a reload that only a reboot gives it cleanly."
    echo "    sudo reboot"
    echo "Then run this exact same command again (bash run_dashboard.sh) to finish."
    exit 0
fi

# --- 2. venv: create fresh, or repair if it's missing --system-site-packages
# Required: hailo_platform isn't pip-installable at all (comes from the apt
# package above), and cv2/numpy are also pulled from the system install
# here. Checked (and self-repaired) on every run, not just first creation.
NEED_INSTALL=0
if [ ! -d "$VENV_NAME" ]; then
    echo "[INFO] Creating venv '$VENV_NAME' (--system-site-packages)..."
    python3 -m venv --system-site-packages "$VENV_NAME"
    NEED_INSTALL=1
elif ! grep -q "^include-system-site-packages = true" "$VENV_NAME/pyvenv.cfg" 2>/dev/null; then
    echo "[WARN] '$VENV_NAME' exists but wasn't created with --system-site-packages"
    echo "       (hailo_platform/cv2 would be invisible to it). Recreating it..."
    rm -rf "$VENV_NAME"
    python3 -m venv --system-site-packages "$VENV_NAME"
    NEED_INSTALL=1
fi

echo "[INFO] Activating $VENV_NAME..."
source "$VENV_NAME/bin/activate"

# Dependencies are only (re)installed when the venv was just created/repaired
# above, or when explicitly forced. Force a (re)install anytime with:
#   INSTALL_DEPS=1 bash run_dashboard.sh
if [ "$NEED_INSTALL" = "1" ] || [ "${INSTALL_DEPS:-0}" = "1" ]; then
    echo "[INFO] Installing/updating requirements.txt..."
    pip install -r requirements.txt
else
    echo "[INFO] Skipping dependency install (venv already set up) -- set INSTALL_DEPS=1 to force."
fi

# --- 3. Self-test: actually open the NPU, not just import the module ------
# Fails fast with a clear message here instead of a deep traceback from
# inside combined_live_pi.py after the dashboard already reported "running".
echo "[INFO] Checking Hailo NPU access..."
python3 - <<'PYEOF'
from hailo_platform import VDevice
v = VDevice()
v.release()
print("[INFO] Hailo NPU OK.")
PYEOF

# --- 4. Launch --------------------------------------------------------
echo
echo "=========================================="
echo "Dashboard starting on http://0.0.0.0:8000"
echo "Open that address from any browser on the network."
echo "Ctrl+C here stops the dashboard (and any"
echo "running session)."
echo "=========================================="
python dashboard.py
