# Edit these lines for YOUR Pi, then run:
#   .\launch_remote.ps1
$PiHost = "pi@YOUR_PI_IP"                                             # username@IP of your Pi
$PiDir  = "/home/pi/combined_dashboard_pi"                            # path to this folder ON the Pi
$DashboardPort = 8000

# Name (or full path) of the venv to use, ON THE PI, inside $PiDir above --
# same one you set up and installed dependencies into yourself. Change this
# if you used a different name; launch_remote.ps1 passes it straight
# through to run_dashboard.sh on the Pi.
$VenvName = "myenv_combined_dashboard"
