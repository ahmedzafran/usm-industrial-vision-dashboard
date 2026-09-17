<#
Run this ON YOUR PC, from PowerShell. Starts the combined dashboard on the
Pi over SSH and opens your browser to it. No key setup, no background jobs
-- every SSH/SCP call below just prompts you for the Pi's password
interactively (standard OpenSSH client behavior), which is all this needs:
the dashboard itself pushes the recording + CSVs straight to your browser
when you click Stop, so there's nothing left to poll or fetch separately.

FIRST TIME ON A NEW PI: edit pi_config.ps1 (same folder) with your Pi's
IP/host and the folder path on the Pi, then just run:
    .\launch_remote.ps1

One script on the Pi side (run_dashboard.sh) does everything: fixes the
Hailo NPU driver package if it's wrong for the chip, sets up the venv,
and launches -- no separate setup step. The ONE thing it can't do for you:
if the driver package had to change, it needs `sudo reboot` once (a real
prompt on the Pi -- unavoidable for a kernel driver reload). If `start`
doesn't come up, this script auto-prints the Pi's log, which will say so
plainly; then run `.\launch_remote.ps1 reboot` and `start` again.

Usage:
    .\launch_remote.ps1            # same as start
    .\launch_remote.ps1 start
    .\launch_remote.ps1 stop
    .\launch_remote.ps1 status
    .\launch_remote.ps1 reboot     # only needed if `start`'s log says so
#>

param(
    [ValidateSet("start", "stop", "status", "reboot")]
    [string]$Command = "start"
)

. (Join-Path $PSScriptRoot "pi_config.ps1")

$PiHostname = $PiHost -replace '^.*@', ''
$Url = "http://${PiHostname}:${DashboardPort}"

function Test-DashboardUp {
    try {
        $resp = Invoke-WebRequest -Uri $Url -TimeoutSec 3 -UseBasicParsing
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

switch ($Command) {
    "reboot" {
        Write-Host "[INFO] Rebooting $PiHost (you may be prompted for its password)..."
        ssh $PiHost "sudo reboot" 2>$null
        Write-Host "[INFO] Reboot triggered. Wait about a minute, then run:"
        Write-Host "       .\launch_remote.ps1"
    }
    "start" {
        if (Test-DashboardUp) {
            Write-Host "[INFO] Dashboard already running -- opening browser."
            Start-Process $Url
            return
        }

        Write-Host "[INFO] Connecting to $PiHost (you may be prompted for its password)..."
        # setsid + nohup detaches it from this SSH session so it keeps
        # running after the connection closes. VENV_NAME is passed through
        # so run_dashboard.sh activates the venv you configured in
        # pi_config.ps1 instead of always using the default name.
        ssh $PiHost "cd $PiDir && VENV_NAME='$VenvName' setsid nohup bash run_dashboard.sh > dashboard.log 2>&1 < /dev/null &"

        Write-Host "[INFO] Waiting for it to come up on $Url ..."
        Write-Host "       (first run installs dependencies -- can take a few minutes)"
        $up = $false
        for ($i = 0; $i -lt 180; $i++) {
            if (Test-DashboardUp) { $up = $true; break }
            Start-Sleep -Seconds 2
        }

        if ($up) {
            Write-Host "[INFO] Dashboard is up."
            Start-Process $Url
        } else {
            Write-Host "[WARN] Didn't come up yet. Last lines of the Pi's log:"
            Write-Host "----------------------------------------"
            ssh $PiHost "tail -n 40 $PiDir/dashboard.log" 2>$null
            Write-Host "----------------------------------------"
            Write-Host "If that shows a Hailo NPU error, SSH in and run 'bash run_dashboard.sh' directly once (it needs a real terminal to prompt for sudo, so it can't be fixed via a detached 'start')."
            Write-Host "Otherwise, full log:  ssh $PiHost `"tail -f $PiDir/dashboard.log`""
        }
    }
    "stop" {
        Write-Host "[INFO] Stopping dashboard on $PiHost (you may be prompted for its password)..."
        ssh $PiHost "pkill -f dashboard.py"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[INFO] Nothing was running."
        }
    }
    "status" {
        if (Test-DashboardUp) {
            Write-Host "Dashboard is running -- $Url"
        } else {
            Write-Host "Dashboard is NOT running."
        }
    }
}
