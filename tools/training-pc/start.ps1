# Starts the Minesweeper training and its dashboard, each only if it is not running already.
# "Start training.bat" next to this file runs it with admin rights (the training task has them).
param([switch]$NoWait)
$work = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Join-Path $work 'RL_test'
$Host.UI.RawUI.WindowTitle = 'Start Minesweeper training'

function Trainer-Running { [bool](Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'rl\.dqn' }) }
function Dashboard-Up {
    try { Invoke-RestMethod -Uri 'http://localhost:8765/api/state' -TimeoutSec 5 | Out-Null; $true } catch { $false }
}

if (Dashboard-Up) {
    'Dashboard: already running'
} else {
    Start-ScheduledTask -TaskName 'RL dashboard'
    $up = $false
    foreach ($i in 1..20) { Start-Sleep -Seconds 1; if (Dashboard-Up) { $up = $true; break } }
    if ($up) { 'Dashboard: started' } else { 'Dashboard: started, but not answering yet (give it a minute)' }
}

if (Trainer-Running) {
    'Training:  already running'
} else {
    $paused = Join-Path $repo 'models\PAUSED'
    if (Test-Path $paused) { Remove-Item $paused; 'Training:  was paused from the dashboard; resuming' }
    Stop-ScheduledTask -TaskName 'RL training' -ErrorAction SilentlyContinue   # closes a window left open by a finished run
    Start-ScheduledTask -TaskName 'RL training'
    'Training:  started. Its window opens in a moment and carries on from the last save.'
}

''
'Dashboard on this PC:  http://localhost:8765'
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -match '^(Wi-Fi|Ethernet)|Tailscale' -and $_.InterfaceAlias -notmatch 'vEthernet' -and $_.IPAddress -notlike '169.254.*' } |
    ForEach-Object { "From other devices:   http://$($_.IPAddress):8765   ($($_.InterfaceAlias))" }
if (-not $NoWait) { ''; Read-Host 'Press Enter to close' | Out-Null }
