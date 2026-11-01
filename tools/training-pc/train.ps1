# Minesweeper DQN training, medium board, the week-long run. Runs in this window and logs to train.log next to
# this script. The "RL training" scheduled task starts it at logon, and the dashboard's Resume button starts it again.
# To stop: Pause on the dashboard (http://<this pc>:8765), Ctrl+C here, or create RL_test\models\<run>.stop.
# The latest state is saved first either way. While RL_test\models\PAUSED exists (Pause makes it), it does not start.
#
# Two phases, one after the other; each carries on from its newest checkpoint after a restart:
#   phase A (med-week-a, 2M clicks, no look-ahead):
#     stage 1, clicks 0 to 1M: -10 for a mine, +0.01 for a safe click
#     stage 2, clicks 1M to 2M: -100 for a mine (the wall: 100x anything a click can earn); a safe click earns
#              0.5 x (g_after^10 - g_before^10) + 0.5 x burst(cells it opened) - 0.001
#   phase B (med-week-b, the rest of the week): the stage-2 reward plus look-ahead 0.9, from phase A's network
# Both: coin flips cost nothing and are not learned from; evaluations judge every lost game; the learning rate
# spikes, decays to a floor and spikes again on a plateau; training sleeps while the GPU is above 90 C.
$Host.UI.RawUI.WindowTitle = 'Minesweeper DQN training: medium, the week-long run (GTX 1050)'
$work = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Join-Path $work 'RL_test'
$py   = Join-Path $work '.venv\Scripts\python.exe'
$log  = Join-Path $work 'train.log'
$paused = Join-Path $repo 'models\PAUSED'
Set-Location $repo
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONPATH = $repo

$common = '--curriculum 3 --n-step 1 --per-alpha 0 --no-bf16 --loss bce --reward-win 0 --reward-progress 0 ' +
          '--wall 100 --skip-coin-flips --augment --ema 0.999 --buffer-size 1000000 --lr-mode restart --lr-patience 4 ' +
          '--eval-every 250000 --eval-games 1000 --eval-breakdown --checkpoint-every 100000 --max-gpu-temp 90'
$phases = @(
    @{ exp = 'med-week-a'; init = '';
       opts = '--reward-mine 10 --reward-safe 0.01 --gamma 0 --stage2-from 1000000 ' +
              '--stage2 mine=100,safe=-0.001,late=0.5,late_power=10,burst=0.5,burst_rate=5 ' +
              '--learning-rate 1e-4 --lr-floor 1e-5 --lr-decay-clicks 200000 ' +
              '--end-e 0.005 --exploration-fraction 0.075 --total-timesteps 2000000' },
    @{ exp = 'med-week-b'; init = 'med-week-a';
       opts = '--reward-mine 100 --reward-safe -0.001 --reward-late 0.5 --late-power 10 --reward-burst 0.5 --burst-rate 5 ' +
              '--gamma 0.9 --learning-rate 3e-5 --lr-floor 3e-6 --lr-decay-clicks 500000 ' +
              '--start-e 0.005 --end-e 0.005 --total-timesteps 200000000' }
)

function Show-And-Log { process { $_; Add-Content -Path $log -Value $_ -Encoding utf8 } }
function Now { Get-Date -Format 'yyyy-MM-dd HH:mm:ss' }
function Newest($pattern) {
    Get-ChildItem (Join-Path $repo 'models') -Filter $pattern | Sort-Object LastWriteTime | Select-Object -Last 1
}
function Phase-Finished($exp) {
    $dir = Get-ChildItem (Join-Path $repo 'runs') -Directory -Filter "$exp-c*" | Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $dir -or -not (Test-Path (Join-Path $dir.FullName 'status.json'))) { return $false }
    $s = Get-Content (Join-Path $dir.FullName 'status.json') -Raw | ConvertFrom-Json
    return ($s.state -eq 'stopped' -and $s.stop_reason -eq 'finished')
}

if (Test-Path $paused) {
    "=== $(Now) paused from the dashboard, so not starting. Resume there, or delete RL_test\models\PAUSED." | Show-And-Log
    return
}

# keep the PC from sleeping while this window trains (Windows lifts it when the window closes)
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[void][Win32.Power]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$crashes = 0
while ($true) {
    $phase = $phases | Where-Object { -not (Phase-Finished $_.exp) } | Select-Object -First 1
    if (-not $phase) { "=== $(Now) every phase has finished." | Show-And-Log; break }
    $latest = Newest "$($phase.exp)-c*-latest.pt"
    if ($latest) {
        $start = "--resume models/$($latest.Name)"
    } else {
        $start = "--exp-name $($phase.exp)"
        if ($phase.init) { $start += " --init models/$((Newest "$($phase.init)-c*-latest.pt").Name)" }
    }
    $cmd = "`"$py`" -m rl.dqn $start $common $($phase.opts)"
    $began = Get-Date
    "=== started $(Now), phase $($phase.exp)" | Show-And-Log
    "=== $cmd" | Show-And-Log
    cmd /c "$cmd 2>&1" | Show-And-Log
    $code = $LASTEXITCODE
    "=== ended $(Now), exit code $code" | Show-And-Log
    if ($code -eq 0) {
        if ((-not (Test-Path $paused)) -and (Phase-Finished $phase.exp)) { $crashes = 0; continue }   # on to the next phase
        break                                                                                          # paused or stopped
    }
    if (Test-Path $paused) { break }
    if (((Get-Date) - $began).TotalHours -gt 1) { $crashes = 0 }
    $crashes++
    if ($crashes -ge 5) { "=== crashed $crashes times in a row; not restarting. The lines above say why." | Show-And-Log; break }
    "=== restarting from the last save in 2 minutes (crash $crashes in a row; gives up at 5)" | Show-And-Log
    Start-Sleep -Seconds 120
}
"=== You can close this window." | Show-And-Log
