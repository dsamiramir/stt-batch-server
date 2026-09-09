<#
.SYNOPSIS
    Registers "STT Batch Server" in Task Scheduler so it starts at logon.

.DESCRIPTION
    Runs as the current user with normal privileges - no admin rights, no stored
    password. The task therefore runs only while this user is logged on, which is
    also what the GPU wants: a real interactive session rather than session 0.

    Re-running this is safe; it replaces the existing registration.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_task.ps1
#>

$ErrorActionPreference = 'Stop'

$proj = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $proj 'scripts\run_server.ps1'
if (-not (Test-Path $runner)) { throw "run_server.ps1 을 찾을 수 없습니다: $runner" }

$taskName = 'STT Batch Server'
$psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

$action = New-ScheduledTaskAction -Execute $psExe `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`"" `
    -WorkingDirectory $proj

$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"

# ExecutionTimeLimit zero = no limit. The default is 3 days, which would kill a
# long-lived server; StartWhenAvailable catches a missed logon trigger.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$settings.Hidden = $true

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask -TaskName $taskName `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description '로컬 GPU STT 배치 서버. 로그온 시 자동 기동.' `
    -Force | Out-Null

Write-Output "등록 완료: '$taskName'"
Write-Output "  실행 대상 : $runner"
Write-Output "  계정      : $env:USERDOMAIN\$env:USERNAME (로그온 시)"
Write-Output ""
Write-Output "지금 바로 시작하려면:  Start-ScheduledTask -TaskName '$taskName'"
Write-Output "상태 확인:            Get-ScheduledTask -TaskName '$taskName' | Get-ScheduledTaskInfo"
Write-Output "해제:                 powershell -File scripts\uninstall_task.ps1"
