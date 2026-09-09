<#
.SYNOPSIS
    Stops and removes the "STT Batch Server" scheduled task.

.DESCRIPTION
    Only touches the task registration. The project, its data directory, and the
    job database are left alone.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\uninstall_task.ps1
#>

$ErrorActionPreference = 'Stop'
$taskName = 'STT Batch Server'

$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Output "'$taskName' 작업이 등록돼 있지 않습니다. 할 일이 없습니다."
    exit 0
}

try { Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop } catch { }
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false

Write-Output "'$taskName' 작업을 제거했습니다."
Write-Output "이미 떠 있는 서버 프로세스는 그대로입니다. 필요하면 직접 종료하세요:"
Write-Output "  Get-NetTCPConnection -LocalPort 8760 -State Listen | ForEach-Object { Stop-Process -Id `$_.OwningProcess -Force }"
