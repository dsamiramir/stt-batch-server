<#
.SYNOPSIS
    Launches the STT batch server. Task Scheduler invokes this at logon.

.DESCRIPTION
    Resolves the project root from this script's location, so the task keeps
    working if the folder moves. Exits quietly when the port is already in use,
    which stops Task Scheduler's restart-on-failure from fighting a server that
    someone started by hand.

    Logging is handled inside app/serve.py (UTF-8, size-rotated into
    data/server.log). Nothing is shell-redirected here on purpose: PowerShell
    5.1 writes redirects as UTF-16 and wraps native stderr in error records.
#>

$ErrorActionPreference = 'Stop'

$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj

$python = Join-Path $proj '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "가상환경이 없습니다: $python  (먼저 'uv sync' 를 실행하세요)"
}

# Port comes from .env when set, so this script and the server never disagree.
$port = 8760
$envFile = Join-Path $proj '.env'
if (Test-Path $envFile) {
    $hit = Select-String -Path $envFile -Pattern '^\s*STT_PORT\s*=\s*(\d+)' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($hit) { $port = [int]$hit.Matches[0].Groups[1].Value }
}

$busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Output "포트 $port 이 이미 사용 중입니다. 기동하지 않고 종료합니다."
    exit 0
}

$env:PYTHONIOENCODING = 'utf-8'
$ErrorActionPreference = 'Continue'
& $python -m app.serve
exit $LASTEXITCODE
