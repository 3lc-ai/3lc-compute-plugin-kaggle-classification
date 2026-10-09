# start_tester.ps1 - start (or stop) the two 3LC services for testing the Kaggle Classification plugin.
#
#   powershell -ExecutionPolicy Bypass -File .\start_tester.ps1            # start
#   powershell -ExecutionPolicy Bypass -File .\start_tester.ps1 -Stop      # stop everything it started
#
# Run it from the tester folder that holds the venv (TESTING.md step 2). Everything the services write
# lands under that folder: .\home (the compute service's settings, the plugin's venv and state, the
# downloaded kit, the prediction CSVs, the ledger) and .\logs (one log per service start). Your 3LC
# login is NOT moved: on Windows the 3LC key store lives under the real %LOCALAPPDATA%\3LC and is read
# from there regardless of the redirect below. A Kaggle token under your real profile is NOT seen by
# the services (the home is redirected); the plugin's Connect button (TESTING.md 7.5b) writes the token
# it is given to .\home\.kaggle\access_token, nowhere else.
#
# Parameters (all optional):
#   -ObjectPort 5015      the 3LC object service port (the Getting Started page's default)
#   -ComputePort 5020     the compute service port (the Getting Started page's default)
#   -ProjectRoot <dir>    keep the plugin's tables and runs in their own 3LC project root (TLC_PROJECT_ROOT_URL
#                         for both services) instead of your default one
#   -ManifestBase <url>   the competition manifest tier; default https://competitions.dev.3lc.ai (the test tier)
#   -TestCatalog <url>    the plugin catalog listing the release candidate (default: the 1.0.0rc11 test catalog)
#   -Stop                 stop the services, their windows and this folder's plugin workers, then exit
#
# ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI).

param(
    [int]$ObjectPort = 5015,
    [int]$ComputePort = 5020,
    [string]$ProjectRoot = "",
    [string]$ManifestBase = "https://competitions.dev.3lc.ai",
    [string]$TestCatalog = "https://raw.githubusercontent.com/3lc-ai/3lc-compute-plugin-kaggle-classification/release/1.0.0rc11/catalog-test.json",
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venv = Join-Path $root '.venv'
$stateHome = Join-Path $root 'home'
$logDir = Join-Path $root 'logs'
$pidFile = Join-Path $root 'start_tester.pids'
$log = Join-Path $root 'start_tester.last.log'
$defaultCatalog = 'https://3lc-public-examples-2-2.s3.amazonaws.com/hub/catalog.json'

function Say([string]$m) {
    Write-Host $m
    for ($i = 0; $i -lt 20; $i++) {
        try { Add-Content -LiteralPath $log -Value $m -Encoding ascii -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 100 }
    }
}

if ($Stop) {
    "stop_tester $(Get-Date -Format s)" | Set-Content -LiteralPath $log -Encoding ascii
    # 1. The windows first (their restart loop would otherwise revive the services).
    if (Test-Path -LiteralPath $pidFile) {
        foreach ($p in (Get-Content -LiteralPath $pidFile | Where-Object { $_ -match '^\d+$' })) {
            try { Stop-Process -Id ([int]$p) -Force -ErrorAction Stop; Say "Closed window pid $p" } catch { }
        }
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
    # 2. The listeners on our two ports.
    foreach ($c in @(Get-NetTCPConnection -LocalPort $ObjectPort, $ComputePort -State Listen -ErrorAction SilentlyContinue)) {
        try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop; Say "Stopped listener on :$($c.LocalPort) (pid $($c.OwningProcess))" } catch { }
    }
    # 3. THIS folder's service processes and plugin workers only (matched on the folder path).
    $mine = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.IndexOf($root, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and ($_.CommandLine -like '*tlc_plugin_sdk.worker*' -or $_.CommandLine -like '*3lc-compute*' -or $_.CommandLine -like '*3lc.exe*service*') }
    foreach ($p in $mine) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Say "Stopped $($p.Name) (pid $($p.ProcessId))" } catch { } }
    $left = @(Get-NetTCPConnection -LocalPort $ObjectPort, $ComputePort -State Listen -ErrorAction SilentlyContinue)
    if ($left.Count) { Say "Still listening: $(($left | ForEach-Object { ':' + $_.LocalPort }) -join ', ')"; exit 1 }
    Say "Stopped: ports :$ObjectPort and :$ComputePort are free."
    exit 0
}

"start_tester $(Get-Date -Format s)" | Set-Content -LiteralPath $log -Encoding ascii
foreach ($exe in '3lc.exe', '3lc-compute.exe') {
    if (-not (Test-Path -LiteralPath (Join-Path $venv "Scripts\$exe"))) { Say "Missing $venv\Scripts\$exe - create the venv first (TESTING.md step 2)."; exit 1 }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Say "uv is not on PATH - the compute service needs it to build the plugin's environment (TESTING.md step 1)."; exit 1 }
$busy = @(Get-NetTCPConnection -LocalPort $ObjectPort, $ComputePort -State Listen -ErrorAction SilentlyContinue)
if ($busy.Count) {
    Say ("Already listening: " + (($busy | ForEach-Object { ":$($_.LocalPort) (pid $($_.OwningProcess))" }) -join ', ') + ". Stop it (-Stop), or pass other ports (-ObjectPort / -ComputePort).")
    exit 1
}
New-Item -ItemType Directory -Force -Path $stateHome, (Join-Path $stateHome 'AppData\Local'), (Join-Path $stateHome 'AppData\Roaming'), $logDir | Out-Null
if ($ProjectRoot) { New-Item -ItemType Directory -Force -Path $ProjectRoot | Out-Null }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

# The service, run in a loop: output to the window AND the log; an exit is reported and restarted after 5 s.
function Service-Loop([string]$name, [string]$invoke) {
    $logFile = Join-Path $logDir "$name-$stamp.log"
    return @"
`$log = '$logFile'
`$n = 0
while (`$true) {
    `$n++
    `$line = "[`$(Get-Date -Format s)] start #`$n - $name"
    Write-Host `$line -ForegroundColor Cyan; Add-Content -LiteralPath `$log -Value `$line
    $invoke 2>&1 | ForEach-Object { `$t = "`$_"; Write-Host `$t; Add-Content -LiteralPath `$log -Value `$t }
    `$code = `$LASTEXITCODE
    `$line = "[`$(Get-Date -Format s)] $name EXITED with code `$code - restarting in 5 s (run start_tester.ps1 -Stop to stop)"
    Write-Host `$line -ForegroundColor Red; Add-Content -LiteralPath `$log -Value `$line
    Start-Sleep -Seconds 5
}
"@
}

$projectLine = if ($ProjectRoot) { "`$env:TLC_PROJECT_ROOT_URL = '$ProjectRoot'" } else { "" }
$uvCacheLine = if ($env:UV_CACHE_DIR) { "`$env:UV_CACHE_DIR = '$($env:UV_CACHE_DIR)'" } else { "" }
$envBlock = @"
`$env:USERPROFILE = '$stateHome'
`$env:LOCALAPPDATA = '$stateHome\AppData\Local'
`$env:APPDATA = '$stateHome\AppData\Roaming'
`$env:HOME = '$stateHome'
$projectLine
$uvCacheLine
Set-Location -LiteralPath '$root'
"@

$objectCmd = @"
`$Host.UI.RawUI.WindowTitle = '3LC tester: object :$ObjectPort'
$envBlock
Write-Host '3LC object service on :$ObjectPort (stop with start_tester.ps1 -Stop; log in logs\object-$stamp.log)'
$(Service-Loop 'object' "& '$venv\Scripts\3lc.exe' service --port $ObjectPort --no-tui")
"@

$computeCmd = @"
`$Host.UI.RawUI.WindowTitle = '3LC tester: compute :$ComputePort'
$envBlock
`$env:TLC_COMPUTE_PORT = '$ComputePort'
`$env:KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL = '$ManifestBase'
`$env:TLC_COMPUTE_PLUGIN_CATALOG_URLS = '$defaultCatalog,$TestCatalog'
Write-Host '3LC compute service on :$ComputePort, manifest from $ManifestBase (stop with start_tester.ps1 -Stop; log in logs\compute-$stamp.log)'
$(Service-Loop 'compute' "& '$venv\Scripts\3lc-compute.exe'")
"@

$shell = (Get-Process -Id $PID).Path
function Open-Window([string]$cmd) {
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
    (Start-Process -FilePath $shell -ArgumentList '-NoExit', '-NoProfile', '-EncodedCommand', $enc -WorkingDirectory $root -PassThru).Id
}
$pids = @()
$pids += Open-Window $objectCmd
Say "Opened window: 3LC tester: object :$ObjectPort (pid $($pids[-1]))"
$pids += Open-Window $computeCmd
Say "Opened window: 3LC tester: compute :$ComputePort (pid $($pids[-1]))"
$pids -join "`r`n" | Set-Content -LiteralPath $pidFile -Encoding ascii

# Ready = both ports answer an HTTP request (any status: both answer 403/404 without the Hub token).
Add-Type -AssemblyName System.Net.Http
$http = New-Object System.Net.Http.HttpClient
$http.Timeout = [TimeSpan]::FromSeconds(5)
$deadline = (Get-Date).AddSeconds(300)
$status = @{}
$ports = @{ object = $ObjectPort; compute = $ComputePort }
while ((Get-Date) -lt $deadline) {
    foreach ($name in 'object', 'compute') {
        if ($status[$name]) { continue }
        try {
            $r = $http.GetAsync("http://127.0.0.1:$($ports[$name])/").GetAwaiter().GetResult()
            $status[$name] = [int]$r.StatusCode
            Say ("  {0} :{1} answers (HTTP {2})" -f $name, $ports[$name], $status[$name])
        } catch { }
    }
    if ($status.object -and $status.compute) { break }
    Start-Sleep -Seconds 2
}
if (-not ($status.object -and $status.compute)) {
    Say 'Not ready after 300 s: check the two service windows (or logs\) for errors. A missing 3LC login is the usual cause: run .\.venv\Scripts\3lc.exe login once.'
    exit 1
}
Say "Ready: open https://hub.3lc.ai/gettingstarted/ and connect the object service on 127.0.0.1:$ObjectPort and the compute service on 127.0.0.1:$ComputePort"
Say "Plugin state and logs live under $root (home\, logs\). Stop with: start_tester.ps1 -Stop"
