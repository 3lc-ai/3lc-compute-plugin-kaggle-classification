# import.ps1 - on the PC: unpack the migration zip into the same relative layout, verify every file against
# MANIFEST.json, keep anything already there as *-old-<date>, rebuild the two environment venvs, run the
# intel-scene untouched check, then print the manual steps.
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\rishi\Desktop\3LC Hackathons\3lc-compute-plugin-kaggle-classification\tools\migrate\import.ps1" -Zip "<path>\migration-2026-10-09.zip"
#
# Parameters:
#   -Zip <path>            the zip export.ps1 wrote (required)
#   -TargetRoot <dir>      the workspace root; default C:\Users\<you>\Desktop\3LC Hackathons
#   -ProjectsRoot <dir>    the 3LC project root; default %LOCALAPPDATA%\3LC\3LC\projects
#   -ClaudeHome <dir>      Claude Code's home; default %USERPROFILE%\.claude
#   -SkipVenvs             unpack and verify only
#
# ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI).

param(
    [Parameter(Mandatory = $true)][string]$Zip,
    [string]$TargetRoot = (Join-Path $env:USERPROFILE 'Desktop\3LC Hackathons'),
    [string]$ProjectsRoot = (Join-Path $env:LOCALAPPDATA '3LC\3LC\projects'),
    [string]$ClaudeHome = (Join-Path $env:USERPROFILE '.claude'),
    [switch]$SkipVenvs
)

$ErrorActionPreference = 'Stop'
$date = Get-Date -Format 'yyyy-MM-dd'
$unpack = Join-Path $TargetRoot "migration-unpacked-$date"
$log = Join-Path $TargetRoot "migration-import-$date.log"
function Say([string]$m) { Write-Host $m; Add-Content -LiteralPath $log -Value $m -Encoding ascii }
New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
"import $(Get-Date -Format s) from $Zip" | Set-Content -LiteralPath $log -Encoding ascii

if (-not (Test-Path -LiteralPath $Zip)) { Say "Zip not found: $Zip"; exit 1 }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Say "uv is not on PATH; install it first (https://docs.astral.sh/uv/)"; exit 1 }

# 1. Unpack.
if (Test-Path -LiteralPath $unpack) { Say "Removing a previous unpack folder $unpack"; Remove-Item -LiteralPath $unpack -Recurse -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
Say "Unpacking $Zip -> $unpack"
[IO.Compression.ZipFile]::ExtractToDirectory($Zip, $unpack)

# 2. Verify every file against MANIFEST.json.
$manifestPath = Join-Path $unpack 'MANIFEST.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { Say "MANIFEST.json missing in the zip"; exit 1 }
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$bad = @()
$n = 0
foreach ($e in $manifest.files) {
    $p = Join-Path $unpack ($e.path.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $p)) { $bad += "missing: $($e.path)"; continue }
    $f = Get-Item -LiteralPath $p
    if ($f.Length -ne $e.bytes) { $bad += "size: $($e.path) ($($f.Length) vs $($e.bytes))"; continue }
    $h = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $e.sha256) { $bad += "sha256: $($e.path)" }
    $n++
}
if ($bad.Count) { Say "VERIFY FAILED ($($bad.Count)):"; $bad | ForEach-Object { Say "  $_" }; exit 1 }
Say "Verified $n files against MANIFEST.json (laptop export of $($manifest.created_at); repo $($manifest.repo.branch) @ $($manifest.repo.commit))"

# 3. Place each top-level entry; whatever is already there is renamed *-old-<date>, never merged.
function Place([string]$from, [string]$to) {
    if (-not (Test-Path -LiteralPath $from)) { Say "  (not in the zip: $from)"; return }
    if (Test-Path -LiteralPath $to) {
        $old = "$to-old-$date"
        $i = 1
        while (Test-Path -LiteralPath $old) { $i++; $old = "$to-old-$date-$i" }
        Rename-Item -LiteralPath $to -NewName (Split-Path -Leaf $old)
        Say "  existing $to -> $old"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $to) | Out-Null
    Move-Item -LiteralPath $from -Destination $to
    Say "  placed $to"
}
Say "Placing the environments and the kit under $TargetRoot"
Place (Join-Path $unpack '3lc-hub-11') (Join-Path $TargetRoot '3lc-hub-11')
Place (Join-Path $unpack '3lc-hub-rc9-clean') (Join-Path $TargetRoot '3lc-hub-rc9-clean')
Place (Join-Path $unpack 'datasets\intel-scene-kit-v1') (Join-Path $TargetRoot 'datasets\intel-scene-kit-v1')
Say "Placing the 3LC projects under $ProjectsRoot"
New-Item -ItemType Directory -Force -Path $ProjectsRoot | Out-Null
foreach ($d in Get-ChildItem -LiteralPath (Join-Path $unpack '3lc-projects') -Directory) { Place $d.FullName (Join-Path $ProjectsRoot $d.Name) }
$idx = Join-Path $unpack '3lc-projects\index.3lc.json'
if ((Test-Path -LiteralPath $idx) -and -not (Test-Path -LiteralPath (Join-Path $ProjectsRoot 'index.3lc.json'))) {
    Move-Item -LiteralPath $idx -Destination (Join-Path $ProjectsRoot 'index.3lc.json'); Say "  placed $ProjectsRoot\index.3lc.json"
} elseif (Test-Path -LiteralPath $idx) { Say "  kept the PC's own $ProjectsRoot\index.3lc.json (the object service re-indexes)" }
Say "Placing Claude Code's memory under $ClaudeHome"
$memSrc = Get-ChildItem -LiteralPath (Join-Path $unpack 'claude\projects') -Directory -ErrorAction SilentlyContinue
foreach ($d in $memSrc) { Place (Join-Path $d.FullName 'memory') (Join-Path $ClaudeHome "projects\$($d.Name)\memory") }
$settingsSrc = Join-Path $unpack 'claude\settings.json'
if (Test-Path -LiteralPath $settingsSrc) {
    $settingsDst = Join-Path $ClaudeHome 'settings.json'
    if (Test-Path -LiteralPath $settingsDst) { Copy-Item -LiteralPath $settingsSrc -Destination "$settingsDst.laptop"; Say "  the PC has its own settings.json; the laptop's is beside it as settings.json.laptop" }
    else { New-Item -ItemType Directory -Force -Path $ClaudeHome | Out-Null; Copy-Item -LiteralPath $settingsSrc -Destination $settingsDst; Say "  placed $settingsDst" }
}
Remove-Item -LiteralPath $unpack -Recurse -Force -ErrorAction SilentlyContinue

# 4. Rebuild both venvs (the recipe of 3lc-hub-11\SETUP.md and TESTING.md step 2, with the laptop's pins).
if (-not $SkipVenvs) {
    $v = $manifest.versions.'3lc-hub-11'
    $pinCompute = if ($v.'3lc-compute') { $v.'3lc-compute' } else { '1.1.0' }
    $pin3lc = if ($v.'3lc') { $v.'3lc' } else { '3.3.0' }
    & uv python install 3.12
    foreach ($envName in '3lc-hub-11', '3lc-hub-rc9-clean') {
        $envDir = Join-Path $TargetRoot $envName
        $venv = Join-Path $envDir '.venv'
        Say "Rebuilding $venv (3lc-compute==$pinCompute, 3lc==$pin3lc)"
        if (Test-Path -LiteralPath $venv) { Remove-Item -LiteralPath $venv -Recurse -Force }
        & uv venv --python 3.12 $venv
        & uv pip install --python (Join-Path $venv 'Scripts\python.exe') --index-url https://pypi.org/simple "3lc-compute==$pinCompute" "3lc==$pin3lc"
        if ($LASTEXITCODE) { Say "venv rebuild failed for $envName"; exit 1 }
    }
    # The managed plugin venvs were not exported; the compute service rebuilds them on the plugin's first route
    # (first-use provisioning); the plugin state under home\.3lc-compute\plugin-state came along intact.
}

# 5. The untouched check (hub-11's snapshots of intel-scene and intel-scene-demo).
$check = Join-Path $TargetRoot '3lc-hub-11\check_intel_scene_untouched.ps1'
if ((Test-Path -LiteralPath $check) -and -not $SkipVenvs) {
    if ($ProjectsRoot -ne 'C:\Users\rishi\AppData\Local\3LC\3LC\projects') { Say "NOTE: check_intel_scene_untouched.ps1 names C:\Users\rishi\AppData\Local\3LC\3LC\projects; your projects root is $ProjectsRoot" }
    Say "Running $check"
    & powershell -ExecutionPolicy Bypass -File $check
    Say "untouched check exit code $LASTEXITCODE (0 = both projects untouched)"
}

Say ""
Say "Manual steps:"
Say "  1. Clone the repo at the same path and check out the exported commit:"
Say "       git clone https://github.com/3lc-ai/3lc-compute-plugin-kaggle-classification.git `"$TargetRoot\3lc-compute-plugin-kaggle-classification`""
Say "       git -C `"$TargetRoot\3lc-compute-plugin-kaggle-classification`" checkout $($manifest.repo.branch) && git -C `"$TargetRoot\3lc-compute-plugin-kaggle-classification`" reset --hard $($manifest.repo.commit)"
Say "       cd into it, then: uv sync --python 3.12 --extra kaggle-classification --group dev; uv run pytest"
Say "  2. Claude Code: log in (claude login), open the repo folder; the memory you exported is under $ClaudeHome\projects\"
Say "  3. 3LC login for both venvs: `"$TargetRoot\3lc-hub-11\.venv\Scripts\3lc.exe`" login (the key store is per machine; the rc9-clean venv shares it on Windows)"
Say "  4. Kaggle: the token was not exported. Start the clean env, open Predict + Submit, paste a fresh KGAT_ token into Connect (it lands in home\.kaggle\access_token)."
Say "  5. Getting Started: https://hub.3lc.ai/gettingstarted/ -> object http://127.0.0.1:5015, compute http://127.0.0.1:5020 (the clean env), then hard refresh (Ctrl+F5) the plugin page."
Say "Start the clean env: powershell -ExecutionPolicy Bypass -File `"$TargetRoot\3lc-hub-rc9-clean\start_tester_rc12.ps1`" -ProjectRoot `"$TargetRoot\3lc-hub-rc9-clean\home\projects`""
