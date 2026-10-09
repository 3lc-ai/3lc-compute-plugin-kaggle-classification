# export.ps1 - stage and zip everything the laptop holds that the PC needs (2026-10-09 migration).
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\rishi\Desktop\3LC Hackathons\3lc-compute-plugin-kaggle-classification\tools\migrate\export.ps1"
#
# Writes <OutRoot>\staging\ (the files in their relative layout) and <OutRoot>\migration-2026-10-09.zip:
#   3lc-hub-11\                     the 1.1.0 demo environment, without any .venv, __pycache__, .kaggle, or *.log older than today
#   3lc-hub-rc9-clean\              the clean tester environment, same exclusions (its project root home\projects rides along)
#   datasets\intel-scene-kit-v1\    the staged kit (shards + tree)
#   3lc-projects\                   the 3LC project root C:\Users\rishi\AppData\Local\3LC\3LC\projects (intel-scene, intel-scene-demo, test1)
#   claude\projects\<repo id>\memory\   Claude Code's memory for this repo; claude\settings.json (model + theme only)
#   MANIFEST.json                   sha256 + size per file, the laptop's versions, the repo commit
# Never: .kaggle, access_token, kaggle.json, 3lc_api_key, .credentials.json, mapping.csv, solution*.csv, any file whose
# text matches the release audit's secret patterns (status.SECRET_PATTERNS). The scan fails the export if one slips in.
# The repo itself is not staged: clone it on the PC at the same path and check out the commit MANIFEST.json names.
#
# ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI).

param(
    [string]$OutRoot = "C:\Users\rishi\Desktop\3LC Hackathons\migration-2026-10-09",
    [string]$Workspace = "C:\Users\rishi\Desktop\3LC Hackathons",
    [string]$ProjectsRoot = "C:\Users\rishi\AppData\Local\3LC\3LC\projects",
    [string]$ClaudeHome = "C:\Users\rishi\.claude",
    [string]$RepoId = "C--Users-rishi-Desktop-3LC-Hackathons-3lc-compute-plugin-kaggle-classification",
    [switch]$NoZip
)

$ErrorActionPreference = 'Stop'
$repo = Join-Path $Workspace '3lc-compute-plugin-kaggle-classification'
$staging = Join-Path $OutRoot 'staging'
$zip = Join-Path $OutRoot 'migration-2026-10-09.zip'
$report = Join-Path $OutRoot 'export.log'
$today = (Get-Date).Date

function Say([string]$m) { Write-Host $m; Add-Content -LiteralPath $report -Value $m -Encoding ascii }

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
"export $(Get-Date -Format s)" | Set-Content -LiteralPath $report -Encoding ascii
if (Test-Path -LiteralPath $staging) { Say "Removing the previous staging folder $staging"; Remove-Item -LiteralPath $staging -Recurse -Force }
New-Item -ItemType Directory -Force -Path $staging | Out-Null

# 1. The sources, copied into their relative layout. robocopy exit codes below 8 are success.
$sources = @(
    @{ name = '3lc-hub-11';        src = (Join-Path $Workspace '3lc-hub-11');        dest = '3lc-hub-11' },
    @{ name = '3lc-hub-rc9-clean'; src = (Join-Path $Workspace '3lc-hub-rc9-clean'); dest = '3lc-hub-rc9-clean' },
    @{ name = 'kit';               src = (Join-Path $Workspace 'datasets\intel-scene-kit-v1'); dest = 'datasets\intel-scene-kit-v1' },
    @{ name = 'projects';          src = $ProjectsRoot;                               dest = '3lc-projects' },
    @{ name = 'claude-memory';     src = (Join-Path $ClaudeHome "projects\$RepoId\memory"); dest = "claude\projects\$RepoId\memory" }
)
$excludeDirs = @('.venv', '__pycache__', '.kaggle', 'node_modules')
$excludeFiles = @('access_token', 'kaggle.json', '3lc_api_key', '.credentials.json', '*.pyc', '*.pids', 'mapping.csv', 'solution*.csv')
foreach ($s in $sources) {
    if (-not (Test-Path -LiteralPath $s.src)) { Say "MISSING source $($s.src)"; exit 1 }
    $dest = Join-Path $staging $s.dest
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Say "Copying $($s.name): $($s.src) -> $dest"
    # A redirected home carries uv's and pip's caches (gigabytes of wheels and git checkouts): never staged.
    $cacheDirs = @((Join-Path $s.src 'home\AppData\Local\uv'), (Join-Path $s.src 'home\AppData\Local\pip'), (Join-Path $s.src 'home\.cache'))
    $rcArgs = @($s.src, $dest, '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:1', '/W:1', '/XD') + $excludeDirs + $cacheDirs + @('/XF') + $excludeFiles
    & robocopy @rcArgs | Out-Null
    if ($LASTEXITCODE -ge 8) { Say "robocopy failed for $($s.name) (exit $LASTEXITCODE)"; exit 1 }
}
$settings = Join-Path $ClaudeHome 'settings.json'
if (Test-Path -LiteralPath $settings) {
    New-Item -ItemType Directory -Force -Path (Join-Path $staging 'claude') | Out-Null
    Copy-Item -LiteralPath $settings -Destination (Join-Path $staging 'claude\settings.json')
    Say "Copied Claude Code settings.json (model + theme)"
}

# 2. Logs older than today go (any *.log anywhere in the two environments).
$old = Get-ChildItem -LiteralPath (Join-Path $staging '3lc-hub-11'), (Join-Path $staging '3lc-hub-rc9-clean') -Recurse -File -Filter '*.log' |
    Where-Object { $_.LastWriteTime -lt $today }
foreach ($f in $old) { Remove-Item -LiteralPath $f.FullName -Force }
Say "Removed $($old.Count) log files older than $($today.ToString('yyyy-MM-dd'))"

# 3. The secret scan: file names first, then the text of every text-like file (the release audit's patterns).
$nameRules = @('^access_token$', '^kaggle\.json$', '^3lc_api_key$', '^\.credentials\.json$', '(?i)^mapping\.csv$', '(?i)^solution[A-Za-z0-9_\-]*\.csv$')
$textRules = @(
    @{ label = 'kaggle access token';        re = 'KGAT_[A-Za-z0-9_\-]{8,}' },
    @{ label = 'kaggle.json credential';     re = '"key"\s*:\s*"[A-Za-z0-9]{16,}"' },
    @{ label = 'kaggle api key assignment';  re = '(?i)\bkaggle_key\b\s*[:=]\s*[''"]?[A-Za-z0-9]{16,}' },
    @{ label = 'bearer token';               re = '(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}' }
)
$textExt = @('.json', '.jsonl', '.md', '.txt', '.log', '.py', '.ps1', '.sh', '.yaml', '.yml', '.toml', '.csv', '.html', '.cfg', '.ini', '.env')
# The repo's own documented fake tokens (tests/test_kaggle_connect.py, tests/test_status.py, the clean-env proofs):
# a match that starts with one of these is a fixture, not a secret. Anything else fails the export.
$fakeTokens = @('KGAT_proofclean', 'KGAT_test0123456789', 'KGAT_other0123456789', 'KGAT_abcdefghijklmnop0123')
$files = Get-ChildItem -LiteralPath $staging -Recurse -File
$hits = @()
foreach ($f in $files) {
    foreach ($r in $nameRules) { if ($f.Name -match $r) { $hits += "$($f.FullName): file name matches $r" } }
    if ($textExt -contains $f.Extension.ToLower() -and $f.Length -lt 20MB) {
        $text = [IO.File]::ReadAllText($f.FullName)
        foreach ($t in $textRules) {
            foreach ($m in [regex]::Matches($text, $t.re)) {
                $v = $m.Value
                $isFake = $false
                foreach ($fk in $fakeTokens) { if ($v.StartsWith($fk)) { $isFake = $true } }
                if (-not $isFake) { $hits += "$($f.FullName): matches the $($t.label) pattern"; break }
            }
        }
    }
}
if ($hits.Count) {
    Say "SECRET SCAN FAILED ($($hits.Count) hits). Nothing was zipped; the staging folder is kept for inspection."
    $hits | ForEach-Object { Say "  $_" }
    exit 1
}
Say "Secret scan: $($files.Count) files, no hit"

# 4. MANIFEST.json: every file with sha256 and size, the laptop's versions, the repo commit.
function Versions([string]$env) {
    $py = Join-Path $env '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $py)) { return @{ python = 'no venv' } }
    $out = & $py -c "import sys, importlib.metadata as m; print(sys.version.split()[0]); print(m.version('3lc')); print(m.version('3lc-compute')); print(m.version('3lc-compute-plugin-sdk'))"
    $plugins = @()
    $mp = Join-Path $env 'home\.3lc-compute\managed-plugins\kaggle-classification'
    if (Test-Path -LiteralPath $mp) { $plugins = @(Get-ChildItem -LiteralPath $mp -Directory | ForEach-Object { $_.Name }) }
    return @{ python = $out[0]; '3lc' = $out[1]; '3lc-compute' = $out[2]; '3lc-compute-plugin-sdk' = $out[3]; 'kaggle-classification versions installed' = $plugins }
}
$commit = (& git -C $repo rev-parse HEAD).Trim()
$branch = (& git -C $repo rev-parse --abbrev-ref HEAD).Trim()
$dirty = @(& git -C $repo status --porcelain | Where-Object { $_ -notmatch 'docs/ui-review' })
$pluginVersion = (Select-String -LiteralPath (Join-Path $repo 'src\kaggle_classification\plugin.toml') -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
$entries = @()
$total = [long]0
foreach ($f in $files) {
    $rel = $f.FullName.Substring($staging.Length + 1).Replace('\', '/')
    $h = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLower()
    $entries += [ordered]@{ path = $rel; bytes = $f.Length; sha256 = $h }
    $total += $f.Length
}
$manifest = [ordered]@{
    created_at = (Get-Date).ToString('s')
    laptop = [ordered]@{ computer = $env:COMPUTERNAME; user = $env:USERNAME; workspace = $Workspace; projects_root = $ProjectsRoot; claude_home = $ClaudeHome }
    repo = [ordered]@{ path = $repo; branch = $branch; commit = $commit; plugin_version = $pluginVersion; uncommitted = $dirty }
    versions = [ordered]@{ '3lc-hub-11' = (Versions (Join-Path $Workspace '3lc-hub-11')); '3lc-hub-rc9-clean' = (Versions (Join-Path $Workspace '3lc-hub-rc9-clean')) }
    layout = [ordered]@{
        '3lc-hub-11' = (Join-Path $Workspace '3lc-hub-11'); '3lc-hub-rc9-clean' = (Join-Path $Workspace '3lc-hub-rc9-clean')
        'datasets/intel-scene-kit-v1' = (Join-Path $Workspace 'datasets\intel-scene-kit-v1'); '3lc-projects' = $ProjectsRoot
        "claude/projects/$RepoId/memory" = (Join-Path $ClaudeHome "projects\$RepoId\memory"); 'claude/settings.json' = $settings
    }
    excluded = [ordered]@{ dirs = $excludeDirs; files = $excludeFiles; logs_older_than = $today.ToString('yyyy-MM-dd') }
    file_count = $files.Count; total_bytes = $total
    files = $entries
}
$manifestPath = Join-Path $staging 'MANIFEST.json'
($manifest | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $manifestPath -Encoding utf8
Say "MANIFEST.json: $($files.Count) files, $([math]::Round($total / 1MB)) MB; repo $branch @ $commit (plugin $pluginVersion)$(if ($dirty.Count) { '; UNCOMMITTED changes: ' + ($dirty -join ', ') })"

# 5. The zip (Zip64-capable, so sizes above 2 GB are fine).
if ($NoZip) { Say "NoZip: staging only at $staging"; exit 0 }
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($staging, $zip, [IO.Compression.CompressionLevel]::Fastest, $false)
$size = (Get-Item -LiteralPath $zip).Length
Say "ZIP: $zip ($([math]::Round($size / 1MB)) MB)"
