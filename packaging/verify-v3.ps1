param([string]$Bundle = $PSScriptRoot)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $Bundle).Path.TrimEnd('\')
$manifestPath = Join-Path $root 'release-manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.issue -ne 204 -or $manifest.executable -ne 'XaiDashboardV3.exe') {
    throw 'Expected a Dashboard v3 release manifest.'
}
$files = @(Get-ChildItem -LiteralPath $root -Force -Recurse -File |
    Where-Object { $_.FullName -ne $manifestPath })
if ($files.Count -ne $manifest.file_count) {
    throw "Unexpected file count: $($files.Count)"
}
$total = 0L
foreach ($file in $files) {
    $name = $file.FullName.Substring($root.Length + 1).Replace('\', '/')
    $entry = $manifest.files.PSObject.Properties[$name]
    if (!$entry -or $file.Length -ne $entry.Value.bytes -or
        (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ne $entry.Value.sha256) {
        throw "File missing from manifest or differs: $name"
    }
    $total += $file.Length
}
if ($total -ne $manifest.total_bytes) {
    throw "Unexpected total file bytes: $total"
}
[ordered]@{
    status = 'verified'
    source_commit = $manifest.source_commit
    manifest_sha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    files = $files.Count
    bytes = $total
} | ConvertTo-Json
