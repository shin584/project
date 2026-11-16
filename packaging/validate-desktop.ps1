param(
    [Parameter(Mandatory=$true)][string]$Bundle,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
$bundlePath = (Resolve-Path -LiteralPath $Bundle).Path
$outputPath = [System.IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputPath) {
    throw 'Output must be a new directory, to avoid mixing validation runs.'
}
$manifest = Get-Content -LiteralPath (Join-Path $bundlePath 'release-manifest.json') -Raw | ConvertFrom-Json
# Validate the complete copied folder, including weights, tokenizer and WebView2.
foreach ($entry in $manifest.files.PSObject.Properties) {
    $candidatePath = [System.IO.Path]::GetFullPath((Join-Path $bundlePath $entry.Name))
    if (-not $candidatePath.StartsWith($bundlePath + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Invalid manifest path: $($entry.Name)"
    }
    $digest = (Get-FileHash -LiteralPath $candidatePath -Algorithm SHA256).Hash
    if ($digest -ne $entry.Value) { throw "Copied file differs: $($entry.Name)" }
}
New-Item -ItemType Directory -Path $outputPath | Out-Null
$environment = @{
    computer_name = $env:COMPUTERNAME
    build_host = $manifest.build_host
    different_pc = ($env:COMPUTERNAME -ne $manifest.build_host)
    os = [Environment]::OSVersion.VersionString
    recorded_at = [DateTime]::UtcNow.ToString('o')
    bundle = $bundlePath
    all_manifest_hashes_match = $true
}
$environment | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'environment.json') -Encoding UTF8
# The executable requires an empty output folder of its own.
$evidencePath = Join-Path $outputPath 'desktop'
$exePath = Join-Path $bundlePath 'XaiDashboard.exe'
$process = Start-Process -FilePath $exePath -ArgumentList @('--validate', ('"' + $evidencePath + '"')) -WorkingDirectory $outputPath -WindowStyle Hidden -PassThru
if (-not $process.WaitForExit(1200000)) {
    Stop-Process -Id $process.Id
    throw 'Desktop validation exceeded 20 minutes.'
}
if ($process.ExitCode -ne 0) { throw "Desktop validation failed: $($process.ExitCode); inspect $evidencePath" }
if (-not $environment.different_pc) {
    Write-Warning 'This is the build PC. A successful run does not pass the other-PC criterion.'
}
Write-Output "Evidence saved to $outputPath. Review the visual checklist before deciding gate 2."
