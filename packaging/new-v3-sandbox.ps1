param(
    [Parameter(Mandatory = $true)][string]$Bundle,
    [Parameter(Mandatory = $true)][string]$Output
)

$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path -LiteralPath $Bundle).Path
if (!(Test-Path -LiteralPath (Join-Path $bundleRoot 'XaiDashboardV3.exe') -PathType Leaf)) {
    throw 'Expected a built Dashboard v3 folder.'
}
$outputRoot = [IO.Path]::GetFullPath($Output)
$null = New-Item -ItemType Directory -Path $outputRoot
$resultRoot = Join-Path $outputRoot 'results'
$null = New-Item -ItemType Directory -Path $resultRoot
$guest = @'
$ErrorActionPreference = 'Stop'
$results = 'C:\ReleaseResults'
try {
    $destination = Join-Path $env:LOCALAPPDATA 'XaiDashboardV3'
    $cwd = Join-Path $env:LOCALAPPDATA 'ReleaseValidationCwd'
    $null = New-Item -ItemType Directory -Path $cwd
    Copy-Item -LiteralPath 'C:\ReleaseInput' -Destination $destination -Recurse
    & (Join-Path $destination 'verify-v3.ps1') -Bundle $destination |
        Set-Content -LiteralPath "$results\verification.json" -Encoding UTF8
    $adapters = @(Get-NetAdapter | Select-Object Name,Status)
    $active = @($adapters | Where-Object Status -eq 'Up')
    $python = @(Get-Command python.exe,py.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Source)
    $paths = @('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE')
    foreach ($name in $paths) {
        [Environment]::SetEnvironmentVariable($name, (Join-Path $cwd 'empty-hf'), 'Process')
    }
    [ordered]@{
        recorded_at = [DateTime]::UtcNow.ToString('o')
        computer = $env:COMPUTERNAME
        os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber
        network_adapters = $adapters
        active_network_adapters = $active.Count
        python_commands = $python
        bundle = $destination
        working_directory = $cwd
        status = 'manual_workload_pending'
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$results\environment.json" -Encoding UTF8
    if ($active.Count) { throw 'The clean guest has an active network adapter.' }
    # This visible window is used to complete the guest checklist in the runbook.
    Start-Process -FilePath (Join-Path $destination 'XaiDashboardV3.exe') -WorkingDirectory $cwd
}
catch {
    $_ | Out-String | Set-Content -LiteralPath "$results\error.txt" -Encoding UTF8
    throw
}
'@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($guest))
$bundleXml = [Security.SecurityElement]::Escape($bundleRoot)
$resultXml = [Security.SecurityElement]::Escape($resultRoot)
$config = Join-Path $outputRoot 'dashboard-v3.wsb'
@"
<Configuration>
  <Networking>Disable</Networking>
  <ClipboardRedirection>Disable</ClipboardRedirection>
  <MemoryInMB>8192</MemoryInMB>
  <MappedFolders>
    <MappedFolder><HostFolder>$bundleXml</HostFolder><SandboxFolder>C:\ReleaseInput</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>$resultXml</HostFolder><SandboxFolder>C:\ReleaseResults</SandboxFolder><ReadOnly>false</ReadOnly></MappedFolder>
  </MappedFolders>
  <LogonCommand><Command>powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded</Command></LogonCommand>
</Configuration>
"@ | Set-Content -LiteralPath $config -Encoding UTF8
Write-Output $config
