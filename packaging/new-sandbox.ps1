param(
    [string]$Bundle = "$PSScriptRoot/../dist/XaiGate1",
    [string]$Output = "$PSScriptRoot/../.packaging/sandbox"
)
$ErrorActionPreference = 'Stop'
$bundlePath = (Resolve-Path -LiteralPath $Bundle).Path
if (-not (Test-Path -LiteralPath (Join-Path $bundlePath 'XaiGate1.exe'))) {
    throw 'Build XaiGate1.exe before generating the sandbox configuration.'
}
$outputPath = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$resultsPath = Join-Path $outputPath 'results'
New-Item -ItemType Directory -Force -Path $resultsPath | Out-Null
$bundleXml = [Security.SecurityElement]::Escape($bundlePath)
$scriptsXml = [Security.SecurityElement]::Escape($PSScriptRoot)
$resultsXml = [Security.SecurityElement]::Escape($resultsPath)
@"
<Configuration>
  <Networking>Disable</Networking>
  <MemoryInMB>8192</MemoryInMB>
  <MappedFolders>
    <MappedFolder><HostFolder>$bundleXml</HostFolder><SandboxFolder>C:\Gate1Bundle</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>$scriptsXml</HostFolder><SandboxFolder>C:\Gate1Scripts</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>$resultsXml</HostFolder><SandboxFolder>C:\Gate1Results</SandboxFolder><ReadOnly>false</ReadOnly></MappedFolder>
  </MappedFolders>
  <LogonCommand><Command>powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Gate1Scripts\validate-sandbox.ps1</Command></LogonCommand>
</Configuration>
"@ | Set-Content -LiteralPath (Join-Path $outputPath 'gate1.wsb') -Encoding UTF8
Write-Output "Open $(Join-Path $outputPath 'gate1.wsb') with Windows Sandbox. Results: $resultsPath"
