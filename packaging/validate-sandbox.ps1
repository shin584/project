# Run by gate1.wsb inside a disposable Windows Sandbox with Networking=Disable.
$ErrorActionPreference = 'Stop'
$resultPath = 'C:\Gate1Results\gate1-result.json'
$environment = [ordered]@{ clean_environment = $false; error = $null }
try {
    if ($env:USERNAME -ne 'WDAGUtilityAccount') { throw 'Expected Windows Sandbox account.' }
    $python = @(Get-Command python, python3, py, pip -ErrorAction SilentlyContinue)
    $cache = Join-Path $env:USERPROFILE '.cache\huggingface'
    $externalModels = Join-Path $env:USERPROFILE 'models'
    $environment.python_commands = @($python | ForEach-Object { $_.Source })
    $environment.hf_cache_exists = Test-Path -LiteralPath $cache
    $environment.external_models_exists = Test-Path -LiteralPath $externalModels
    $environment.active_network_adapters = @(Get-NetAdapter | Where-Object Status -eq 'Up').Count
    if ($python.Count -or $environment.hf_cache_exists -or $environment.external_models_exists) {
        throw 'Sandbox must have no Python, HF cache, or external models.'
    }
    if ($environment.active_network_adapters -ne 0) { throw 'Sandbox networking is still enabled.' }
    $environment.clean_environment = $true
    $copiedBundle = Join-Path $env:USERPROFILE 'Desktop\Gate1'
    if (Test-Path -LiteralPath $copiedBundle) { throw 'Use a fresh Sandbox for each attempt.' }
    Copy-Item -LiteralPath 'C:\Gate1Bundle' -Destination $copiedBundle -Recurse
    $run = Start-Process -FilePath (Join-Path $copiedBundle 'XaiGate1.exe') `
        -ArgumentList '--auto-close --output C:\Gate1Results\gate1-result.json' `
        -WorkingDirectory $env:USERPROFILE -WindowStyle Hidden -Wait -PassThru
    $report = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    $passed = $run.ExitCode -eq 0 -and $report.smoke_status -eq 'pass' -and `
        $report.chart_rendered -and $report.webview2_peak_rss_bytes -gt 0 -and `
        @($report.tcp_listeners).Count -eq 0 -and @($report.monitor_errors).Count -eq 0 -and `
        @($report.blocked_python_network_attempts).Count -eq 0
    $report.gate_status = if ($passed) { 'pass' } else { 'fail' }
    $report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resultPath -Encoding UTF8
} catch {
    $environment.error = $_.ToString()
    @{gate_status='fail'; error=$_.ToString()} | ConvertTo-Json |
        Set-Content -LiteralPath $resultPath -Encoding UTF8
} finally {
    $environment | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath 'C:\Gate1Results\environment.json' -Encoding UTF8
}
