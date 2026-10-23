[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$FilePath,
    [Parameter(Mandatory=$true)][string]$CaseId,
    [Parameter(Mandatory=$true)][int]$ProcessId,
    [string]$Receiver = 'http://192.168.56.1:8790/evidence'
)
$ErrorActionPreference = 'Stop'
$maxBytes = 50MB
$secret = [Environment]::GetEnvironmentVariable('SMARTSOC_EVIDENCE_SECRET', 'Process')
if ([string]::IsNullOrWhiteSpace($secret) -or $secret.Length -lt 32) { throw 'SMARTSOC_EVIDENCE_SECRET is not set or is too short.' }
if ($CaseId -notmatch '^[A-Za-z0-9_-]{1,64}$') { throw 'CaseId may contain only letters, digits, underscore, and hyphen (max 64).' }

$item = Get-Item -LiteralPath $FilePath -ErrorAction Stop
if ($item.PSIsContainer) { throw 'FilePath must identify one file.' }
$fullPath = [IO.Path]::GetFullPath($item.FullName)
$allowedRoots = @(
    [IO.Path]::GetFullPath((Join-Path $env:USERPROFILE 'Downloads')),
    [IO.Path]::GetFullPath($env:TEMP)
)
if (-not ($allowedRoots | Where-Object { $fullPath.StartsWith($_ + '\', [StringComparison]::OrdinalIgnoreCase) })) {
    throw 'For this lab script, the file must be under the current user Downloads or TEMP folder.'
}
if ($item.Length -le 0 -or $item.Length -gt $maxBytes) { throw 'File size is outside the allowed 1 byte to 50 MiB range.' }

# Stop only the explicitly supplied PID, and only if Windows reports that its executable image is this exact file.
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
if (-not $proc -or -not $proc.ExecutablePath) { throw 'PID is not running or its image path cannot be verified; no action taken.' }
$processImage = [IO.Path]::GetFullPath($proc.ExecutablePath)
if (-not $processImage.Equals($fullPath, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'PID image path does not match FilePath; no process was stopped and no file was removed.'
}
Stop-Process -Id $ProcessId -Force -ErrorAction Stop
$deadline = (Get-Date).AddSeconds(5)
do {
    Start-Sleep -Milliseconds 200
    $stillRunning = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
} while ($stillRunning -and (Get-Date) -lt $deadline)
if ($stillRunning) { throw 'Process termination was not verified; file was not uploaded or removed.' }

# Read and hash only after the exact matching process has stopped.
$bytes = [IO.File]::ReadAllBytes($fullPath)
$shaObj = [Security.Cryptography.SHA256]::Create()
try { $sha = ([BitConverter]::ToString($shaObj.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() } finally { $shaObj.Dispose() }
$name = [IO.Path]::GetFileName($fullPath) -replace '[^A-Za-z0-9._-]', '_'
if ($name.Length -gt 180) { $name = $name.Substring($name.Length - 180) }
$timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds().ToString()
$nonce = [Guid]::NewGuid().ToString('N')

# The signature covers the metadata and complete file bytes; do not print the secret or signature.
$metadata = @($timestamp, $nonce, $CaseId, $name, $sha) -join "`n"
$prefix = [Text.Encoding]::UTF8.GetBytes($metadata + "`n")
$payload = New-Object byte[] ($prefix.Length + $bytes.Length)
[Array]::Copy($prefix, 0, $payload, 0, $prefix.Length)
[Array]::Copy($bytes, 0, $payload, $prefix.Length, $bytes.Length)
$hmac = [Security.Cryptography.HMACSHA256]::new([Text.Encoding]::UTF8.GetBytes($secret))
try { $signature = ([BitConverter]::ToString($hmac.ComputeHash($payload))).Replace('-', '').ToLowerInvariant() } finally { $hmac.Dispose() }
$headers = @{
    'X-SmartSOC-Case' = $CaseId
    'X-SmartSOC-Filename' = $name
    'X-SmartSOC-SHA256' = $sha
    'X-SmartSOC-Timestamp' = $timestamp
    'X-SmartSOC-Nonce' = $nonce
    'X-SmartSOC-Signature' = $signature
}
$response = Invoke-RestMethod -Method Post -Uri $Receiver -Headers $headers -ContentType 'application/octet-stream' -Body $bytes -TimeoutSec 30
if ($response.status -ne 'STAGED' -or $response.sha256 -ne $sha) { throw 'Host did not confirm the same SHA-256. Victim copy was retained.' }

# User requested removal from the victim. Remove only after verified host staging.
Remove-Item -LiteralPath $fullPath -Force -ErrorAction Stop
Write-Host "Process terminated and verified: PID $ProcessId"
Write-Host "Evidence staged on host: $($response.stored_name)"
Write-Host "SHA256: $sha"
Write-Host 'Victim file removed after host hash confirmation.'







