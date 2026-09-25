# Starts Reels Studio and opens it in your browser.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Starting Docker Desktop..."
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    do { Start-Sleep 3; docker info *> $null } until ($LASTEXITCODE -eq 0)
}

$files = @("-f", "docker-compose.yml")
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU found: transcription and encoding will use it."
} else {
    $files += @("-f", "docker-compose.cpu.yml")
    Write-Host "No NVIDIA GPU found: running on CPU."
}
docker compose @files up -d --build
if ($LASTEXITCODE -ne 0) { throw "docker compose failed" }

Write-Host "Waiting for Reels Studio..."
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing -TimeoutSec 2 | Out-Null; break }
    catch { Start-Sleep 2 }
}
Start-Process "http://localhost:8000"
Write-Host "Reels Studio is running at http://localhost:8000"
