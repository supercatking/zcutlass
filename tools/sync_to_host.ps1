param(
  [string]$Source = "\\wsl.localhost\Ubuntu\home\zyz\zcutlass",
  [string]$Destination = "C:\Users\Admin\Documents\Codex\zcutlass"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Source)) {
  throw "WSL source path not found: $Source"
}

$resolvedDestination = [System.IO.Path]::GetFullPath($Destination)
if ($resolvedDestination -notlike "C:\Users\Admin\Documents\Codex\zcutlass*") {
  throw "Refusing to mirror into unexpected destination: $resolvedDestination"
}

New-Item -ItemType Directory -Force -Path $resolvedDestination | Out-Null

robocopy $Source $resolvedDestination /MIR /XD ".git" "build" ".cache" ".vscode" /XF "*.o" "*.obj" "*.a" "*.lib" "*.so" "*.dll" "*.exe" "*.pdb" "*.sass" "*.cubin" "*.ptx" "*.json" "*.log"
$code = $LASTEXITCODE

# Robocopy uses 0-7 for success conditions.
if ($code -gt 7) {
  throw "robocopy failed with exit code $code"
}

Write-Host "Mirrored zcutlass source to $resolvedDestination"
exit 0

