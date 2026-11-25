# Installs the newest RustingStrata release for Windows x86_64 into %LOCALAPPDATA%\Programs\RustingStrata, adds that
# folder to the user PATH and makes a Start menu shortcut that opens the chat page. No admin rights needed.
#   irm https://raw.githubusercontent.com/RustingStudio/RustingStrata/main/install.ps1 | iex
# Later updates: rustingstrata update
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # the progress bar makes Invoke-WebRequest many times slower on PowerShell 5
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$dir = Join-Path $env:LOCALAPPDATA 'Programs\RustingStrata'
$exe = Join-Path $dir 'rustingstrata.exe'
$url = 'https://github.com/RustingStudio/RustingStrata/releases/latest/download/rustingstrata-windows-x86_64.exe'
New-Item -ItemType Directory -Force $dir | Out-Null
Write-Host "downloading $url"
Invoke-WebRequest -UseBasicParsing $url -OutFile "$exe.new"
try { Move-Item -Force "$exe.new" $exe }
catch { throw "could not replace $exe (is rustingstrata running? close it and run this again)" }
Write-Host "installed $(& $exe --version) to $exe"

$path = [Environment]::GetEnvironmentVariable('Path', 'User')
if (($path -split ';') -notcontains $dir) {
    [Environment]::SetEnvironmentVariable('Path', (($path, $dir) | Where-Object { $_ }) -join ';', 'User')
    $env:Path += ";$dir"
    Write-Host "added $dir to your PATH (new terminals see it)"
}

$lnk = Join-Path ([Environment]::GetFolderPath('Programs')) 'RustingStrata.lnk'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = $exe
$s.WorkingDirectory = $dir
$s.Description = 'RustingStrata chat page'
$s.Save()
Write-Host "Start menu: RustingStrata (opens the chat page in the browser; it downloads models)"
Write-Host ''
Write-Host 'Start:  rustingstrata serve --gpu --open   (chat page in the browser)'
Write-Host '   or:  rustingstrata interface            (a few questions in the terminal)'
Write-Host 'Update: rustingstrata update'
