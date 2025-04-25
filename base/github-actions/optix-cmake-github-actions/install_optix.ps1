## -------------------
## Constants
## -------------------

# Dictionary of known OptiX versions and thier download URLS, which do not follow a consistent pattern :(
$OPTIX_ALLVERSIONS = @{
    "7.3.0" = "NVIDIA-OptiX-SDK-7.4.0-windows";
}

## -------------------
## Select OptiX version
## -------------------

$OPTIX_VERSION_MAJOR_MINOR = $env:optix

# Validate OptiX version, extracting components via regex
$optix_ver_matched = $OPTIX_VERSION_MAJOR_MINOR -match "^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)$"
if(-not $optix_ver_matched){
    Write-Output "Invalid OptiX version specified, <major>.<minor>.<patch> required. '$OPTIX_VERSION_MAJOR_MINOR'."
    exit 1
}
$OPTIX_MAJOR=$Matches.major
$OPTIX_MINOR=$Matches.minor
$OPTIX_PATCH=$Matches.patch

## ------------------------------------------------
## Select OptiX packages to install from environment
## ------------------------------------------------

$DIR=$OPTIX_ALLVERSIONS[$OPTIX_VERSION_MAJOR_MINOR]
$OptiX_INSTALL_DIR="$(Get-Location)\optix-cmake-github-actions\$DIR"

# Set environmental variables in this session
$env:OptiX_INSTALL_DIR = "$($OptiX_INSTALL_DIR)"
Write-Output "OptiX_INSTALL_DIR $($OptiX_INSTALL_DIR)"

# If executing on github actions, emit the appropriate echo statements to update environment variables
if (Test-Path "env:GITHUB_ACTIONS") {
    # Set paths for subsequent steps, using ${OptiX_INSTALL_DIR}
    echo "Adding OptiX to OptiX_INSTALL_DIR"
    echo "OptiX_INSTALL_DIR=$env:OptiX_INSTALL_DIR" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
