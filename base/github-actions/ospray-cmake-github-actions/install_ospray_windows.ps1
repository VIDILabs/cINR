## -------------------
## Constants
## -------------------

# Dictionary of known ospray versions and thier download URLS, which do not follow a consistent pattern :(
$OSPRAY_KNOWN_URLS = @{
    "2.5.0"  = "https://github.com/ospray/ospray/releases/download/v2.5.0/ospray-2.5.0.x86_64.windows.zip";
    "2.6.0"  = "https://github.com/ospray/ospray/releases/download/v2.6.0/ospray-2.6.0.x86_64.windows.zip";
    "2.12.0" = "https://github.com/ospray/ospray/releases/download/v2.12.0/ospray-2.12.0.x86_64.windows.zip"
}

$TBB_KNOWN_URLS = @{
    "2021.1.1" = "https://github.com/oneapi-src/oneTBB/releases/download/v2021.1.1/oneapi-tbb-2021.1.1-win.zip";
    "2021.2.0" = "https://github.com/oneapi-src/oneTBB/releases/download/v2021.2.0/oneapi-tbb-2021.2.0-win.zip";
    "2021.9.0" = "https://github.com/oneapi-src/oneTBB/releases/download/v2021.9.0/oneapi-tbb-2021.9.0-win.zip"
}

$TBB_VERSIONS = @{
    "2.5.0"  = "2021.1.1";
    "2.6.0"  = "2021.2.0"
    "2.12.0" = "2021.9.0"
}

## -------------------
## Select OSPRay version
## -------------------

# Get the ospray version from the environment as env:ospray.
$OSPRAY_VERSION_FULL = $env:ospray
# Make sure OSPRAY_VERSION_FULL is set and valid, otherwise error.

# Validate OSPRay version, extracting components via regex
$ospray_ver_matched = $OSPRAY_VERSION_FULL -match "^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)$"
if(-not $ospray_ver_matched){
    Write-Output "Invalid OSPRay version specified, <major>.<minor>.<patch> required. '$OSPRAY_VERSION_FULL'."
    exit 1
}
$OSPRAY_MAJOR=$Matches.major
$OSPRAY_MINOR=$Matches.minor
$OSPRAY_PATCH=$Matches.patch

$TBB_VERSION_FULL=$TBB_VERSIONS[$OSPRAY_VERSION_FULL]

## ------------------------------------------------
## Select OSPRay packages to install from environment
## ------------------------------------------------

$RELEASE_URL = $OSPRAY_KNOWN_URLS[$OSPRAY_VERSION_FULL]
$RELEASE_FILE = "ospray-$OSPRAY_VERSION_FULL.x86_64.windows"
echo "RELEASE_URL $RELEASE_URL"
echo "RELEASE_FILE $RELEASE_FILE"

$TBB_RELEASE_URL = $TBB_KNOWN_URLS[$TBB_VERSION_FULL]
$TBB_RELEASE_FILE = "oneapi-tbb-$TBB_VERSION_FULL"
echo "TBB_RELEASE_URL $TBB_RELEASE_URL"
echo "TBB_RELEASE_FILE $TBB_RELEASE_FILE"


## ------------
## Install OSPRay
## ------------

(New-Object System.Net.WebClient).DownloadFile($RELEASE_URL, "$RELEASE_FILE.zip")
Expand-Archive -Path "$RELEASE_FILE.zip" -Destination $(Get-Location)

(New-Object System.Net.WebClient).DownloadFile($TBB_RELEASE_URL, "$TBB_RELEASE_FILE.zip")
Expand-Archive -Path "$TBB_RELEASE_FILE.zip" -Destination $(Get-Location)

## ------------------------------------------------
## Select OSPRay packages to install from environment
## ------------------------------------------------

$OSPRAY_CMAKE_DIR="$(Get-Location)\$RELEASE_FILE\lib\cmake\ospray-$OSPRAY_VERSION_FULL"
$TBB_CMAKE_DIR="$(Get-Location)\$TBB_RELEASE_FILE\lib\cmake\tbb"

# Set environmental variables in this session
$env:OSPRAY_CMAKE_DIR = "$($OSPRAY_CMAKE_DIR)"
$env:TBB_CMAKE_DIR = "$($TBB_CMAKE_DIR)"
Write-Output "OSPRAY_CMAKE_DIR $($OSPRAY_CMAKE_DIR)"
Write-Output "TBB_CMAKE_DIR $($TBB_CMAKE_DIR)"

# If executing on github actions, emit the appropriate echo statements to update environment variables
if (Test-Path "env:GITHUB_ACTIONS") {
    # Set paths for subsequent steps, using ${OSPRAY_CMAKE_DIR} & ${TBB_CMAKE_DIR}
    echo "Adding OSPRay/TBB to OSPRAY_CMAKE_DIR and TBB_CMAKE_DIR"
    echo "OSPRAY_CMAKE_DIR=$env:OSPRAY_CMAKE_DIR" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    echo "TBB_CMAKE_DIR=$env:TBB_CMAKE_DIR" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
