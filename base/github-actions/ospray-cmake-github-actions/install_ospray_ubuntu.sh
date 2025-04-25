# @todo - better / more robust parsing of inputs from env vars.
## -------------------
## Constants
## -------------------

## -------------------
## Bash functions
## -------------------
# returns 0 (true) if a >= b
function version_ge() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$2" ]
}
# returns 0 (true) if a > b
function version_gt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_ge $1 $2
}
# returns 0 (true) if a <= b
function version_le() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$1" ]
}
# returns 0 (true) if a < b
function version_lt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_le $1 $2
}

## -------------------
## Select OSPRay version
## -------------------

# Get the ospray version from the environment as $ospray.
OSPRAY_VERSION_MAJOR_MINOR=${ospray}

# Split the version.
# We (might/probably) don't know PATCH at this point - it depends which version gets installed.
OSPRAY_MAJOR=$(echo "${OSPRAY_VERSION_MAJOR_MINOR}" | cut -d. -f1)
OSPRAY_MINOR=$(echo "${OSPRAY_VERSION_MAJOR_MINOR}" | cut -d. -f2)
OSPRAY_PATCH=$(echo "${OSPRAY_VERSION_MAJOR_MINOR}" | cut -d. -f3)
# use lsb_release to find the OS.
UBUNTU_VERSION=$(lsb_release -sr)
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

echo "OSPRAY_MAJOR: ${OSPRAY_MAJOR}"
echo "OSPRAY_MINOR: ${OSPRAY_MINOR}"
echo "OSPRAY_PATCH: ${OSPRAY_PATCH}"
echo "UBUNTU_VERSION: ${UBUNTU_VERSION}"

# If we don't know the OSPRAY_MAJOR or MINOR, error.
if [ -z "${OSPRAY_MAJOR}" ] ; then
    echo "Error: Unknown OSPRay Major version. Aborting."
    exit 1
fi
if [ -z "${OSPRAY_MINOR}" ] ; then
    echo "Error: Unknown OSPRay Minor version. Aborting."
    exit 1
fi
# If we don't know the Ubuntu version, error.
if [ -z ${UBUNTU_VERSION} ]; then
    echo "Error: Unknown Ubuntu version. Aborting."
    exit 1
fi

## -----------------
## Prepare to install
## -----------------
RELEASE_URL="https://github.com/ospray/ospray/releases/download/v${OSPRAY_VERSION_MAJOR_MINOR}/ospray-${OSPRAY_VERSION_MAJOR_MINOR}.x86_64.linux.tar.gz"
RELEASE_FILE="${PWD}/ospray-${OSPRAY_VERSION_MAJOR_MINOR}.x86_64.linux"

echo "RELEASE_URL ${RELEASE_URL}"
echo "RELEASE_FILE ${RELEASE_FILE}"

## -----------------
## Download and install
## -----------------
wget ${RELEASE_URL}
tar -xzvf "${RELEASE_FILE}.tar.gz"

if [[ $? -ne 0 ]]; then
    echo "OSPRay Installation Error."
    exit 1
fi

## -----------------
## Set environment vars / vars to be propagated
## -----------------

export OSPRAY_CMAKE_DIR="${RELEASE_FILE}/lib/cmake/ospray-${OSPRAY_VERSION_MAJOR_MINOR}"
export LD_LIBRARY_PATH="${RELEASE_FILE}/lib:${LD_LIBRARY_PATH}"

# If executed on github actions, make the appropriate echo statements to update the environment
if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${OSPRAY_CMAKE_DIR}
    echo "Adding OSPRay to OSPRAY_CMAKE_DIR and LD_LIBRARY_PATH"
    echo "OSPRAY_CMAKE_DIR=${OSPRAY_CMAKE_DIR}" >> $GITHUB_ENV
    echo "LD_LIBRARY_PATH=${RELEASE_FILE}/lib:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
fi
