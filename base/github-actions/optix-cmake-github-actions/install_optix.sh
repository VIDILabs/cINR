# @todo - better / more robust parsing of inputs from env vars.
## -------------------
## Constants
## -------------------

declare -A OPTIX_ALLVERSIONS=( 
    ["7.3.0"]="NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64"
)

## -------------------
## Select OptiX version
## -------------------

# Get the OPTIX version from the environment as $optix.
OPTIX_VERSION_MAJOR_MINOR=${optix}

# Split the version.
# We (might/probably) don't know PATCH at this point - it depends which version gets installed.
OPTIX_MAJOR=$(echo "${OPTIX_VERSION_MAJOR_MINOR}" | cut -d. -f1)
OPTIX_MINOR=$(echo "${OPTIX_VERSION_MAJOR_MINOR}" | cut -d. -f2)
OPTIX_PATCH=$(echo "${OPTIX_VERSION_MAJOR_MINOR}" | cut -d. -f3)
# use lsb_release to find the OS.
UBUNTU_VERSION=$(lsb_release -sr)
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

echo "OPTIX_MAJOR: ${OPTIX_MAJOR}"
echo "OPTIX_MINOR: ${OPTIX_MINOR}"
echo "OPTIX_PATCH: ${OPTIX_PATCH}"
echo "UBUNTU_VERSION: ${UBUNTU_VERSION}"

# If we don't know the OPTIX_MAJOR or MINOR, error.
if [ -z "${OPTIX_MAJOR}" ] ; then
    echo "Error: Unknown OPTIX Major version. Aborting."
    exit 1
fi
if [ -z "${OPTIX_MINOR}" ] ; then
    echo "Error: Unknown OPTIX Minor version. Aborting."
    exit 1
fi
# If we don't know the Ubuntu version, error.
if [ -z ${UBUNTU_VERSION} ]; then
    echo "Error: Unknown Ubuntu version. Aborting."
    exit 1
fi

## -----------------
## Set environment vars / vars to be propagated
## -----------------
DIR="${OPTIX_ALLVERSIONS[${OPTIX_VERSION_MAJOR_MINOR}]}"

export OptiX_INSTALL_DIR="${PWD}/optix-cmake-github-actions/${DIR}"
echo export OptiX_INSTALL_DIR="${PWD}/optix-cmake-github-actions/${DIR}"

# If executed on github actions, make the appropriate echo statements to update the environment
if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${OptiX_INSTALL_DIR}
    echo "Adding OptiX to OptiX_INSTALL_DIR"
    echo "OptiX_INSTALL_DIR=${OptiX_INSTALL_DIR}" >> $GITHUB_ENV
fi
