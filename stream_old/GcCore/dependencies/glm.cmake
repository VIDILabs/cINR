if(BUILD_GLM)

	set(GLM_LIBRARY_NAME			"glm")
	set(GLM_VERSION					0.9.9.8)
	set(GLM_URL_HASH                37e2a3d62ea3322e43593c34bae29f57e3e251ea89f4067506c94043769ade4c)
	set(GLM_LIBRARY_FULLNAME		${GLM_LIBRARY_NAME}-${GLM_VERSION})

	set(GLM_EXTERNALPROJECT_NAME	${GLM_LIBRARY_NAME}-externalproject)
	set(GLM_BUILD_DIR				${CMAKE_BINARY_DIR}/${GLM_LIBRARY_FULLNAME})
	set(GLM_INSTALL_DIR				${CMAKE_SOURCE_DIR}/${GLM_LIBRARY_NAME})

	if(UNIX)

		#######################################################################
		# Download and build the external dependency
		#######################################################################

		ExternalProject_Add(${COMPONENT_NAME}
			PREFIX ${GLM_LIBRARY_FULLNAME}
			DOWNLOAD_DIR ${GLM_BUILD_DIR}
			STAMP_DIR    ${GLM_BUILD_DIR}/stamp
			SOURCE_DIR   ${GLM_BUILD_DIR}/src
			BINARY_DIR   ${GLM_BUILD_DIR}
			URL "https://github.com/g-truc/glm/releases/download/${GLM_VERSION}/glm-${GLM_VERSION}.zip"
			URL_HASH "SHA256=${GLM_URL_HASH}"
			CONFIGURE_COMMAND ""
			BUILD_COMMAND ""
			INSTALL_COMMAND "${CMAKE_COMMAND}" -E copy_directory
				<SOURCE_DIR>/glm
				${GLM_INSTALL_DIR}/glm
			BUILD_ALWAYS OFF
		)

	else(UNIX)

		message(FATAL_ERROR "System not supported")

	endif(UNIX)

endif(BUILD_GLM)
