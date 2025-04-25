if(BUILD_LZ4)

	set(LZ4_LIBRARY_NAME			"lz4")
	set(LZ4_VERSION					1.8.1)
	set(LZ4_LIBRARY_FULLNAME		${LZ4_LIBRARY_NAME}-${LZ4_VERSION})

	set(LZ4_EXTERNALPROJECT_NAME	${LZ4_LIBRARY_NAME}-externalproject)
	set(LZ4_BUILD_DIR				${CMAKE_BINARY_DIR}/${LZ4_LIBRARY_FULLNAME})
	set(LZ4_INSTALL_DIR				${CMAKE_SOURCE_DIR}/${LZ4_LIBRARY_FULLNAME})
	SET(LZ4_INSTALL_INCLUDE_DIR		${LZ4_INSTALL_DIR}/include)
	SET(LZ4_INSTALL_LIB_DIR			${LZ4_INSTALL_DIR}/lib)

	set(LZ4_CONFIGFILEPACKAGE_CONFIGFILE	${LZ4_LIBRARY_NAME}-config.cmake)
	set(LZ4_CONFIGFILEPACKAGE_TARGETFILE	${LZ4_LIBRARY_NAME}-targets.cmake)

	file(MAKE_DIRECTORY	${LZ4_INSTALL_DIR})
	file(MAKE_DIRECTORY ${LZ4_INSTALL_INCLUDE_DIR})
	file(MAKE_DIRECTORY ${LZ4_INSTALL_LIB_DIR})

	if(UNIX)

		#######################################################################
		# Download and build the external dependency
		#######################################################################
		ExternalProject_Add(
			${LZ4_EXTERNALPROJECT_NAME}
			GIT_REPOSITORY "https://github.com/lz4/lz4/"
			GIT_TAG "v1.8.1.2"
			SOURCE_DIR ${LZ4_BUILD_DIR}
			BINARY_DIR ${LZ4_BUILD_DIR}
			UPDATE_COMMAND ""
			PATCH_COMMAND ""
			CONFIGURE_COMMAND ""
			BUILD_COMMAND make -j 12
			INSTALL_COMMAND ""
		)

		ExternalProject_Add_Step(
			${LZ4_EXTERNALPROJECT_NAME} postbuild_step

			COMMENT "Copying the generated files and creating symbolic links"

			COMMAND
				${CMAKE_COMMAND} -E copy
				${LZ4_BUILD_DIR}/lib/lz4.h
				${LZ4_BUILD_DIR}/lib/lz4hc.h
				${LZ4_INSTALL_INCLUDE_DIR}

			COMMAND
				${CMAKE_COMMAND} -E copy
				${LZ4_BUILD_DIR}/lib/liblz4.so
				${LZ4_BUILD_DIR}/lib/liblz4.so.1
				${LZ4_BUILD_DIR}/lib/liblz4.so.1.8.1
				${LZ4_INSTALL_LIB_DIR}

			COMMAND
				${CMAKE_COMMAND} -E create_symlink
				${LZ4_INSTALL_LIB_DIR}/liblz4.so.1.8.1
				${LZ4_INSTALL_LIB_DIR}/liblz4.so.1

			COMMAND
				${CMAKE_COMMAND} -E create_symlink
				${LZ4_INSTALL_LIB_DIR}/liblz4.so.1
				${LZ4_INSTALL_LIB_DIR}/liblz4.so

			DEPENDEES build
		)

		#######################################################################
		# Create an interface target defining the usage requirements of the
		# dependency
		#######################################################################
		add_library(${LZ4_LIBRARY_NAME} INTERFACE)

		target_include_directories(
			${LZ4_LIBRARY_NAME}
			INTERFACE
				${LZ4_INSTALL_DIR}/include
		)

		target_link_libraries(
			${LZ4_LIBRARY_NAME}
			INTERFACE
				${LZ4_INSTALL_DIR}/lib/liblz4.so
		)

	else(UNIX)

		message(FATAL_ERROR "System not supported")

	endif(UNIX)

	###########################################################################
	# Generate the config-files to be consumed by a call to find_package() from
	# an external project
	###########################################################################
	export(
		TARGETS
			${LZ4_LIBRARY_NAME}
		FILE
			${LZ4_CONFIGFILEPACKAGE_TARGETFILE}
	)

	set(CONF_PACKAGE_NAME	${LZ4_LIBRARY_NAME})
	set(CONF_INCLUDE_DIR	${LZ4_INSTALL_DIR}/include)
	set(CONF_BINARY_DIR		${LZ4_INSTALL_DIR}/bin)

	configure_file(
		__package__-config.cmake.in
		${LZ4_CONFIGFILEPACKAGE_CONFIGFILE}
		@ONLY
	)

	###########################################################################
	# Install the previously generated config-files into the config-file
	# package directory
	###########################################################################
	ExternalProject_Add_Step(
		${LZ4_EXTERNALPROJECT_NAME} export_step

		COMMENT "Installing ${LZ4_LIBRARY_NAME} config-file package"

		COMMAND
			${CMAKE_COMMAND} -E copy
			${LZ4_CONFIGFILEPACKAGE_TARGETFILE}
			${DEPENDENCIES_CONFIGFILEPACKAGES_DIR}

		COMMAND
			${CMAKE_COMMAND} -E copy
			${LZ4_CONFIGFILEPACKAGE_CONFIGFILE}
			${DEPENDENCIES_CONFIGFILEPACKAGES_DIR}

		DEPENDEES install
	)

endif(BUILD_LZ4)
