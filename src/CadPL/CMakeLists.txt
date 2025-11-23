# project name
set(LIB_NAME CadPL)
set(LIB_NAME_UPPER)
string(TOUPPER ${LIB_NAME} LIB_NAME_UPPER)

# include CadR macros
include(${CMAKE_SOURCE_DIR}/CMakeModules/CADRMacros.cmake)

# linkage
if(CADR_DYNAMIC_LINKAGE)
	set(CADR_USER_DEFINED_DYNAMIC_OR_STATIC "SHARED")
	add_definitions(-D${LIB_NAME_UPPER}_LIBRARY)
else()
	set(CADR_USER_DEFINED_DYNAMIC_OR_STATIC "STATIC")
	add_definitions(-D${LIB_NAME_UPPER}_LIBRARY_STATIC)
endif()

# public headers
set(CADPL_PUBLIC_HEADERS
	PipelineSceneGraph.h
	PipelineLibrary.h
	ShaderGenerator.h
	ShaderLibrary.h
	ShaderState.h
	DebugUtils.h
	)

# private headers
set(CADPL_PRIVATE_HEADERS
	)

# sources
set(CADPL_SOURCES
	PipelineSceneGraph.cpp
	PipelineLibrary.cpp
	ShaderGenerator.cpp
	ShaderLibrary.cpp
	ShaderState.cpp
	DebugUtils.cpp
	)

set(CADPL_PREGEN_SOURCES
	Pregenerate.cpp
	ShaderGenerator.h
	ShaderGenerator.cpp
	DebugUtils.h
	DebugUtils.cpp
	ShaderState.h
	ShaderState.cpp
)

# grouping of source files
source_group("Code" FILES ${CADPL_PUBLIC_HEADERS} ${CADPL_PRIVATE_HEADERS} ${CADPL_SOURCES})

# dependencies
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)

# shaderc library
option(SHADERC_SKIP_INSTALL "Skip installation" ON)
option(SHADERC_SKIP_TESTS "Skip building tests" ON)
option(SHADERC_SKIP_EXAMPLES "Skip building examples" ON)
option(SHADERC_SKIP_EXECUTABLES  "Skip building executables." ON)
add_subdirectory(../../3rdParty/shaderc shaderc_build)

add_custom_command(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ShaderGeneratorHash.hpp
	COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/ShaderGenerator.cpp -DDST=${CMAKE_CURRENT_BINARY_DIR}/ShaderGeneratorHash.hpp -P ${CMAKE_CURRENT_SOURCE_DIR}/generator_hash.cmake
	DEPENDS ShaderGenerator.cpp
)
add_custom_target(generator_hash DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ShaderGeneratorHash.hpp)

add_executable(${LIB_NAME}_pregen ${CADPL_PREGEN_SOURCES})
set_property(TARGET ${LIB_NAME}_pregen PROPERTY CXX_STANDARD 20)
target_link_libraries(${LIB_NAME}_pregen Vulkan::Headers CadR shaderc_shared)
target_include_directories(${LIB_NAME}_pregen PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/PregeneratedShaders.hpp
	COMMAND ${LIB_NAME}_pregen ${CMAKE_CURRENT_BINARY_DIR}/shaders
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	DEPENDS ${CADPL_PREGEN_SOURCES}
)

# CADR library
add_library(${LIB_NAME}
	${CADR_USER_DEFINED_DYNAMIC_OR_STATIC}
	${CADPL_PUBLIC_HEADERS}
	${CADPL_PRIVATE_HEADERS}
	${CADPL_SOURCES}
	${CMAKE_CURRENT_BINARY_DIR}/shaders/PregeneratedShaders.hpp
	)
add_dependencies(${LIB_NAME} generator_hash ${LIB_NAME}_pregen)

# compile definitions
if(WIN32)
	if(CADR_DYNAMIC_LINKAGE)
		target_compile_definitions(${LIB_NAME}
			PRIVATE
				"CADPL_EXPORT=__declspec(dllexport)"
			INTERFACE
				"CADPL_EXPORT=__declspec(dllimport)"
		)
		target_compile_definitions(${LIB_NAME}_pregen
				PRIVATE
				"CADPL_EXPORT=__declspec(dllexport)"
				INTERFACE
				"CADPL_EXPORT=__declspec(dllimport)"
		)
	else()
		target_compile_definitions(${LIB_NAME}
			PUBlIC
				"CADPL_EXPORT="
		)
		target_compile_definitions(${LIB_NAME}_pregen
				PUBlIC
				"CADPL_EXPORT="
		)
	endif()
else()
	target_compile_definitions(${LIB_NAME}
		PUBLIC
			"CADPL_EXPORT=__attribute__((visibility(\"default\")))"
	)
	target_compile_definitions(${LIB_NAME}_pregen
			PUBLIC
			"CADPL_EXPORT=__attribute__((visibility(\"default\")))"
	)
endif()

# target includes
get_filename_component(parent_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set_target_properties(${LIB_NAME} PROPERTIES
	INCLUDE_DIRECTORIES "${parent_dir}"
	INTERFACE_INCLUDE_DIRECTORIES "${parent_dir}"
	)
target_include_directories(${LIB_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# target libraries
target_link_libraries(${LIB_NAME} Vulkan::Headers glm CadR shaderc_shared)

# other target settings
set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 20)

target_compile_definitions(${LIB_NAME} PUBLIC CADPL_USE_PREGEN)

# headers installation
install(FILES
	${CADPL_PUBLIC_HEADERS}
	DESTINATION include/${LIB_NAME}
	)

# libraries installation
install(TARGETS ${LIB_NAME}
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	)

# cmake configuration file
# FIXME: is this working or broken?
#configure_file(
#	CADRConfig.cmake.in
#	CADRConfig.cmake
#	@ONLY
#	)

# install cmake configuration file
# FIXME: is this working or broken?
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CADRConfig.cmake"
#        DESTINATION "${INSTALL_CMAKE_CONFIG_DIR}" # INSTALL_CMAKE_CONFIG_DIR is defined in master CMakeLists file
#        )
