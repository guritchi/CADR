# find package nlohmann_json
# if not found, do not include this example in build
include(${CMAKE_SOURCE_DIR}/CMakeModules/CADRMacros.cmake)
find_package_with_message(nlohmann_json nlohmann_json::nlohmann_json)


# glTFreader project
set(APP_NAME gltfReader)
project(${APP_NAME})

set(APP_SOURCES
	main.cpp
	StbImageImplementation.cpp
	VulkanWindow.cpp
	)

set(APP_INCLUDES
	VulkanWindow.h
	)

# dependencies
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)
if(UNIX)
	set(libs Vulkan::Headers glm nlohmann_json::nlohmann_json stdc++fs)
elseif(WIN32)
	set(libs Vulkan::Headers glm nlohmann_json::nlohmann_json)
endif()

# GUI dependencies
include(VulkanWindow.cmake)
VulkanWindowConfigure(APP_SOURCES APP_INCLUDES libs defines includes)
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdParty/imgui)
list(APPEND APP_SOURCES
	${IMGUI_DIR}/imgui.cpp
	${IMGUI_DIR}/imgui_draw.cpp
	${IMGUI_DIR}/imgui_tables.cpp
	${IMGUI_DIR}/imgui_widgets.cpp
	${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
	${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
	)

if(${GUI_TYPE} STREQUAL "Win32")
	list(APPEND APP_SOURCES ${IMGUI_DIR}/backends/imgui_impl_win32.cpp)
elseif(${GUI_TYPE} STREQUAL "SDL3")
	list(APPEND APP_SOURCES ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp)
elseif(${GUI_TYPE} STREQUAL "SDL2")
	list(APPEND APP_SOURCES ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp)
elseif(${GUI_TYPE} STREQUAL "GLFW")
	list(APPEND APP_SOURCES ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp)
#elseif(${GUI_TYPE} STREQUAL "Qt6")
#elseif(${GUI_TYPE} STREQUAL "Qt5")
else()
	message(FATAL_ERROR "Unsupported imgui backend: ${GUI_TYPE}")
endif()


# executable
add_executable(${APP_NAME} ${APP_SOURCES} ${APP_INCLUDES})

# target
target_include_directories(${APP_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${includes} ${IMGUI_DIR})
target_compile_definitions(${APP_NAME} PRIVATE ${defines} IMGUI_IMPL_VULKAN_NO_PROTOTYPES)
if(${GUI_TYPE} STREQUAL "GLFW")
	target_compile_definitions(${APP_NAME} PRIVATE GLFW_INCLUDE_VULKAN)
endif()
if (${VULKAN_VALIDATION})
	target_compile_definitions(${APP_NAME} PRIVATE VULKAN_VALIDATION)
endif()
target_link_libraries(${APP_NAME} ${libs} CadR CadPL)
set_property(TARGET ${APP_NAME} PROPERTY FOLDER "${examples_folder_name}")
set_property(TARGET ${APP_NAME} PROPERTY CXX_STANDARD 20)

# copy DLLs (SDL3.dll, SDL2.dll, glfw3.dll, Qt stuff,...)
if(WIN32)
	if(${GUI_TYPE} STREQUAL "SDL3" AND SDL3_DLL)
		add_custom_command(TARGET ${APP_NAME}
			POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SDL3_DLL}" $<TARGET_FILE_DIR:${APP_NAME}>)
	elseif(${GUI_TYPE} STREQUAL "SDL2" AND SDL2_DLL)
		add_custom_command(TARGET ${APP_NAME}
			POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SDL2_DLL}" $<TARGET_FILE_DIR:${APP_NAME}>)
	elseif(${GUI_TYPE} STREQUAL "GLFW" AND glfw3_DLL)
		add_custom_command(TARGET ${APP_NAME}
			POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${glfw3_DLL}" $<TARGET_FILE_DIR:${APP_NAME}>)
	elseif(${GUI_TYPE} STREQUAL "Qt6")
		add_custom_command(TARGET ${APP_NAME}
			POST_BUILD COMMAND Qt6::windeployqt
					--no-translations  # skip Qt translations
					--no-opengl-sw  # skip software OpenGL
					--no-system-d3d-compiler  # skip D3D stuff
					--no-svg  # skip svg support
					--no-compiler-runtime # skip VC runtime DLLs
					\"$<TARGET_FILE:${APP_NAME}>\"
					COMMENT "Deploying Qt related dependencies...")
	elseif(${GUI_TYPE} STREQUAL "Qt5")
		add_custom_command(TARGET ${APP_NAME}
			POST_BUILD COMMAND "${QT5_WINDEPLOYQT_EXECUTABLE}"
					--no-translations  # skip Qt translations
					--no-widgets  # skip Qt widgets
					--no-opengl-sw  # skip software OpenGL
					--no-angle  # skip software OpenGL (ANGLE)
					--no-compiler-runtime # skip VC runtime DLLs
					\"$<TARGET_FILE:${APP_NAME}>\"
					COMMENT "Deploying Qt related dependencies...")
	endif()
endif()
