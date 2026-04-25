include(${CMAKE_SOURCE_DIR}/CMakeModules/CADRMacros.cmake)

set(APP_NAME shaderBinaryUtil)
project(${APP_NAME})

set(APP_SOURCES
    main.cpp
)

find_package(Vulkan REQUIRED)

#option(SHADERC_SKIP_INSTALL "Skip installation" ON)
#option(SHADERC_SKIP_TESTS "Skip building tests" ON)
#option(SHADERC_SKIP_EXAMPLES "Skip building examples" ON)
#option(SHADERC_SKIP_EXECUTABLES  "Skip building executables." ON)
#add_subdirectory(../../3rdParty/shaderc shaderc_build)

if(UNIX)
    set(libs Vulkan::Headers stdc++fs)
elseif(WIN32)
    set(libs Vulkan::Headers shaderc_shared) # Vulkan::shaderc_combined
endif()

add_executable(${APP_NAME} ${APP_SOURCES} ${APP_INCLUDES})

target_include_directories(${APP_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${includes})
target_compile_definitions(${APP_NAME} PRIVATE ${defines})
target_link_libraries(${APP_NAME} ${libs} CadR)
set_property(TARGET ${APP_NAME} PROPERTY FOLDER "${examples_folder_name}")
set_property(TARGET ${APP_NAME} PROPERTY CXX_STANDARD 20)