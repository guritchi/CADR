include(${CMAKE_SOURCE_DIR}/CMakeModules/CADRMacros.cmake)

set(APP_NAME imageUtil)
project(${APP_NAME})

set(APP_SOURCES
    main.cpp
)

add_executable(${APP_NAME} ${APP_SOURCES} ${APP_INCLUDES})

target_include_directories(${APP_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${includes})
target_compile_definitions(${APP_NAME} PRIVATE ${defines})

set_property(TARGET ${APP_NAME} PROPERTY FOLDER "${examples_folder_name}")
set_property(TARGET ${APP_NAME} PROPERTY CXX_STANDARD 20)