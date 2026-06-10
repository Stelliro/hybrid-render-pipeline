cmake_minimum_required(VERSION 3.21)
project(hybrid-render-pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Recommended toolchain: -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
# All dependencies are listed in vcpkg.json (manifest mode).

# ── Dependencies ──────────────────────────────────────────────────────

find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Stb REQUIRED)

# ── Sources ───────────────────────────────────────────────────────────

set(HRP_SOURCES
    src/main.cpp
    src/Renderer.cpp
    src/ScreenSpaceEffects.cpp
    src/Camera.cpp
    src/Input.cpp
    src/Log.cpp
    src/Platform.cpp
    src/Types.cpp
)

set(HRP_HEADERS
    src/Renderer.hpp
    src/ScreenSpaceEffects.hpp
    src/Camera.hpp
    src/Input.hpp
    src/Log.hpp
    src/Platform.hpp
    src/Types.hpp
    src/Frustum.hpp
    src/ErrorCodes.hpp
)

add_executable(hrp_demo ${HRP_SOURCES} ${HRP_HEADERS})

target_include_directories(hrp_demo PRIVATE
    src
    ${Stb_INCLUDE_DIR}
)

target_link_libraries(hrp_demo PRIVATE
    Vulkan::Vulkan
    GPUOpen::VulkanMemoryAllocator
    glm::glm
    SDL3::SDL3
    spdlog::spdlog
)

if(MSVC)
    target_compile_options(hrp_demo PRIVATE /W4 /permissive- /Zc:__cplusplus)
    target_compile_definitions(hrp_demo PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(hrp_demo PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# ── Shader compilation ────────────────────────────────────────────────
# Compiles every shader under shaders/ to build/shaders/*.spv. Renderer.cpp
# searches "build/shaders/", "shaders/", and "../shaders/" at runtime.

find_program(GLSLC NAMES glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES bin)
if(NOT GLSLC)
    message(WARNING "glslc not found. Shaders will not be compiled automatically. "
                    "Set VULKAN_SDK or add glslc to PATH.")
else()
    file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/shaders/*.vert"
        "${CMAKE_SOURCE_DIR}/shaders/*.frag"
        "${CMAKE_SOURCE_DIR}/shaders/*.comp")

    set(SPIRV_OUTPUTS)
    set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPV_OUT "${SHADER_OUT_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT  ${SPV_OUT}
            COMMAND ${GLSLC} --target-env=vulkan1.3 -O ${SHADER} -o ${SPV_OUT}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader ${SHADER_NAME}"
            VERBATIM)
        list(APPEND SPIRV_OUTPUTS ${SPV_OUT})
    endforeach()

    add_custom_target(hrp_shaders DEPENDS ${SPIRV_OUTPUTS})
    add_dependencies(hrp_demo hrp_shaders)
endif()
