find_program(GLSLC glslc REQUIRED)

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
set(SHADER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SHADERS
    ${SHADER_SOURCE_DIR}/*.vert
    ${SHADER_SOURCE_DIR}/*.frag
    ${SHADER_SOURCE_DIR}/*.comp
)

file(GLOB_RECURSE GLSL_INCLUDES ${SHADER_SOURCE_DIR}/*.glsl)

set(SPIRV_SHADERS)

foreach(shader ${SHADERS})
    get_filename_component(FILENAME ${shader} NAME)
    set(output ${SHADER_OUTPUT_DIR}/${FILENAME}.spv)

    add_custom_command(
        OUTPUT ${output}
        COMMAND ${GLSLC} 
            ${shader}
            -I ${SHADER_INCLUDE_DIR}
            -o ${output}
        DEPENDS ${shader} ${GLSL_INCLUDES}
        COMMENT "Compiling shader ${FILENAME}"
        VERBATIM
    )

    list(APPEND SPIRV_SHADERS ${output})
endforeach()

add_custom_target(shaders
    DEPENDS ${SPIRV_SHADERS}
)