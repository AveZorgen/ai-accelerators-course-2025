file(GLOB child_dirs
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "*")

set(run_targets)

foreach(child ${child_dirs})
    set(child_path "${CMAKE_CURRENT_SOURCE_DIR}/${child}")
    if(IS_DIRECTORY "${child_path}" AND EXISTS "${child_path}/run.sh")
        string(REGEX REPLACE "[^A-Za-z0-9_]" "_" target_suffix "${child}")
        set(target_name "run_mixed_ascend_${target_suffix}")

        add_custom_target(${target_name}
            COMMAND bash run.sh --soc-version Ascend310P3 --run-mode sim
            WORKING_DIRECTORY "${child_path}"
            USES_TERMINAL
        )

        list(APPEND run_targets ${target_name})
    endif()
endforeach()

if(run_targets)
    add_custom_target(run_mixed_ascend_all
        DEPENDS ${run_targets}
    )
endif()
