set(DEPTHAI_BOOTLOADER_SHARED_FOLDER ${CMAKE_CURRENT_LIST_DIR}/depthai-bootloader-shared)

set(DEPTHAI_BOOTLOADER_SHARED_SOURCES
    ${DEPTHAI_BOOTLOADER_SHARED_FOLDER}/src/SBR.c
)

set(DEPTHAI_BOOTLOADER_SHARED_PUBLIC_INCLUDE
    ${DEPTHAI_BOOTLOADER_SHARED_FOLDER}/include
)

set(DEPTHAI_BOOTLOADER_SHARED_3RDPARTY_INCLUDE
    ${DEPTHAI_BOOTLOADER_SHARED_FOLDER}/3rdparty
)

set(DEPTHAI_BOOTLOADER_SHARED_INCLUDE
    ${DEPTHAI_BOOTLOADER_SHARED_FOLDER}/src
)

# Try retriving depthai-bootloader-shared commit hash
find_package(Git)
if(GIT_FOUND AND NOT DEPTHAI_DOWNLOADED_SOURCES)

    # Check that submodule is initialized and updated
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule status ${DEPTHAI_BOOTLOADER_SHARED_FOLDER}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE statusCommit
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(SUBSTRING ${statusCommit} 0 1 status)
    if(${status} STREQUAL "-")
        message(FATAL_ERROR "Submodule 'depthai-shared' not initialized/updated. Run 'git submodule update --init --recursive' first")
    endif()   
    
    # Get depthai-bootloader-shared current commit
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${DEPTHAI_BOOTLOADER_SHARED_FOLDER}
        RESULT_VARIABLE DEPTHAI_BOOTLOADER_SHARED_COMMIT_RESULT
        OUTPUT_VARIABLE DEPTHAI_BOOTLOADER_SHARED_COMMIT_HASH
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(${DEPTHAI_BOOTLOADER_SHARED_COMMIT_RESULT} EQUAL 0)
        set(DEPTHAI_BOOTLOADER_SHARED_COMMIT_FOUND TRUE)
    else()
        set(DEPTHAI_BOOTLOADER_SHARED_COMMIT_FOUND FALSE)
    endif()
endif()