# SOURCES AND HEADERS
set(APP_SOURCES
    ../src/library.cpp PARENT_SCOPE)

set(APP_HEADERS
        ../include/mnist/library.h PARENT_SCOPE)

set(LIBRARY_INCLUDES
        ../include
        "${CMAKE_BINARY_DIR}/configured_files/include"
        PARENT_SCOPE)