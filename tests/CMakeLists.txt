set(TEST_MAIN ${PROJECT_NAME}_unit_tests)
set(TEST_SOURCES library.cpp)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)

add_executable(${TEST_MAIN} ${TEST_SOURCES} ${APP_SOURCES} ${APP_HEADERS})

#target_link_libraries(${TEST_MAIN} PUBLIC ${PROJECT_NAME})
target_link_libraries(${TEST_MAIN} PRIVATE
        GTest::gtest_main
        mmpaszkowski::mnist_loader
        monogon::monogon
        fmt::fmt
        )

if (${MNIST_ENABLE_WARNINGS})
    target_set_warnings(TARGET ${TEST_MAIN} ENABLE ON AS_ERROR OFF)
endif ()

add_target_static_analyers(${TEST_MAIN})

if (MNIST_ENABLE_COVERAGE)
    set(COVERAGE_MAIN "coverage")
    set(COVERAGE_EXCLUDES
            "${PROJECT_SOURCE_DIR}/app/*"
            "${PROJECT_SOURCE_DIR}/cmake/*"
            "${PROJECT_SOURCE_DIR}/docs/*"
            "${PROJECT_SOURCE_DIR}/external/*"
            "${PROJECT_SOURCE_DIR}/tests/*"
            "/usr/include/*")
    setup_target_for_coverage_lcov(
            NAME ${COVERAGE_MAIN}
            EXECUTABLE ${TEST_MAIN}
            DEPENDENCIES ${TEST_MAIN}
    )
endif()