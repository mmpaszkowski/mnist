set (MAIN_SOURCE "main.cpp")

add_executable(${PROJECT_NAME} ${MAIN_SOURCE} ${APP_SOURCES} ${APP_HEADERS})

find_package(Matplot++ REQUIRED)

target_link_libraries(${PROJECT_NAME} PUBLIC
    mmpaszkowski::mnist_loader
    monogon::monogon
    fmt::fmt
    Matplot++::matplot)

target_include_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_INCLUDES})


if (${MNIST_ENABLE_WARNINGS})
    target_set_warnings(TARGET ${PROJECT_NAME} ENABLE ON AS_ERROR OFF)
endif()

add_target_static_analyers(${PROJECT_NAME})
