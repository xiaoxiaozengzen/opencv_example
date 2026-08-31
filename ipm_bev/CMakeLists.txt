cmake_minimum_required(VERSION 3.14)
project(ipm_bev)

find_package(OpenCV REQUIRED)

set(TARGET_NAME ${PROJECT_NAME})


add_executable(${TARGET_NAME}_exe ${SRC_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME}.cpp)
target_include_directories(${TARGET_NAME}_exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME}_exe ${OpenCV_LIBRARIES})