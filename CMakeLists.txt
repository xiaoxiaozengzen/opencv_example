cmake_minimum_required(VERSION 3.14)
project(opencv_example)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")


add_executable(qr_decode_exe ${SRC_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/src/qr_decode.cpp)
target_include_directories(qr_decode_exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(qr_decode_exe ${OpenCV_LIBRARIES})

add_executable(mat_exe ${SRC_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/src/mat.cpp)
target_include_directories(mat_exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mat_exe ${OpenCV_LIBRARIES})

add_executable(video_writer_exe ${SRC_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/src/video_writer.cpp)
target_include_directories(video_writer_exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(video_writer_exe ${OpenCV_LIBRARIES})

add_subdirectory(cuda_example)