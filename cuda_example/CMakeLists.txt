project(opencv_example_cuda)

add_executable(opencv_cuda_example opencv_cuda_example.cpp)
target_include_directories(opencv_cuda_example PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_cuda_example ${OpenCV_LIBRARIES})
