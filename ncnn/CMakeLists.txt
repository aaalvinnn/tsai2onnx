cmake_minimum_required(VERSION 3.5)
find_package(REQUIRED)

# include directories
include_directories(ncnn/build/install/include/ncnn) # ncnn
include_directories(lib)    # user include dir
link_directories(ncnn/build/install/lib)
link_directories(lib)

FIND_PACKAGE( OpenMP REQUIRED)  
if(OPENMP_FOUND)  
    message("OPENMP FOUND")  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  

## --- C++11 build flags ---
add_definitions(-D_GLIBCXX_USE_C99=1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## build
add_executable(main)
aux_source_directory(. sources)
aux_source_directory(lib sources)
target_sources(main PUBLIC ${sources})
target_link_libraries(main ncnn)

