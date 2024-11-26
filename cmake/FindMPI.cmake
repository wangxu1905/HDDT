
find_path(MPI_INCLUDE_DIR
  NAMES mpi.h
  HINTS /usr/lib/x86_64-linux-gnu/openmpi/include
  PATHS /usr/lib/x86_64-linux-gnu/openmpi/include
  PATH_SUFFIXES openmpi
)

find_library(MPI_LIBRARY
  NAMES mpi
  HINTS /usr/lib/x86_64-linux-gnu/openmpi/lib
  PATHS /usr/lib/x86_64-linux-gnu/openmpi/lib
)

if(MPI_INCLUDE_DIR AND MPI_LIBRARY)
  message(STATUS "Found MPI: includes in ${MPI_INCLUDE_DIR}, libraries in ${MPI_LIBRARY}")
  mark_as_advanced(MPI_INCLUDE_DIR MPI_LIBRARY)
else()
  message(FATAL_ERROR "Could not find MPI. Please make sure MPI is installed and properly configured.")
endif()

# include_directories(${MPI_INCLUDE_DIR})
# target_link_libraries(your_target_name ${MPI_LIBRARY})

# 
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib")

# # -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi