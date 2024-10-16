find_path(MPI_INCLUDE_DIR
    NAMES mpi.h
    HINTS /usr/include /usr/local/include /opt/local/include /sw/include
    PATH_SUFFIXES mpi openmpi mpich
)

find_library(MPI_LIBRARY
    NAMES mpi mpich openmpi
    HINTS /usr/lib /usr/local/lib /opt/local/lib /sw/lib
    PATH_SUFFIXES mpi openmpi mpich
)

if(NOT MPI_INCLUDE_DIR OR NOT MPI_LIBRARY)
    message(FATAL_ERROR "Could not find MPI. Please make sure MPI is installed and properly configured.")
endif()