# Install script for directory: /home/ar_dev/ar_server/lib/cudasift

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES
    "/home/ar_dev/ar_server/lib/cudasift/cudaImage.cu"
    "/home/ar_dev/ar_server/lib/cudasift/cudaImage.h"
    "/home/ar_dev/ar_server/lib/cudasift/cudaSiftH.cu"
    "/home/ar_dev/ar_server/lib/cudasift/cudaSiftH.h"
    "/home/ar_dev/ar_server/lib/cudasift/matching.cu"
    "/home/ar_dev/ar_server/lib/cudasift/cudaSiftD.h"
    "/home/ar_dev/ar_server/lib/cudasift/cudaSift.h"
    "/home/ar_dev/ar_server/lib/cudasift/cudautils.h"
    "/home/ar_dev/ar_server/lib/cudasift/geomFuncs.cpp"
    "/home/ar_dev/ar_server/lib/cudasift/mainSift.cpp"
    "/home/ar_dev/ar_server/lib/cudasift/cudaSiftD.cu"
    "/home/ar_dev/ar_server/lib/cudasift/CMakeLists.txt"
    "/home/ar_dev/ar_server/lib/cudasift/Copyright.txt"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/data" TYPE FILE FILES
    "/home/ar_dev/ar_server/lib/cudasift/data/left.pgm"
    "/home/ar_dev/ar_server/lib/cudasift/data/righ.pgm"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ar_dev/ar_server/lib/cudasift/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
