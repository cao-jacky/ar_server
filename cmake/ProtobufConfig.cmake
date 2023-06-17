# Set the installation prefix
set(PROTOBUF_PREFIX_PATH "/protobuf-3.17.3" CACHE PATH "Protobuf installation prefix")

# Find the Protobuf libraries
find_library(PROTOBUF_LIBRARY NAMES protobuf HINTS ${PROTOBUF_PREFIX_PATH}/lib)

# Find the Protobuf headers
find_path(PROTOBUF_INCLUDE_DIR google/protobuf/message.h HINTS ${PROTOBUF_PREFIX_PATH}/include)

# Set the Protobuf version
set(PROTOBUF_VERSION "3.17.3")

# Provide the configuration information
set(Protobuf_FOUND TRUE)
set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIR})
set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARY})
set(Protobuf_VERSION ${PROTOBUF_VERSION})

