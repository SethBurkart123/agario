cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0167 OLD)
project(agario_cpp_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED COMPONENTS thread json)
find_package(Threads REQUIRED)

add_executable(agario_server
  src/main.cpp
  src/server.cpp
  src/world.cpp
  src/config.cpp
  src/bots/manager.cpp
  src/bots/registry.cpp
  src/bots/core.cpp
)

target_include_directories(agario_server PRIVATE src)
target_compile_definitions(agario_server PRIVATE BOOST_ERROR_CODE_HEADER_ONLY BOOST_SYSTEM_NO_DEPRECATED)
target_link_libraries(agario_server PRIVATE Boost::thread Boost::json Threads::Threads)
