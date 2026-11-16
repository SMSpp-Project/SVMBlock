# --------------------------------------------------------------------------- #
#    CMake file for SVMBlock/test                                             #
#                                                                             #
#    In this file we define the unit tests for this library.                  #
#    Tests that should be supported by ctest executable must be added         #
#    with the add_test() command.                                             #
#                                                                             #
#                                Donato Meoli                                 #
#                         Dipartimento di Informatica                         #
#                             Universita' di Pisa                             #
# --------------------------------------------------------------------------- #

# The classes of the modules register themselves in the factories from
# static objects, which nothing here names: a shared library is kept by
# what its module asks of the linker, while an archive has to be taken
# whole [see SMSppModule.cmake of SMS++].
smspp_keep_modules()

# ----- SVMBlock_test --------------------------------------------------      #

add_executable(${modName}_test test.cpp)
target_link_libraries(${modName}_test PRIVATE ${modNamespace}::${modName})

smspp_copy_runtime(${modName}_test)

add_test(NAME ${modName}_test
         COMMAND ${modName}_test
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# The label lets the module CI select exactly the tests exercising this
# module with `ctest -L <module>` (see tests/cmake/TestLabels.cmake in the
# umbrella for the system-tests counterpart).
set_tests_properties(${modName}_test PROPERTIES LABELS "${modName}")

# --------------------------------------------------------------------------- #
