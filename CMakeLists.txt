# --------------------------------------------------------------------------- #
#    Main CMake file for SMS++ SVMBlock                                       #
#                                                                             #
#    This file allows one to build the library using CMake.                   #
#    To do so, you can use the following commands:                            #
#                                                                             #
#        $ cmake -S <source-path> -B <build-path>                             #
#        $ cmake --build <build-path>                                         #
#                                                                             #
#    The following command also installs the library in the system:           #
#                                                                             #
#        $ cmake --build <build-path> --target install                        #
#                                                                             #
#                                Donato Meoli                                 #
#                         Dipartimento di Informatica                         #
#                             Universita' di Pisa                             #
# --------------------------------------------------------------------------- #
cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.15)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeriveVersion.cmake)
smspp_derive_version(SMSPP_MODULE_VERSION)

project(SVMBlock
        VERSION ${SMSPP_MODULE_VERSION}
        DESCRIPTION "SMS++ Support Vector Machines module"
        HOMEPAGE_URL https://gitlab.com/smspp/svmblock
        LANGUAGES C CXX)

# These variables make the code harder to read but easier to change.
set(modName ${PROJECT_NAME})

# C-safe form of modName for the preprocessor macros in <modName>Config.h
string(MAKE_C_IDENTIFIER "${modName}" modMacro)
set(modNamespace "SMS++")

# Find out if it's being called by the umbrella.
get_directory_property(hasParent PARENT_DIRECTORY)

# This adds the cmake directory to the module search paths,
# allowing us to use our modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ----- Settings ------------------------------------------------------------ #
# Sets the default build type (if none was specified).
# See: https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html
include(BuildType)

# ----- Settings ------------------------------------------------------------ #
option(SVMBlock_USE_LIBSVM "Whether SVMBlock will use LIBSVM or not." ON)
option(SVMBlock_USE_LIBLINEAR
       "Whether SVMBlock will use LIBLINEAR or not." ON)

# ----- Requirements -------------------------------------------------------- #
# If it's not being called by the umbrella, we need to
# look for the system-installed libraries.
if (NOT hasParent)
    find_package(SMS++ REQUIRED)
endif ()

if (SVMBlock_USE_LIBSVM)
    # Needed for LIBSVMSolver
    find_package(LIBSVM 3.0)
    if (NOT LIBSVM_FOUND)
        set(SVMBlock_USE_LIBSVM OFF)
    endif ()
endif ()

if (SVMBlock_USE_LIBLINEAR)
    # Needed for LIBLINEARSolver
    find_package(LIBLINEAR 2.0)
    if (NOT LIBLINEAR_FOUND)
        set(SVMBlock_USE_LIBLINEAR OFF)
    endif ()
endif ()

# Whether LIBSVMSolver is in the library, which whoever uses SVMBlock has to
# know: the tests only run the batch that attaches it when it is there. It is
# a cache entry because LIBSVM_FOUND, being set by find_package(), only lives
# in this directory.
set(SVMBlock_HAS_LIBSVM ${SVMBlock_USE_LIBSVM}
    CACHE INTERNAL "Whether SVMBlock was built with LIBSVM.")

set(SVMBlock_HAS_LIBLINEAR ${SVMBlock_USE_LIBLINEAR}
    CACHE INTERNAL "Whether SVMBlock was built with LIBLINEAR.")

# ----- Configuration header ------------------------------------------------ #
# This will generate a *Config.h header in the build directory.
configure_file(cmake/${modName}Config.h.in ${modName}Config.h)

# ----- Library ------------------------------------------------------------- #
# With the BUILD_SHARED_LIBS variable we can specify if the library will
# be STATIC or SHARED, so no reason to do it now.
add_library(${modName})
target_compile_features(${modName} PUBLIC cxx_std_17)

if (MSVC AND BUILD_SHARED_LIBS)
    target_link_options(${modName} INTERFACE "/INCLUDE:SMSpp_force_load_${modName}")
endif ()

# When adding source files with target_sources(), PRIVATE means that the files
# should only be added to this library, whereas PUBLIC means they should be
# added to this library and to any target that links to it.
# INTERFACE can be used for sources that should not be added to this library
# but should be added to anything that links to it.
# Note: do not GLOB files here.
target_sources(${modName} PRIVATE
        src/SVMBlock.cpp
        src/SVCBlock.cpp
        src/SVRBlock.cpp
        src/SMOSolver.cpp)

# When using target_include_directories(), PUBLIC means that any targets
# that link to this target also need that include directory.
# Other options are PRIVATE (only affect the current target, not dependencies),
# and INTERFACE (only needed for dependencies).

# Different INSTALL_INTERFACE and BUILD_INTERFACE paths are used when
# generating the target import file (***Targets.cmake).
# This means that if a target finds this library in its build directory
# will look into the BUILD_INTERFACE path for its headers, if it finds it
# installed in the system will look into the INSTALL_INTERFACE path.
target_include_directories(
        ${modName} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:include/${modNamespace}>)

# When linking other targets to the library with target_link_libraries(),
# PRIVATE means that the libraries will be linked only to this library,
# PUBLIC means they will be linked also to the targets that depend on this
# library, INTERFACE means they will be linked only to the targets that depend
# on this library.
target_link_libraries(${modName} PUBLIC ${modNamespace}::SMS++)

if (SVMBlock_USE_LIBSVM)
    message(STATUS "SVMBlock: LIBSVM found, configuring LIBSVMSolver")
    target_sources(${modName} PRIVATE src/LIBSVMSolver.cpp)
    target_link_libraries(${modName} PUBLIC LIBSVM::LIBSVM)
endif ()

if (SVMBlock_USE_LIBLINEAR)
    message(STATUS "SVMBlock: LIBLINEAR found, configuring LIBLINEARSolver")
    target_sources(${modName} PRIVATE src/LIBLINEARSolver.cpp)
    target_link_libraries(${modName} PUBLIC LIBLINEAR::LIBLINEAR)
endif ()

# This alias is defined so that executables in this same project can use
# the library with this notation.
add_library(${modNamespace}::${modName} ALIAS ${modName})

# ----- Tests --------------------------------------------------------------- #
if (BUILD_TESTING)
    # enable testing in this directory's scope: the umbrella calls
    # enable_testing() only after this subdirectory has been processed, so the
    # test would otherwise not be registered with CTest
    enable_testing()
    add_subdirectory(test)
endif ()

# ----- Custom commands and targets ----------------------------------------- #

# Download libsvm.tgz from GitLab Package Registry if it doesn't exist: the
# data sets of the LIBSVM repository, in the sparse text format that
# SVMBlock::load( istream , 'l' ) reads, which the batches of tests/SVMBlock
# run on
set(DATA_URL "https://gitlab.com/api/v4/projects/85124108/packages/generic/libsvm/latest/libsvm.tgz")
set(DATA_ARCHIVE ${CMAKE_CURRENT_SOURCE_DIR}/data/libsvm.tgz)
set(DATA_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/download_svm_libsvm.cmake)

file(WRITE ${DATA_SCRIPT} "
file(DOWNLOAD \"${DATA_URL}\" \"${DATA_ARCHIVE}\" SHOW_PROGRESS)
")

add_custom_command(
        OUTPUT ${DATA_ARCHIVE}
        COMMAND ${CMAKE_COMMAND} -E echo "Downloading libsvm.tgz from GitLab Package Registry..."
        COMMAND ${CMAKE_COMMAND} -P ${DATA_SCRIPT}
        COMMENT "Downloading libsvm.tgz"
        VERBATIM
)

add_custom_target(download_svm_libsvm
        DEPENDS ${DATA_ARCHIVE})

# Custom command to extract the libsvm.tgz archive if the .extracted file does
# not exist
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/data/.extracted
        COMMAND tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/data/libsvm.tgz
        COMMAND echo > ${CMAKE_CURRENT_SOURCE_DIR}/data/.extracted
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
        COMMENT "Extracting libsvm.tgz"
        VERBATIM
        DEPENDS download_svm_libsvm)

# Create custom targets to run the data extraction
add_custom_target(extract_svm_libsvm
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/.extracted)

# Ensure that extract_svm_libsvm is executed after the download
add_dependencies(extract_svm_libsvm download_svm_libsvm)

# ----- Install instructions ------------------------------------------------ #
# The following commands are used when installing the library
# and its CMake configuration files on the system.
# They are not required for local builds (see below).

include(GNUInstallDirs)

# Install the library
install(TARGETS ${modName}
        EXPORT ${modName}Targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # .dll
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # .so .dylib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}) # .lib

# Install the headers
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${modNamespace})

# Install the auto-generated configuration header (see above).
install(FILES ${PROJECT_BINARY_DIR}/${modName}Config.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${modNamespace})

# (Generate and) install the target import file, that allows other
# CMake projects to import the target.
install(EXPORT ${modName}Targets
        NAMESPACE ${modNamespace}::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${modName})

# Generate the package version file, that allows other
# CMake projects to know the version.
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${modName}ConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion)

# Generate the package configuration file, that allows other
# CMake projects to find the library with find_package().
configure_package_config_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/${modName}Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${modName}Config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${modName})

# Install the package version and configuration files.
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${modName}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${modName}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${modName})

# Install the README and LICENSE files.
install(FILES
        ${CMAKE_CURRENT_LIST_DIR}/README.md
        ${CMAKE_CURRENT_LIST_DIR}/LICENSE
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${modName})

# ----- Add the build tree to RPATH------------------------------------------ #
# Add the binary directory to RPATH, so executables that use
# dynamic loading will look into it.
if (hasParent)
    set(SMSPP_BUILD_RPATH
            "${SMSPP_BUILD_RPATH};${CMAKE_CURRENT_BINARY_DIR}"
            PARENT_SCOPE)
endif ()

# ----- Packaging support --------------------------------------------------- #
# The following commands allow one to build a proper package using CPack.
# See: https://cmake.org/cmake/help/latest/manual/cpack.1.html

# Using this with the umbrella project is too messy.
if (NOT hasParent)
    set(CPACK_PACKAGE_VENDOR "SMS++ Team")
    set(CPACK_PACKAGE_CONTACT "Donato Meoli")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_LIST_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_LIST_DIR}/README.md")
    set(CPACK_SOURCE_IGNORE_FILES
            \\\\.git/
            \\\\.gitignore$
            \\\\.gitattributes$)
    include(CPack)
endif ()

# --------------------------------------------------------------------------- #
# Remove from the search paths the cmake directory we added at the beginning.
list(REMOVE_AT CMAKE_MODULE_PATH -1)

# --------------------------------------------------------------------------- #
