# --------------------------------------------------------------------------- #
#    CMake find module for LIBSVM                                             #
#                                                                             #
#    This module finds LIBSVM include directories and libraries.              #
#    Use it by invoking find_package() with the form:                         #
#                                                                             #
#        find_package(LIBSVM [version] [EXACT] [REQUIRED])                    #
#                                                                             #
#    The results are stored in the following variables:                       #
#                                                                             #
#        LIBSVM_FOUND         - True if headers are found                     #
#        LIBSVM_INCLUDE_DIRS  - Include directories                           #
#        LIBSVM_LIBRARIES     - Libraries to be linked                        #
#        LIBSVM_DLL           - The runtime DLL, where there is one           #
#        LIBSVM_VERSION       - Version number                                #
#                                                                             #
#    This module reads hints about search locations from variables:           #
#                                                                             #
#        LIBSVM_ROOT          - Custom path to LIBSVM                         #
#                                                                             #
#    The following IMPORTED target is also defined:                           #
#                                                                             #
#        LIBSVM::LIBSVM                                                       #
#                                                                             #
#    This find module is provided because LIBSVM does not provide             #
#    a CMake configuration file on its own.                                   #
#                                                                             #
#                                Donato Meoli                                 #
#                         Dipartimento di Informatica                         #
#                             Universita' di Pisa                             #
# --------------------------------------------------------------------------- #
include(FindPackageHandleStandardArgs)

# Drop the cached paths if one of them does not exist any more, so that
# they are all searched again from the same place
set(_LIBSVM_cached LIBSVM_INCLUDE_DIR LIBSVM_LIBRARY LIBSVM_LIBRARY_DEBUG
        LIBSVM_DLL LIBSVM_DLL_DEBUG LIBSVM_VERSION)
foreach (_var ${_LIBSVM_cached})
    foreach (_path ${${_var}})
        if (IS_ABSOLUTE "${_path}" AND NOT EXISTS "${_path}")
            set(_LIBSVM_stale TRUE)
        endif ()
    endforeach ()
endforeach ()
if (_LIBSVM_stale)
    message(STATUS "Cached LIBSVM paths not found, searching again")
    foreach (_var ${_LIBSVM_cached})
        unset(${_var} CACHE)
    endforeach ()
endif ()
unset(_LIBSVM_cached)
unset(_LIBSVM_stale)
unset(_var)
unset(_path)

# The cached version is the one of the include directory it was read from
if (NOT "${LIBSVM_INCLUDE_DIR}" STREQUAL
        "${_LIBSVM_VERSION_INCLUDE_DIR}")
    unset(LIBSVM_VERSION CACHE)
endif ()

# Check if already in cache
if (LIBSVM_INCLUDE_DIR AND LIBSVM_LIBRARY AND LIBSVM_VERSION)
    set(LIBSVM_FOUND TRUE)
endif ()

if (NOT LIBSVM_FOUND)

    # ----- Find the LIBSVM include directory ------------------------------- #
    # svm.h is in a libsvm directory in some distributions (the Debian
    # package, the vcpkg port) and right in the include directory in others
    # (the Homebrew package): what is searched for is the directory holding
    # either, and LIBSVMSolver.h includes the one that is there.
    find_path(LIBSVM_INCLUDE_DIR
            NAMES libsvm/svm.h svm.h
            PATHS ${LIBSVM_ROOT}
            PATH_SUFFIXES include
            DOC "LIBSVM include directory.")

    # ----- Find the LIBSVM library ----------------------------------------- #
    # the release library, then the debug one where a distribution has it
    # (vcpkg), else the release one; the runtime DLL where the library is
    # one (Windows), which the imported target then carries; the places that
    # a system does not have are simply not there
    find_library(LIBSVM_LIBRARY
            NAMES svm libsvm
            HINTS ${LIBSVM_ROOT} ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}
                  $ENV{LIBRARY_PREFIX}
            PATH_SUFFIXES lib
            DOC "LIBSVM library.")

    find_library(LIBSVM_LIBRARY_DEBUG
            NAMES svm libsvm
            HINTS ${LIBSVM_ROOT}/debug
                  ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug
            PATH_SUFFIXES lib
            NO_DEFAULT_PATH
            DOC "LIBSVM debug library.")
    if (NOT LIBSVM_LIBRARY_DEBUG)
        set(LIBSVM_LIBRARY_DEBUG ${LIBSVM_LIBRARY}
                CACHE FILEPATH "LIBSVM debug library." FORCE)
    endif ()

    find_file(LIBSVM_DLL
            NAMES svm.dll libsvm.dll
            HINTS ${LIBSVM_ROOT} ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}
                  $ENV{LIBRARY_PREFIX}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH
            DOC "LIBSVM runtime DLL.")

    find_file(LIBSVM_DLL_DEBUG
            NAMES svm.dll libsvm.dll
            HINTS ${LIBSVM_ROOT}/debug
                  ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH
            DOC "LIBSVM debug runtime DLL.")
    if (NOT LIBSVM_DLL_DEBUG AND LIBSVM_DLL)
        set(LIBSVM_DLL_DEBUG ${LIBSVM_DLL}
                CACHE FILEPATH "LIBSVM debug runtime DLL." FORCE)
    endif ()

    # ----- Parse the version ----------------------------------------------- #
    # LIBSVM_VERSION is an integer, e.g. 324 for 3.24
    if (LIBSVM_INCLUDE_DIR)
        if (EXISTS "${LIBSVM_INCLUDE_DIR}/libsvm/svm.h")
            set(_LIBSVM_header "${LIBSVM_INCLUDE_DIR}/libsvm/svm.h")
        else ()
            set(_LIBSVM_header "${LIBSVM_INCLUDE_DIR}/svm.h")
        endif ()
        file(STRINGS "${_LIBSVM_header}"
                _LIBSVM_version_line REGEX "#define LIBSVM_VERSION")
        unset(_LIBSVM_header)

        string(REGEX REPLACE ".*LIBSVM_VERSION *\([0-9]*\).*" "\\1"
                _LIBSVM_version "${_LIBSVM_version_line}")
        math(EXPR _LIBSVM_version_major "${_LIBSVM_version} / 100")
        math(EXPR _LIBSVM_version_minor "${_LIBSVM_version} % 100")

        set(LIBSVM_VERSION "${_LIBSVM_version_major}.${_LIBSVM_version_minor}")
        unset(_LIBSVM_version_line)
        unset(_LIBSVM_version)
        unset(_LIBSVM_version_major)
        unset(_LIBSVM_version_minor)
    endif ()

    # ----- Handle the standard arguments ----------------------------------- #
    # The following macro manages the QUIET, REQUIRED and version-related
    # options passed to find_package(). It also sets <PackageName>_FOUND if
    # REQUIRED_VARS are set.
    find_package_handle_standard_args(
            LIBSVM
            REQUIRED_VARS LIBSVM_LIBRARY LIBSVM_INCLUDE_DIR
            VERSION_VAR LIBSVM_VERSION)
endif ()

# A found package keeps its version in the cache, with the include
# directory it was read from
if (LIBSVM_FOUND)
    set(LIBSVM_VERSION "${LIBSVM_VERSION}" CACHE INTERNAL
            "LIBSVM version.")
    set(_LIBSVM_VERSION_INCLUDE_DIR "${LIBSVM_INCLUDE_DIR}"
            CACHE INTERNAL "Include directory of the LIBSVM version.")
endif ()

# ----- Export the target --------------------------------------------------- #
if (LIBSVM_FOUND)
    set(LIBSVM_INCLUDE_DIRS ${LIBSVM_INCLUDE_DIR})
    set(LIBSVM_LIBRARIES ${LIBSVM_LIBRARY})

    if (NOT TARGET LIBSVM::LIBSVM)
        # a DLL with its import library, or a library linked as it is
        if (LIBSVM_DLL)
            add_library(LIBSVM::LIBSVM SHARED IMPORTED)
            set_target_properties(
                    LIBSVM::LIBSVM PROPERTIES
                    IMPORTED_IMPLIB "${LIBSVM_LIBRARY}"
                    IMPORTED_IMPLIB_DEBUG "${LIBSVM_LIBRARY_DEBUG}"
                    IMPORTED_LOCATION "${LIBSVM_DLL}"
                    IMPORTED_LOCATION_DEBUG "${LIBSVM_DLL_DEBUG}"
                    INTERFACE_INCLUDE_DIRECTORIES "${LIBSVM_INCLUDE_DIRS}")
        else ()
            add_library(LIBSVM::LIBSVM UNKNOWN IMPORTED)
            set_target_properties(
                    LIBSVM::LIBSVM PROPERTIES
                    IMPORTED_LOCATION "${LIBSVM_LIBRARY}"
                    IMPORTED_LOCATION_DEBUG "${LIBSVM_LIBRARY_DEBUG}"
                    INTERFACE_INCLUDE_DIRECTORIES "${LIBSVM_INCLUDE_DIRS}")
        endif ()
    endif ()
endif ()

# --------------------------------------------------------------------------- #
