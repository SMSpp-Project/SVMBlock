# --------------------------------------------------------------------------- #
#    CMake find module for LIBLINEAR                                          #
#                                                                             #
#    This module finds LIBLINEAR include directories and libraries.           #
#    Use it by invoking find_package() with the form:                         #
#                                                                             #
#        find_package(LIBLINEAR [version] [EXACT] [REQUIRED])                 #
#                                                                             #
#    The results are stored in the following variables:                       #
#                                                                             #
#        LIBLINEAR_FOUND         - True if headers are found                  #
#        LIBLINEAR_INCLUDE_DIRS  - Include directories                        #
#        LIBLINEAR_LIBRARIES     - Libraries to be linked                     #
#        LIBLINEAR_DLL           - The runtime DLL, where there is one        #
#        LIBLINEAR_VERSION       - Version number                             #
#                                                                             #
#    This module reads hints about search locations from variables:           #
#                                                                             #
#        LIBLINEAR_ROOT          - Custom path to LIBLINEAR                   #
#                                                                             #
#    The following IMPORTED target is also defined:                           #
#                                                                             #
#        LIBLINEAR::LIBLINEAR                                                 #
#                                                                             #
#    This find module is provided because LIBLINEAR does not provide          #
#    a CMake configuration file on its own.                                   #
#                                                                             #
#                                Donato Meoli                                 #
#                         Dipartimento di Informatica                         #
#                             Universita' di Pisa                             #
# --------------------------------------------------------------------------- #
include(FindPackageHandleStandardArgs)

# Drop the cached paths if one of them does not exist any more, so that
# they are all searched again from the same place
set(_LIBLINEAR_cached LIBLINEAR_INCLUDE_DIR LIBLINEAR_LIBRARY
        LIBLINEAR_LIBRARY_DEBUG LIBLINEAR_DLL LIBLINEAR_DLL_DEBUG
        LIBLINEAR_VERSION)
foreach (_var ${_LIBLINEAR_cached})
    foreach (_path ${${_var}})
        if (IS_ABSOLUTE "${_path}" AND NOT EXISTS "${_path}")
            set(_LIBLINEAR_stale TRUE)
        endif ()
    endforeach ()
endforeach ()
if (_LIBLINEAR_stale)
    message(STATUS "Cached LIBLINEAR paths not found, searching again")
    foreach (_var ${_LIBLINEAR_cached})
        unset(${_var} CACHE)
    endforeach ()
endif ()
unset(_LIBLINEAR_cached)
unset(_LIBLINEAR_stale)
unset(_var)
unset(_path)

# The cached version is the one of the include directory it was read from
if (NOT "${LIBLINEAR_INCLUDE_DIR}" STREQUAL
        "${_LIBLINEAR_VERSION_INCLUDE_DIR}")
    unset(LIBLINEAR_VERSION CACHE)
endif ()

# Check if already in cache
if (LIBLINEAR_INCLUDE_DIR AND LIBLINEAR_LIBRARY AND LIBLINEAR_VERSION)
    set(LIBLINEAR_FOUND TRUE)
endif ()

if (NOT LIBLINEAR_FOUND)

    # ----- Find the LIBLINEAR include directory ---------------------------- #
    # linear.h is included as <linear.h>: the Debian package puts it straight
    # into the include directory, other distributions into one of their own,
    # so what is searched for is the directory containing it, whichever it is,
    # and that directory is the one that ends up on the include path.
    find_path(LIBLINEAR_INCLUDE_DIR
            NAMES linear.h
            PATHS ${LIBLINEAR_ROOT}
            PATH_SUFFIXES include include/liblinear
            DOC "LIBLINEAR include directory.")

    # ----- Find the LIBLINEAR library -------------------------------------- #
    # the release library, then the debug one where a distribution has it
    # (vcpkg), else the release one; the runtime DLL where the library is
    # one (Windows), which the imported target then carries; the places that
    # a system does not have are simply not there
    find_library(LIBLINEAR_LIBRARY
            NAMES linear liblinear
            HINTS ${LIBLINEAR_ROOT}
                  ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}
                  $ENV{LIBRARY_PREFIX}
            PATH_SUFFIXES lib
            DOC "LIBLINEAR library.")

    find_library(LIBLINEAR_LIBRARY_DEBUG
            NAMES linear liblinear
            HINTS ${LIBLINEAR_ROOT}/debug
                  ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug
            PATH_SUFFIXES lib
            NO_DEFAULT_PATH
            DOC "LIBLINEAR debug library.")
    if (NOT LIBLINEAR_LIBRARY_DEBUG)
        set(LIBLINEAR_LIBRARY_DEBUG ${LIBLINEAR_LIBRARY}
                CACHE FILEPATH "LIBLINEAR debug library." FORCE)
    endif ()

    find_file(LIBLINEAR_DLL
            NAMES linear.dll liblinear.dll
            HINTS ${LIBLINEAR_ROOT}
                  ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}
                  $ENV{LIBRARY_PREFIX}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH
            DOC "LIBLINEAR runtime DLL.")

    find_file(LIBLINEAR_DLL_DEBUG
            NAMES linear.dll liblinear.dll
            HINTS ${LIBLINEAR_ROOT}/debug
                  ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH
            DOC "LIBLINEAR debug runtime DLL.")
    if (NOT LIBLINEAR_DLL_DEBUG AND LIBLINEAR_DLL)
        set(LIBLINEAR_DLL_DEBUG ${LIBLINEAR_DLL}
                CACHE FILEPATH "LIBLINEAR debug runtime DLL." FORCE)
    endif ()

    # ----- Parse the version ----------------------------------------------- #
    # LIBLINEAR_VERSION is an integer, e.g. 230 for 2.30
    if (LIBLINEAR_INCLUDE_DIR)
        file(STRINGS
                "${LIBLINEAR_INCLUDE_DIR}/linear.h"
                _LIBLINEAR_version_line REGEX "#define LIBLINEAR_VERSION")

        string(REGEX REPLACE ".*LIBLINEAR_VERSION *\([0-9]*\).*" "\\1"
                _LIBLINEAR_version "${_LIBLINEAR_version_line}")
        math(EXPR _LIBLINEAR_version_major "${_LIBLINEAR_version} / 100")
        math(EXPR _LIBLINEAR_version_minor "${_LIBLINEAR_version} % 100")

        set(LIBLINEAR_VERSION
                "${_LIBLINEAR_version_major}.${_LIBLINEAR_version_minor}")
        unset(_LIBLINEAR_version_line)
        unset(_LIBLINEAR_version)
        unset(_LIBLINEAR_version_major)
        unset(_LIBLINEAR_version_minor)
    endif ()

    # ----- Handle the standard arguments ----------------------------------- #
    # The following macro manages the QUIET, REQUIRED and version-related
    # options passed to find_package(). It also sets <PackageName>_FOUND if
    # REQUIRED_VARS are set.
    find_package_handle_standard_args(
            LIBLINEAR
            REQUIRED_VARS LIBLINEAR_LIBRARY LIBLINEAR_INCLUDE_DIR
            VERSION_VAR LIBLINEAR_VERSION)
endif ()

# A found package keeps its version in the cache, with the include
# directory it was read from
if (LIBLINEAR_FOUND)
    set(LIBLINEAR_VERSION "${LIBLINEAR_VERSION}" CACHE INTERNAL
            "LIBLINEAR version.")
    set(_LIBLINEAR_VERSION_INCLUDE_DIR "${LIBLINEAR_INCLUDE_DIR}"
            CACHE INTERNAL "Include directory of the LIBLINEAR version.")
endif ()

# ----- Export the target --------------------------------------------------- #
if (LIBLINEAR_FOUND)
    set(LIBLINEAR_INCLUDE_DIRS ${LIBLINEAR_INCLUDE_DIR})
    set(LIBLINEAR_LIBRARIES ${LIBLINEAR_LIBRARY})

    if (NOT TARGET LIBLINEAR::LIBLINEAR)
        # a DLL with its import library, or a library linked as it is
        if (LIBLINEAR_DLL)
            add_library(LIBLINEAR::LIBLINEAR SHARED IMPORTED)
            set_target_properties(
                    LIBLINEAR::LIBLINEAR PROPERTIES
                    IMPORTED_IMPLIB "${LIBLINEAR_LIBRARY}"
                    IMPORTED_IMPLIB_DEBUG "${LIBLINEAR_LIBRARY_DEBUG}"
                    IMPORTED_LOCATION "${LIBLINEAR_DLL}"
                    IMPORTED_LOCATION_DEBUG "${LIBLINEAR_DLL_DEBUG}"
                    INTERFACE_INCLUDE_DIRECTORIES "${LIBLINEAR_INCLUDE_DIRS}")
        else ()
            add_library(LIBLINEAR::LIBLINEAR UNKNOWN IMPORTED)
            set_target_properties(
                    LIBLINEAR::LIBLINEAR PROPERTIES
                    IMPORTED_LOCATION "${LIBLINEAR_LIBRARY}"
                    IMPORTED_LOCATION_DEBUG "${LIBLINEAR_LIBRARY_DEBUG}"
                    INTERFACE_INCLUDE_DIRECTORIES "${LIBLINEAR_INCLUDE_DIRS}")
        endif ()
    endif ()
endif ()

# --------------------------------------------------------------------------- #
