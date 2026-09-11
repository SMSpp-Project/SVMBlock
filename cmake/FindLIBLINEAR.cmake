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
#        LIBLINEAR_DLL           - The found runtime DLL (Windows only)       #
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

# Check if already in cache
if (WIN32)
    if (LIBLINEAR_INCLUDE_DIR AND LIBLINEAR_LIBRARY AND LIBLINEAR_LIBRARY_DEBUG
            AND LIBLINEAR_DLL AND LIBLINEAR_VERSION)
        set(LIBLINEAR_FOUND TRUE)
    endif ()
else ()
    if (LIBLINEAR_INCLUDE_DIR AND LIBLINEAR_LIBRARY AND LIBLINEAR_VERSION)
        set(LIBLINEAR_FOUND TRUE)
    endif ()
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
    if (UNIX)
        find_library(LIBLINEAR_LIBRARY
                NAMES linear liblinear
                PATHS ${LIBLINEAR_ROOT}/lib
                DOC "LIBLINEAR library.")

        set(LIBLINEAR_LIBRARY_DEBUG ${LIBLINEAR_LIBRARY}
                CACHE FILEPATH "LIBLINEAR debug library." FORCE)
    elseif (WIN32)
        find_library(LIBLINEAR_LIBRARY
                NAMES linear liblinear
                PATHS
                ${LIBLINEAR_ROOT}/lib
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib
                $ENV{LIBRARY_LIB}
                NO_DEFAULT_PATH
                DOC "LIBLINEAR library.")

        find_library(LIBLINEAR_LIBRARY_DEBUG
                NAMES linear liblinear
                PATHS
                ${LIBLINEAR_ROOT}/debug/lib
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib
                NO_DEFAULT_PATH
                DOC "LIBLINEAR debug library.")

        # Release-only distributions ship no debug build: fall back to the
        # release library so that a Release configure succeeds.
        if (NOT LIBLINEAR_LIBRARY_DEBUG)
            set(LIBLINEAR_LIBRARY_DEBUG ${LIBLINEAR_LIBRARY}
                    CACHE FILEPATH "LIBLINEAR debug library." FORCE)
        endif ()

        # ----- Find the LIBLINEAR runtime DLL on Windows ------------------- #
        find_file(LIBLINEAR_DLL
                NAMES linear.dll liblinear.dll
                PATHS
                ${LIBLINEAR_ROOT}/bin
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin
                $ENV{LIBRARY_BIN}
                NO_DEFAULT_PATH
                DOC "LIBLINEAR runtime DLL.")

        find_file(LIBLINEAR_DLL_DEBUG
                NAMES linear.dll liblinear.dll
                PATHS
                ${LIBLINEAR_ROOT}/debug/bin
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin
                NO_DEFAULT_PATH
                DOC "LIBLINEAR debug runtime DLL.")

        if (NOT LIBLINEAR_DLL_DEBUG AND LIBLINEAR_DLL)
            set(LIBLINEAR_DLL_DEBUG ${LIBLINEAR_DLL}
                    CACHE FILEPATH "LIBLINEAR debug runtime DLL." FORCE)
        endif ()
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

        set(LIBLINEAR_VERSION "${_LIBLINEAR_version_major}.${_LIBLINEAR_version_minor}")
        unset(_LIBLINEAR_version_line)
        unset(_LIBLINEAR_version)
        unset(_LIBLINEAR_version_major)
        unset(_LIBLINEAR_version_minor)
    endif ()

    # ----- Handle the standard arguments ----------------------------------- #
    # The following macro manages the QUIET, REQUIRED and version-related
    # options passed to find_package(). It also sets <PackageName>_FOUND if
    # REQUIRED_VARS are set.
    if (WIN32)
        # The debug library/DLL are optional (they fall back to the release
        # ones above), so they are deliberately kept out of REQUIRED_VARS.
        find_package_handle_standard_args(
                LIBLINEAR
                REQUIRED_VARS LIBLINEAR_LIBRARY LIBLINEAR_DLL LIBLINEAR_INCLUDE_DIR
                VERSION_VAR LIBLINEAR_VERSION)
    else ()
        find_package_handle_standard_args(
                LIBLINEAR
                REQUIRED_VARS LIBLINEAR_LIBRARY LIBLINEAR_INCLUDE_DIR
                VERSION_VAR LIBLINEAR_VERSION)
    endif ()
endif ()

# ----- Export the target --------------------------------------------------- #
if (LIBLINEAR_FOUND)
    set(LIBLINEAR_INCLUDE_DIRS ${LIBLINEAR_INCLUDE_DIR})
    set(LIBLINEAR_LIBRARIES ${LIBLINEAR_LIBRARY})

    if (NOT TARGET LIBLINEAR::LIBLINEAR)
        if (WIN32)
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
