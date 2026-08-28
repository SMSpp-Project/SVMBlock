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
#        LIBSVM_DLL           - The found runtime DLL (Windows only)          #
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

# Check if already in cache
if (WIN32)
    if (LIBSVM_INCLUDE_DIR AND LIBSVM_LIBRARY AND LIBSVM_LIBRARY_DEBUG
            AND LIBSVM_DLL AND LIBSVM_VERSION)
        set(LIBSVM_FOUND TRUE)
    endif ()
else ()
    if (LIBSVM_INCLUDE_DIR AND LIBSVM_LIBRARY AND LIBSVM_VERSION)
        set(LIBSVM_FOUND TRUE)
    endif ()
endif ()

if (NOT LIBSVM_FOUND)

    # ----- Find the LIBSVM include directory ------------------------------- #
    # svm.h is included as <libsvm/svm.h>, which is where every distribution
    # puts it (the Debian and Homebrew packages, the vcpkg port), hence what
    # is searched for is the directory *containing* the libsvm one.
    find_path(LIBSVM_INCLUDE_DIR
            NAMES libsvm/svm.h
            PATHS ${LIBSVM_ROOT}
            PATH_SUFFIXES include
            DOC "LIBSVM include directory.")

    # ----- Find the LIBSVM library ----------------------------------------- #
    if (UNIX)
        find_library(LIBSVM_LIBRARY
                NAMES svm
                PATHS ${LIBSVM_ROOT}/lib
                DOC "LIBSVM library.")

        set(LIBSVM_LIBRARY_DEBUG ${LIBSVM_LIBRARY}
                CACHE FILEPATH "LIBSVM debug library." FORCE)
    elseif (WIN32)
        find_library(LIBSVM_LIBRARY
                NAMES svm libsvm
                PATHS
                ${LIBSVM_ROOT}/lib
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib
                $ENV{LIBRARY_LIB}
                NO_DEFAULT_PATH
                DOC "LIBSVM library.")

        find_library(LIBSVM_LIBRARY_DEBUG
                NAMES svm libsvm
                PATHS
                ${LIBSVM_ROOT}/debug/lib
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib
                NO_DEFAULT_PATH
                DOC "LIBSVM debug library.")

        # Release-only distributions ship no debug build: fall back to the
        # release library so that a Release configure succeeds.
        if (NOT LIBSVM_LIBRARY_DEBUG)
            set(LIBSVM_LIBRARY_DEBUG ${LIBSVM_LIBRARY}
                    CACHE FILEPATH "LIBSVM debug library." FORCE)
        endif ()

        # ----- Find the LIBSVM runtime DLL on Windows ---------------------- #
        find_file(LIBSVM_DLL
                NAMES svm.dll libsvm.dll
                PATHS
                ${LIBSVM_ROOT}/bin
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin
                $ENV{LIBRARY_BIN}
                NO_DEFAULT_PATH
                DOC "LIBSVM runtime DLL.")

        find_file(LIBSVM_DLL_DEBUG
                NAMES svm.dll libsvm.dll
                PATHS
                ${LIBSVM_ROOT}/debug/bin
                ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin
                NO_DEFAULT_PATH
                DOC "LIBSVM debug runtime DLL.")

        if (NOT LIBSVM_DLL_DEBUG AND LIBSVM_DLL)
            set(LIBSVM_DLL_DEBUG ${LIBSVM_DLL}
                    CACHE FILEPATH "LIBSVM debug runtime DLL." FORCE)
        endif ()
    endif ()

    # ----- Parse the version ----------------------------------------------- #
    # LIBSVM_VERSION is an integer, e.g. 324 for 3.24
    if (LIBSVM_INCLUDE_DIR)
        file(STRINGS
                "${LIBSVM_INCLUDE_DIR}/libsvm/svm.h"
                _LIBSVM_version_line REGEX "#define LIBSVM_VERSION")

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
    if (WIN32)
        # The debug library/DLL are optional (they fall back to the release
        # ones above), so they are deliberately kept out of REQUIRED_VARS.
        find_package_handle_standard_args(
                LIBSVM
                REQUIRED_VARS LIBSVM_LIBRARY LIBSVM_DLL LIBSVM_INCLUDE_DIR
                VERSION_VAR LIBSVM_VERSION)
    else ()
        find_package_handle_standard_args(
                LIBSVM
                REQUIRED_VARS LIBSVM_LIBRARY LIBSVM_INCLUDE_DIR
                VERSION_VAR LIBSVM_VERSION)
    endif ()
endif ()

# ----- Export the target --------------------------------------------------- #
if (LIBSVM_FOUND)
    set(LIBSVM_INCLUDE_DIRS ${LIBSVM_INCLUDE_DIR})
    set(LIBSVM_LIBRARIES ${LIBSVM_LIBRARY})

    if (NOT TARGET LIBSVM::LIBSVM)
        if (WIN32)
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
