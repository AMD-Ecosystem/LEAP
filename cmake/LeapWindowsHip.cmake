# Included before project() for LEAP_GPU=AMD on Windows.
#
# CMake's Windows platform modules cannot pair ROCm's clang++ (the HIP compiler)
# with an MSVC-style C++ compiler: next to cl.exe the HIP compiler is rejected,
# and next to clang-cl it is given cl-style flags (/DWIN32, /Zi, /machine:x64)
# that clang++ does not accept. So the C++ sources are compiled with the same
# clang++, found here the way CMake finds it on Linux (HIPCXX, then hipconfig).
if(CMAKE_VERSION VERSION_LESS 3.29)
    message(FATAL_ERROR "Building LEAP for AMD GPUs on Windows requires CMake 3.29 or newer.")
endif()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    message(FATAL_ERROR "The \"${CMAKE_GENERATOR}\" generator does not support HIP; configure with -G Ninja.")
endif()

# pip's isolated build sets PYTHONPATH so that Python processes do not see the
# environment's own packages. A pip-installed ROCm SDK's hipconfig, which this
# file, CMake and hipify run during the configure, needs them.
unset(ENV{PYTHONPATH})
unset(ENV{PYTHONNOUSERSITE})

if(NOT CMAKE_HIP_COMPILER AND NOT "$ENV{HIPCXX}" STREQUAL "")
    set(CMAKE_HIP_COMPILER "$ENV{HIPCXX}")
endif()

if(NOT CMAKE_HIP_COMPILER)
    # A pip-installed ROCm SDK puts hipconfig next to the Python interpreter,
    # which scikit-build-core passes in even when the environment is not activated.
    set(_leap_hipconfig_hints)
    foreach(_leap_python IN ITEMS "${Python_EXECUTABLE}" "${Python3_EXECUTABLE}")
        if(_leap_python)
            get_filename_component(_leap_python_dir "${_leap_python}" DIRECTORY)
            list(APPEND _leap_hipconfig_hints "${_leap_python_dir}")
        endif()
    endforeach()
    find_program(LEAP_HIPCONFIG hipconfig
        HINTS ENV HIP_PATH ENV ROCM_PATH ${_leap_hipconfig_hints}
        PATH_SUFFIXES bin)
    set(_leap_hip_clang_dir)
    if(LEAP_HIPCONFIG)
        execute_process(COMMAND "${LEAP_HIPCONFIG}" --hipclangpath
            OUTPUT_VARIABLE _leap_hip_clang_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
    find_program(LEAP_HIP_CLANGXX clang++
        HINTS "${_leap_hip_clang_dir}" ENV HIP_PATH ENV ROCM_PATH
        PATH_SUFFIXES bin lib/llvm/bin
        NO_DEFAULT_PATH)
    if(LEAP_HIP_CLANGXX)
        set(CMAKE_HIP_COMPILER "${LEAP_HIP_CLANGXX}" CACHE FILEPATH "HIP compiler")
    endif()
endif()

if(CMAKE_HIP_COMPILER AND NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${CMAKE_HIP_COMPILER}")
endif()

# CMake enables RC for clang++ on Windows. Outside a Visual Studio developer
# prompt rc.exe is not on PATH, so take it from the newest Windows SDK.
if(NOT DEFINED CMAKE_RC_COMPILER AND "$ENV{RC}" STREQUAL "")
    find_program(LEAP_RC_COMPILER rc)
    if(NOT LEAP_RC_COMPILER)
        cmake_host_system_information(RESULT _leap_kits_root QUERY WINDOWS_REGISTRY
            "HKLM/SOFTWARE/Microsoft/Windows Kits/Installed Roots" VALUE KitsRoot10)
        file(GLOB _leap_rc_compilers "${_leap_kits_root}/bin/10.*/x64/rc.exe")
        if(_leap_rc_compilers)
            list(SORT _leap_rc_compilers COMPARE NATURAL ORDER DESCENDING)
            list(GET _leap_rc_compilers 0 _leap_rc_compiler)
            set(CMAKE_RC_COMPILER "${_leap_rc_compiler}" CACHE FILEPATH "RC compiler")
        endif()
    endif()
endif()

# FindOpenMP locates libomp.lib for clang++ on Windows only through the LIB
# variable of a Visual Studio developer prompt. Every source here is compiled
# as HIP and linked with clang's own -fopenmp, which adds libomp itself.
if(NOT DEFINED OpenMP_CXX_FLAGS AND NOT DEFINED OpenMP_CXX_LIB_NAMES)
    set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "CXX compiler flags for OpenMP parallelization")
    set(OpenMP_CXX_LIB_NAMES "" CACHE STRING "CXX compiler libraries for OpenMP parallelization")
endif()

# CMake picks the build machine's GPUs with rocm_agent_enumerator, which ROCm
# does not ship on Windows; amdgpu-arch, next to clang++, reports the same.
if(CMAKE_HIP_COMPILER AND NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    get_filename_component(_leap_hip_clang_dir "${CMAKE_HIP_COMPILER}" DIRECTORY)
    find_program(LEAP_AMDGPU_ARCH amdgpu-arch HINTS "${_leap_hip_clang_dir}" NO_DEFAULT_PATH)
    if(LEAP_AMDGPU_ARCH)
        execute_process(COMMAND "${LEAP_AMDGPU_ARCH}"
            OUTPUT_VARIABLE _leap_hip_archs
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _leap_amdgpu_arch_result
            ERROR_QUIET)
        string(REGEX REPLACE "[\r\n]+" ";" _leap_hip_archs "${_leap_hip_archs}")
        if(_leap_amdgpu_arch_result EQUAL 0 AND _leap_hip_archs)
            list(REMOVE_DUPLICATES _leap_hip_archs)
            set(CMAKE_HIP_ARCHITECTURES "${_leap_hip_archs}" CACHE STRING "HIP architectures")
        endif()
    endif()
endif()
