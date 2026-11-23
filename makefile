##############################################################################
################################ makefile ####################################
##############################################################################
#                                                                            #
#   makefile of SVMBlock                                                     #
#                                                                            #
#   Note that $(SMS++INC) is assumed to include any -I directive             #
#   corresponding to external libraries needed by SMS++, at least to the     #
#   extent in which they are needed by the parts of SMS++ used by            #
#   SVMBlock.                                                                #
#                                                                            #
#   Input:  $(CC)        = compiler command                                  #
#           $(SW)        = compiler options                                  #
#           $(SMS++INC)  = the -I$( core SMS++ directory )                   #
#           $(SMS++OBJ)  = the libSMS++ library itself                       #
#           $(SVMBkSDR)  = the directory where the source is                 #
#                                                                            #
#   LIBSVM and LIBLINEAR are found through $(LIBSVM_ROOT) and                #
#   $(LIBLINEAR_ROOT), set in extlib/makefile-default-paths-* (overridable   #
#   via extlib/makefile-paths): LIBSVMSolver is built if LIBSVM is there,    #
#   LIBLINEARSolver if LIBLINEAR is, and the rest always.                    #
#                                                                            #
#   Output: $(SVMBkOBJ)  = the final object(s) / library                     #
#           $(SVMBkH)    = the .h files to include                           #
#           $(SVMBkINC)  = the -I$( source directory )                       #
#                                                                            #
#                             Antonio Frangioni                              #
#                         Dipartimento di Informatica                        #
#                             Universita' di Pisa                            #
#                                                                            #
##############################################################################

include $(SVMBkSDR)/../extlib/makefile-libLIBSVM
include $(SVMBkSDR)/../extlib/makefile-libLIBLINEAR

# whether LIBSVM and LIBLINEAR are there, by their headers
SVMBkLIBSVM := $(wildcard $(libLIBSVMBSCDIR)/include/libsvm/svm.h \
	$(libLIBSVMBSCDIR)/include/svm.h)
SVMBkLIBLINEAR := $(wildcard $(libLIBLINEARBSCDIR)/include/linear.h \
	$(libLIBLINEARBSCDIR)/include/liblinear/linear.h)

# macros to be exported - - - - - - - - - - - - - - - - - - - - - - - - - - -

SVMBkOBJ = $(SVMBkSDR)/obj/SVMBlock.o \
	$(SVMBkSDR)/obj/SVCBlock.o \
	$(SVMBkSDR)/obj/SVRBlock.o \
	$(SVMBkSDR)/obj/SMOSolver.o

SVMBkINC = -I$(SVMBkSDR)/include

SVMBkH   = $(SVMBkSDR)/include/SVMBlock.h \
	$(SVMBkSDR)/include/SVCBlock.h \
	$(SVMBkSDR)/include/SVRBlock.h \
	$(SVMBkSDR)/include/SMOSolver.h \
	$(SVMBkSDR)/include/SVMBlockArch.h

SVMBkLIB =

ifneq ($(SVMBkLIBSVM),)
SVMBkOBJ += $(SVMBkSDR)/obj/LIBSVMSolver.o
SVMBkH += $(SVMBkSDR)/include/LIBSVMSolver.h
SVMBkINC += $(libLIBSVMINC)
SVMBkLIB += $(libLIBSVMLIB)
endif

ifneq ($(SVMBkLIBLINEAR),)
SVMBkOBJ += $(SVMBkSDR)/obj/LIBLINEARSolver.o
SVMBkH += $(SVMBkSDR)/include/LIBLINEARSolver.h
SVMBkINC += $(libLIBLINEARINC)
SVMBkLIB += $(libLIBLINEARLIB)
endif

# the C++20 module smspp.svmblock - - - - - - - - - - - - - - - - - - - - - -
# [see makefile-mod of SMS++]

$(eval $(call SMSppMODrule,SVMBk,$(SVMBkSDR),SVMBlock,smspp.svmblock,\
	SMS++))

# clean - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

clean::
	rm -f $(SVMBkOBJ) $(SVMBkSDR)/*~ $(SVMBkSDR)/tools/svm_arch \
	$(SVMBkSDR)/include/SVMBlockArch.h

# the properties of the machine, measured once at the first compilation and
# written into a header that is not part of the sources - - - - - - - - - - -

# the memory of the machine by hwloc, where a program that loads its topology
# compiles and links, unless SVMBkHWLOC is set to 0
SVMBkHWLOC ?= $(shell printf '\043include <hwloc.h>\nint main(){\
	hwloc_topology_t t;return hwloc_topology_init(&t);}\n' | \
	$(CC) -x c++ - -o /dev/null -lhwloc 2>/dev/null && echo 1)

$(SVMBkSDR)/include/SVMBlockArch.h: $(SVMBkSDR)/tools/svm_arch.cpp
	$(CC) $(SVMBkSDR)/tools/svm_arch.cpp -o $(SVMBkSDR)/tools/svm_arch \
	$(SW) -pthread $(if $(filter 1,$(SVMBkHWLOC)),\
	-DSVMBlock_USE_HWLOC -lhwloc)
	$(SVMBkSDR)/tools/svm_arch $(SVMBkSDR)/include

# dependencies: every .o from its .cpp + every recursively included .h- - - -

$(SVMBkSDR)/obj/SVMBlock.o: $(SVMBkSDR)/src/SVMBlock.cpp \
	$(SVMBkSDR)/include/SVMBlock.h $(SVMBkSDR)/include/SVMBlockArch.h \
	$(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/SVMBlock.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

$(SVMBkSDR)/obj/SVCBlock.o: $(SVMBkSDR)/src/SVCBlock.cpp \
	$(SVMBkSDR)/include/SVCBlock.h $(SVMBkSDR)/include/SVMBlock.h \
	$(SVMBkSDR)/include/SVMBlockArch.h $(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/SVCBlock.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

$(SVMBkSDR)/obj/SVRBlock.o: $(SVMBkSDR)/src/SVRBlock.cpp \
	$(SVMBkSDR)/include/SVRBlock.h $(SVMBkSDR)/include/SVMBlock.h \
	$(SVMBkSDR)/include/SVMBlockArch.h $(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/SVRBlock.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

$(SVMBkSDR)/obj/SMOSolver.o: $(SVMBkSDR)/src/SMOSolver.cpp \
	$(SVMBkSDR)/include/SMOSolver.h $(SVMBkSDR)/include/SVMBlock.h \
	$(SVMBkSDR)/include/SVMBlockArch.h $(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/SMOSolver.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

$(SVMBkSDR)/obj/LIBSVMSolver.o: $(SVMBkSDR)/src/LIBSVMSolver.cpp \
	$(SVMBkSDR)/include/LIBSVMSolver.h $(SVMBkSDR)/include/SVCBlock.h \
	$(SVMBkSDR)/include/SVRBlock.h $(SVMBkSDR)/include/SVMBlock.h \
	$(SVMBkSDR)/include/SVMBlockArch.h $(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/LIBSVMSolver.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

$(SVMBkSDR)/obj/LIBLINEARSolver.o: $(SVMBkSDR)/src/LIBLINEARSolver.cpp \
	$(SVMBkSDR)/include/LIBLINEARSolver.h $(SVMBkSDR)/include/SVCBlock.h \
	$(SVMBkSDR)/include/SVRBlock.h $(SVMBkSDR)/include/SVMBlock.h \
	$(SVMBkSDR)/include/SVMBlockArch.h $(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/LIBLINEARSolver.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

########################## End of makefile ###################################
