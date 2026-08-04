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
#   Output: $(SVMBkOBJ)  = the final object(s) / library                     #
#           $(SVMBkH)    = the .h files to include                           #
#           $(SVMBkINC)  = the -I$( source directory )                       #
#                                                                            #
#                             Antonio Frangioni                              #
#                         Dipartimento di Informatica                        #
#                             Universita' di Pisa                            #
#                                                                            #
##############################################################################

# macros to be exported - - - - - - - - - - - - - - - - - - - - - - - - - - -

SVMBkOBJ = $(SVMBkSDR)/obj/SVMBlock.o \
	$(SVMBkSDR)/obj/SVCBlock.o \
	$(SVMBkSDR)/obj/SVRBlock.o \
	$(SVMBkSDR)/obj/SMOSolver.o

SVMBkINC = -I$(SVMBkSDR)/include

SVMBkH   = $(SVMBkSDR)/include/SVMBlock.h \
	$(SVMBkSDR)/include/SVCBlock.h \
	$(SVMBkSDR)/include/SVRBlock.h \
	$(SVMBkSDR)/include/SMOSolver.h

# clean - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

clean::
	rm -f $(SVMBkOBJ) $(SVMBkSDR)/*~

# dependencies: every .o from its .cpp + every recursively included .h- - - -

$(SVMBkSDR)/obj/SVMBlock.o: $(SVMBkSDR)/src/SVMBlock.cpp \
	$(SVMBkSDR)/include/SVMBlock.h $(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/SVMBlock.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

$(SVMBkSDR)/obj/SVCBlock.o: $(SVMBkSDR)/src/SVCBlock.cpp \
	$(SVMBkSDR)/include/SVCBlock.h $(SVMBkSDR)/include/SVMBlock.h \
	$(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/SVCBlock.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

$(SVMBkSDR)/obj/SVRBlock.o: $(SVMBkSDR)/src/SVRBlock.cpp \
	$(SVMBkSDR)/include/SVRBlock.h $(SVMBkSDR)/include/SVMBlock.h \
	$(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/SVRBlock.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

$(SVMBkSDR)/obj/SMOSolver.o: $(SVMBkSDR)/src/SMOSolver.cpp \
	$(SVMBkSDR)/include/SMOSolver.h $(SVMBkSDR)/include/SVMBlock.h \
	$(SMS++H) $(SMS++OBJ)
	$(CC) -c $(SVMBkSDR)/src/SMOSolver.cpp -o $@ \
	$(SVMBkINC) $(SMS++INC) $(SW)

########################## End of makefile ###################################
