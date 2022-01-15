# For macOS v10 or above

DEBUGFLAG = -g
DEFINES = -DNOMATLAB 
CC=gcc
CFLAGS= -c
FFLAGS= -fPIC 
LIBFLAGS= -shared
SHLIBEXT= dylib

all: libproxtv

libproxtv: wrapper.cpp LPopt.cpp utils.cpp
	    $(CC) $(FFLAGS) $(DEFINES) $(LIBFLAGS) -o libproxtv.$(SHLIBEXT) wrapper.cpp LPopt.cpp utils.cpp
		mv libproxtv.$(SHLIBEXT) ..

debug:
	    $(CC) $(FFLAGS) $(DEFINES) $(LIBFLAGS) $(DEBUGFLAG) -o libproxtv.$(SHLIBEXT) wrapper.cpp LPopt.cpp utils.cpp

