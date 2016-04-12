include make.inc

CC=gcc
CXX=g++ -std=c++0x
MPICXX=mpicxx -std=c++0x

ARCH_FLAGS=-m64 -D _D2_SINGLE
CFLAGS=-Wextra -Wall -pedantic-errors -O3 $(ARCH_FLAGS)
LDFLAGS=$(ARCH_FLAGS)
DEFINES=
INCLUDES=-I$(RABIT)/include -I$(MOSEK)/h
LIBRARIES=-L./lib -ld2suite -L$(RABIT)/lib -lrabit -L$(MOSEK)/bin -Wl,-rpath,$(MOSEK)/bin -lmosek64 -lpthread $(BLAS_LIB)

OS=$(shell uname)

C_SOURCE_FILES=\
	src/common/blas_like32.c\
	src/common/blas_like64.c

CPP_SOURCE_FILES=\
	src/common/solver_mosek.cpp

CPP_SOURCE_WITH_MAIN=\
	src/test/test_euclidean.cpp\
	src/test/test_20newsgroups_io.cpp\
	src/test/test_mnist.cpp

RABIT_SOURCE_WITH_MAIN=\
	src/test/test_20newsgroups_io_rabit.cpp


ALL_OBJECTS=\
	$(patsubst %.c, %.o, $(C_SOURCE_FILES))\
	$(patsubst %.cpp, %.o, $(CPP_SOURCE_FILES))

EXECUTABLES=\
	$(patsubst %_rabit.cpp, %.rabit, $(RABIT_SOURCE_WITH_MAIN))\
	$(patsubst %.cpp, %, $(CPP_SOURCE_WITH_MAIN))

TESTS=\
	$(patsubst %.cpp, %.test, $(CPP_SOURCE_WITH_MAIN))\
	$(patsubst %_rabit.cpp, %.rabit_test, $(RABIT_SOURCE_WITH_MAIN))

DEPENDENCY_FILES=\
	$(patsubst %.o, %.d, $(ALL_OBJECTS))\
	$(patsubst %, %.d, $(EXECUTABLES))

LIB=\
	lib/libd2suite.a


all: $(EXECUTABLES) $(LIB)

lib/libd2suite.a: $(ALL_OBJECTS)
	ar cr $@ $+

%.o: %.cpp Makefile
	@# Make dependecy file
	$(CXX) -MM -MT $@ -MF $(patsubst %.cpp,%.d,$<) $(CFLAGS) $(DEFINES) $(INCLUDES) $<
	@# Compile
	$(CXX) $(CFLAGS) $(DEFINES) $(INCLUDES) -c -o $@ $<

ifeq ($(OS), Darwin)
%.rabit: %_rabit.cpp Makefile $(LIB)
	$(CXX) -MM -MT $@ -MF $(patsubst %_rabit.cpp,%.rabit.d,$<) $(CFLAGS) $(DEFINES) $(INCLUDES) $<
	$(CXX) $(LDFLAGS) $(DEFINES) $(INCLUDES) -o $@ $< $(LIBRARIES)
	install_name_tool -change @loader_path/libmosek64.$(MOSEK_VERSION).dylib $(MOSEK)/bin/libmosek64.$(MOSEK_VERSION).dylib $@
else
	$(CXX) -MM -MT $@ -MF $(patsubst %_rabit.cpp,%.rabit.d,$<) $(CFLAGS) $(DEFINES) $(INCLUDES) $<
	$(CXX) $(LDFLAGS) $(DEFINES) $(INCLUDES) -o $@ $< $(LIBRARIES)
endif

ifeq ($(OS), Darwin)
%: %.cpp Makefile $(LIB)
	$(CXX) -MM -MT $@ -MF $(patsubst %.cpp,%.d,$<) $(CFLAGS) $(DEFINES) $(INCLUDES) $<
	$(CXX) $(LDFLAGS) $(DEFINES) $(INCLUDES) -o $@ $< $(LIBRARIES)
	install_name_tool -change @loader_path/libmosek64.$(MOSEK_VERSION).dylib $(MOSEK)/bin/libmosek64.$(MOSEK_VERSION).dylib $@
else
	$(CXX) -MM -MT $@ -MF $(patsubst %.cpp,%.d,$<) $(CFLAGS) $(DEFINES) $(INCLUDES) $<
	$(CXX) $(LDFLAGS) $(DEFINES) $(INCLUDES) -o $@ $< $(LIBRARIES)
endif

-include $(DEPENDENCY_FILES)

.PHONY: clean test

clean: 
	@rm $(EXECUTABLES) $(LIB)
	@for pattern in '*.o' '*.d'; do \
		find . -name "$$pattern" | xargs rm; \
	done

test: $(TESTS) all

%.test: %
	$< > /dev/null

%.rabit_test: %.rabit
	../rabit/tracker/rabit_demo.py -n 4 $<
