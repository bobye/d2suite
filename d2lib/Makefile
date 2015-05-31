CC=gcc
CXX=g++

ARCH_FLAGS=-m64
CFLAGS=-Wextra -Wall -pedantic-errors -O3 $(ARCH_FLAGS)
LDFLAGS=$(ARCH_FLAGS)
DEFINES=
INCLUDES=-I../rabit/include
LIBRARIES=-L../rabit/lib -lrabit

CPP_SOURCE_FILES=\


CPP_SOURCE_WITH_MAIN=\
	src/test/test_euclidean_io.cpp\
	src/test/test_20newsgroups_io.cpp


ALL_OBJECTS=\
	$(patsubst %.cpp, %.o, $(CPP_SOURCE_FILES))

EXECUTABLES=\
	$(patsubst %.cpp, %, $(CPP_SOURCE_WITH_MAIN))

TESTS=\
	$(patsubst %.cpp, %.test, $(CPP_SOURCE_WITH_MAIN))

DEPENDENCY_FILES=\
	$(patsubst %.o, %.d, $(ALL_OBJECTS))\
	$(patsubst %, %.d, $(EXECUTABLES))


all: $(EXECUTABLES)

%.o: %.cpp Makefile
	@# Make dependecy file
	$(CXX) -MM -MT $@ -MF $(patsubst %.cpp,%.d,$<) $(CFLAGS) $(DEFINES) $(INCLUDES) $<
	@# Compile
	$(CXX) $(CFLAGS) $(DEFINES) $(INCLUDES) -c -o $@ $<

%: %.cpp Makefile
	$(CXX) -MM -MT $@ -MF $(patsubst %.cpp,%.d,$<) $(CFLAGS) $(DEFINES) $(INCLUDES) $<
	$(CXX) $(LDFLAGS) $(DEFINES) $(INCLUDES) -o $@ $< $(LIBRARIES)


-include $(DEPENDENCY_FILES)

.PHONY: clean test

clean: 
	@rm $(EXECUTABLES)
	@for pattern in '*.o' '*.d'; do \
		find . -name "$$pattern" | xargs rm; \
	done

test: $(TESTS) all

%.test: %
	$< > /dev/null

