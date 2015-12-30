LIBS = opencv_core opencv_highgui opencv_imgproc
LIBPATHS = /usr/share

LINKFLAGS = $(addprefix -l,$(LIBS))
LIBPATHFLAGS = $(addprefix -L,$(LIBPATHS))

CC = mpiCC

default: main2

all: main main2 contourify

main: main.cpp
	$(CC) -o main $^ $(LINKFLAGS) $(LIBPATHFLAGS)

main2: main2.cpp
	$(CC) -o main2 $^ $(LINKFLAGS) $(LIBPATHFLAGS)

contourify: contourify.cpp
	$(CC) -o contourify $^ $(LINKFLAGS) $(LIBPATHFLAGS)

opencvUtil.o: opencvUtil.hpp opencvUtil.cpp
	$(CC) -c -o opencvUtil.o opencvUtil.cpp $(LINKFLAGS) $(LIBPATHFLAGS)

uf.o: uf.hpp uf.cpp
	$(CC) -c -o uf.o uf.cpp $(LINKFLAGS) $(LIBPATHFLAGS)

clean:
	rm main main2 contourify *.o