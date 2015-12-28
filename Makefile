LIBS = opencv_core opencv_highgui opencv_imgproc
LIBPATHS = /usr/share

LINKFLAGS = $(addprefix -l,$(LIBS))
LIBPATHFLAGS = $(addprefix -L,$(LIBPATHS))

CC = mpiCC

all: main.cpp opencvUtil.o uf.o
	$(CC) -o main $^ $(LINKFLAGS) $(LIBPATHFLAGS)

opencvUtil.o: opencvUtil.hpp opencvUtil.cpp
	$(CC) -c -o opencvUtil.o opencvUtil.cpp $(LINKFLAGS) $(LIBPATHFLAGS)

uf.o: uf.hpp uf.cpp
	$(CC) -c -o uf.o uf.cpp $(LINKFLAGS) $(LIBPATHFLAGS)

main2: main2.cpp
	$(CC) -o main2 $^ $(LINKFLAGS) $(LIBPATHFLAGS)

clean:
	rm main main2 *.o