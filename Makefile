LIBS = opencv_core opencv_highgui opencv_imgproc
LIBPATHS = /usr/share

LINKFLAGS = $(addprefix -l,$(LIBS))
LIBPATHFLAGS = $(addprefix -L,$(LIBPATHS))

all: main.cpp opencvUtil.o uf.o
	g++ -o main $^ $(LINKFLAGS) $(LIBPATHFLAGS)

opencvUtil.o: opencvUtil.hpp opencvUtil.cpp
	g++ -c -o opencvUtil.o opencvUtil.cpp $(LINKFLAGS) $(LIBPATHFLAGS)

uf.o: uf.hpp uf.cpp
	g++ -c -o uf.o uf.cpp $(LINKFLAGS) $(LIBPATHFLAGS)

clean:
	rm main *.o