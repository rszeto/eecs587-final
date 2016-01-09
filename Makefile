LIBS = opencv_core opencv_highgui opencv_imgproc
LIBPATHS = /usr/share

LINKFLAGS = $(addprefix -l,$(LIBS))
LIBPATHFLAGS = $(addprefix -L,$(LIBPATHS))

CC = mpiCC

all: main main2 main3 main4 main5 main6 contourify

main: main.cpp util.o
	$(CC) -o main $^ $(LINKFLAGS) $(LIBPATHFLAGS)

main2: main2.cpp util.o
	$(CC) -o main2 $^ $(LINKFLAGS) $(LIBPATHFLAGS)

main3: main3.cpp util.o
	$(CC) -o main3 $^ $(LINKFLAGS) $(LIBPATHFLAGS)

main4: main4.cpp util.o
	$(CC) -o main4 $^ $(LINKFLAGS) $(LIBPATHFLAGS)

main5: main5.cpp util.o
	$(CC) -o main5 $^ $(LINKFLAGS) $(LIBPATHFLAGS)

main6: main6.cpp util.o
	$(CC) -o main6 $^ $(LINKFLAGS) $(LIBPATHFLAGS)

contourify: contourify.cpp
	$(CC) -o contourify $^ $(LINKFLAGS) $(LIBPATHFLAGS)

# opencvUtil.o: opencvUtil.hpp opencvUtil.cpp
# 	$(CC) -c -o opencvUtil.o opencvUtil.cpp $(LINKFLAGS) $(LIBPATHFLAGS)

util.o: util.hpp util.cpp
	$(CC) -c -o util.o util.cpp $(LINKFLAGS) $(LIBPATHFLAGS)

# uf.o: uf.hpp uf.cpp
# 	$(CC) -c -o uf.o uf.cpp $(LINKFLAGS) $(LIBPATHFLAGS)

clean:
	rm main main2 main3 main4 main5 contourify *.o