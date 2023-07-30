all: build

build: sf

sf.o: sf.c error_util.h
	g++ -Wall -c -o sf.o sf.c -I /usr/local/cuda/include
sf: sf.o error_util.h
	/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -o sf sf.o -L/usr/local/cuda/lib64 -lcudnn

clean:
	rm -f sf.o sf

run: sf
	./sf

check_memory_leak: sf
	/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check=full sf
