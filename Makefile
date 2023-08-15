all: build

build: sf

softmax.o: softmax.h softmax.c
	gcc -Wall -c -o softmax.o softmax.c
softmax: softmax.o
	gcc -Wall -c -o softmax softmax.o
sf.o: sf.c error_util.h
	g++ -Wall -c -o sf.o sf.c -I /usr/local/cuda/include
sf: softmax.o sf.o
	/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -o sf softmax.o sf.o -L/usr/local/cuda/lib64 -lcudnn

clean:
	rm -f softmax.o sf.o sf

run: sf
	./sf

check_memory_leak: sf
	/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check=full sf
