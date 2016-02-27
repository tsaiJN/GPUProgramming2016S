all:
	nvcc -o main -std=c++11 -I ../utils/ main.cu

run:
	nvcc -o main -std=c++11 -I ../utils/ main.cu
	./main test.txt

clean:
	rm -f main
