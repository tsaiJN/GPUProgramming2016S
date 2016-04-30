CXXFLAGS += -std=c++11 -D_MWAITXINTRIN_H_INCLUDED
OBJECTS = pgm.o lab3.o

all: pgm.o lab3.o
	nvcc -o main -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -I ../utils/ main.cu $(OBJECTS)

lab3.o:
	nvcc -c -o lab3.o -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -I ../utils/ lab3.cu # -c option: not to run the linker


run:
	./main img_background.ppm img_target.ppm img_mask.pgm 130 600 output.ppm 50000
	imgcat output.ppm

clean:
	rm -f $(OBJECTS) main output.ppm

