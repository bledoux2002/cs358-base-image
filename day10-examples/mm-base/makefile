build:
	rm -f mm
	mpic++ -O2 -Wall main.cpp mm.cpp -fopenmp -o mm

run:
	mpiexec -n 4 mm
