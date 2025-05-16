build:
	rm -f cs
	mpic++ -O2 -Wall main.cpp cs.cpp bitmap.cpp -Wno-unused-but-set-variable -Wno-unused-function -Wno-write-strings -Wno-unused-result -o cs

run:
	mpiexec -n 4 cs
