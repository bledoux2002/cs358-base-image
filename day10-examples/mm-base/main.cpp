/* main.cpp */

//
// MPI-based Matrix Multiplication app
//
// Uses MPI to implement basic matrix multiplication
//
// Usage:
//   mpiexec -n numProcs mm [-?] [-n MatrixSize]
//
// Author:
//   Prof. Joe Hummel
//   Northwestern University
//

#include <iostream>
#include <string>
#include <cmath>
#include <cstring>
#include <chrono>
#include <mpi.h>

#include "alloc2D.h"

using namespace std;


//
// Globals:
//
static int _matrixSize;

//
// Function prototypes:
//
void CreateAndFillMatrices(int N, double** &A, double** &B, double &TL, double &TR, double &BL, double &BR);
void CheckResults(int N, double** C, double TL, double TR, double BL, double BR);
void ProcessCmdLineArgs(int argc, char* argv[]);

//
// Defined in mm.cpp:
//
double** main_process(double** A, double** B, int _matrixSize, int numProcs);
void     worker_process(int myRank, int numProcs);


//
// main:
//
int main(int argc, char *argv[])
{
	int myRank, numProcs;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	//
	// Run workers and main processes:
	//
	if (myRank > 0) {

		worker_process(myRank, numProcs);

	}
	else {

		//
		// Main process processes any command-line args, creates
		// the matrices, and then kicks off the matrix multiply:
		//
		_matrixSize = 2000;

		ProcessCmdLineArgs(argc, argv);

		cout << "** MPI-Based Matrix Multiply Application **" << endl;
		cout << endl;
		cout << "Matrix size: " << _matrixSize << "x" << _matrixSize << endl;
		cout << "# processes: " << numProcs << endl;
    cout << endl;

		//
		// Create and fill the matrices to multiply:
		//
		double** A, ** B, TL, TR, BL, BR;
		CreateAndFillMatrices(_matrixSize, A, B, TL, TR, BL, BR);

		//
		// Start clock and multiply:
		//
		auto start = chrono::high_resolution_clock::now();

		double** C = main_process(A, B, _matrixSize, numProcs);

		auto stop = chrono::high_resolution_clock::now();
		auto diff = stop - start;
		auto duration = chrono::duration_cast<chrono::milliseconds>(diff);

		//
		// Done, check results and output timing:
		//
		CheckResults(_matrixSize, C, TL, TR, BL, BR);

		cout << endl;
		cout << "** Done!  Time: " << duration.count() / 1000.0 << " secs" << endl;
		cout << "** Execution complete **" << endl;
		cout << endl;
		cout << "** Results are correct" << endl;
		cout << endl;

		Delete2dMatrix(A);
		Delete2dMatrix(B);
		Delete2dMatrix(C);

	}//main_process

	MPI_Finalize();
	return 0;
}


//
// CreateAndFillMatrices:  fills A and B with predefined values, and then set TL, TR, BL and BR
// to the expected top-left, top-right, bottom-left and bottom-right values after the multiply.
//
void CreateAndFillMatrices(int N, double** &A, double** &B, double &TL, double &TR, double &BL, double &BR)
{
	A = New2dMatrix<double>(N, N);
	B = New2dMatrix<double>(N, N);

	//
	// A looks like:  
	//   1  1  1  1  ...  1
	//   2  2  2  2  ...  2
	//   .  .  .  .  ...  .
	//   .  .  .  .  ...  .
	//   N  N  N  N  ...  N
	//
	for (int r = 0; r < N /*rows*/; r++)
		for (int c = 0; c < N /*cols*/; c++)
			A[r][c] = r + 1;

	//
	// B looks like:
	//   1  2  3  4  ...  N
	//   1  2  3  4  ...  N
	//   .  .  .  .  ...  .
	//   .  .  .  .  ...  .
	//   1  2  3  4  ...  N
	//
	for (int r = 0; r < N /*rows*/; r++)
		for (int c = 0; c < N /*cols*/; c++)
			B[r][c] = c + 1;

	//
	// expected values:
	//
	double dN = N;  // use double to overflow errors with large N:
 
	TL = dN;        // C[0,0] == Sum(1..1)
	TR = dN*dN;     // C[0,N-1] == Sum(N..N)
	BL = dN*dN;     // C[N-1, 0] == Sum(N..N)
	BR = dN*dN*dN;  // C[N-1, N-1] == SUM(N^2..N^2)
}


//
// Checks the results against some expected results:
//
void CheckResults(int N, double** C, double TL, double TR, double BL, double BR)
{ 
	bool b1 = ( fabs(C[0][0]     - TL) < 0.0000001 );
	bool b2 = ( fabs(C[0][N-1]   - TR) < 0.0000001 );
	bool b3 = ( fabs(C[N-1][0]   - BL) < 0.0000001 );
	bool b4 = ( fabs(C[N-1][N-1] - BR) < 0.0000001 );

	if (!b1 || !b2 || !b3 || !b4)
	{
		cout << "** ERROR: matrix multiply yielded incorrect results" << endl << endl;
		exit(0);
	}
}


//
// processCmdLineArgs:
//
void ProcessCmdLineArgs(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++)
	{

		if (strcmp(argv[i], "-?") == 0)  // help:
		{
			cout << "**Usage: mpiexec -n numProcs mm [-?] [-n MatrixSize]" << endl << endl;
			exit(0);
		}
		else if ((strcmp(argv[i], "-n") == 0) && (i+1 < argc))  // matrix size:
		{
			i++;
			_matrixSize = atoi(argv[i]);
		}
		else  // error: unknown arg
		{
			cout << "**Unknown argument: '" << argv[i] << "'" << endl;
			cout << "**Usage: mpiexec -n numProcs mm [-?] [-n MatrixSize]" << endl << endl;
			exit(0);
		}

	}//for
}
