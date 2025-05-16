/* mm.cpp */

//
// Matrix multiplication implementation, computing C=A*B where A and B
// are NxN matrices. The resulting matrix C is therefore NxN.
// 
// MM is split across the main process and the N-1 worker processes.
//
#include <iostream>
#include <string>
#include <omp.h>
#include <mpi.h>

#include "alloc2D.h"

using namespace std;


//
// main MPI process:
//
double** main_process(double** A, double** B, int N, int numProcs)
{
  cout << "main starting..." << endl;
  cout.flush();
  
  double** C = New2dMatrix<double>(N, N);

  //
  // Initialize target matrix in prep for summing:
  //
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      C[i][j] = 0.0;

  //
  // For every row i of A and column j of B:
  //
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      for (int k = 0; k < N; k++)
      {
        C[i][j] += (A[i][k] * B[k][j]);
      }
    }
  }

  //
  // return pointer to result matrix:
  //
  return C;
}


//
// worker MPI process(es):
//
void worker_process(int myRank, int numProcs)
{
  cout << "worker " << myRank << " starting..." << endl;
  cout.flush();
  
}
