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

  int myRank = 0;

  //
  // Send A and B to workers:
  //
  for (int w = 1; w < numProcs; w++) {

    int dest = w;
    int count = 1;
    int tag = 0;

    //
    // first we send the size of the matrix:
    //
    MPI_Send(&N, count, MPI_INT, dest, tag, MPI_COMM_WORLD);

    //
    // now we send A:
    //
    count = N * N;

    MPI_Send(A[0], count, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

    //
    // and finally B:
    //
    MPI_Send(B[0], count, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
  }

  //
  // instead of doing nothing, the main process also 
  // performs it's own share of MM:
  //
  double** C = New2dMatrix<double>(N, N);

  //
  // For every row i of A and column j of B:
  //
  int chunkSize = N / numProcs;
  int startRow = chunkSize * myRank;
  int endRow = startRow + chunkSize;

  //
  // Initialize target matrix in prep for summing:
  //
  for (int i = startRow; i < endRow; i++)
    for (int j = 0; j < N; j++)
      C[i][j] = 0.0;

  //
  // MM:
  //
  for (int i = startRow; i < endRow; i++)
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
  // now receive the remaining results from the workers:
  //
  for (int w = 1; w < numProcs; w++) {

    int src = w;
    int count = chunkSize * N;
    int tag = 0;

    //
    // receive chunk of results from worker:
    //
    int row = chunkSize * w;

    MPI_Recv(C[row], count, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  //
  // done, return pointer to result matrix:
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

  //
  // first we receive size of matrix N from main:
  //
  // MPI_Status status;

  int src = 0;   // receive from main
  int count = 1;  
  int tag = 0;

  int N;

  MPI_Recv(&N, count, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  //
  // now we allocate space for matrices A and B, and 
  // receive them from main:
  //
  double** A = New2dMatrix<double>(N, N);
  double** B = New2dMatrix<double>(N, N);

  //
  // NOTE: we know that underlying "matrix" is single
  // 1D array, pointed to by A[0] and B[0], of length
  // N*N.
  //
  count = N * N;

  MPI_Recv(A[0], count, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Recv(B[0], count, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  //
  // we have A and B, now we perform MM for our block of rows:
  //
  double** C = New2dMatrix<double>(N, N);

  //
  // For every row i of A and column j of B:
  //
  int chunkSize = N / numProcs;
  int startRow = chunkSize * myRank;
  int endRow = startRow + chunkSize;

  //
  // Initialize target matrix in prep for summing:
  //
  for (int i = startRow; i < endRow; i++)
    for (int j = 0; j < N; j++)
      C[i][j] = 0.0;

  //
  // MM:
  //
  for (int i = startRow; i < endRow; i++)
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
  // done, send our results back to main:
  //
  int dest = 0;
  count = chunkSize * N;
  tag = 0;

  MPI_Send(C[startRow], count, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

  //
  // cleanup:
  //
  Delete2dMatrix(A);
  Delete2dMatrix(B);
  Delete2dMatrix(C);
}
