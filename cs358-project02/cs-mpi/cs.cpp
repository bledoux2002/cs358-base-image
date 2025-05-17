/* cs.cpp */

//
// Performs the actual contrast stretching.
//
// Benjamin Ledoux
//
// Initial author:
//   Prof. Joe Hummel
//   Northwestern University
//

#include "app.h"
#include "matrix.h"
#include <mpi.h>


//
// Given 9 values (imagine a pixel and its 8 surrounding neighbors), returns the median
// value while also returning min and max values (via last 2 parameters)
//
uchar median(uchar a, uchar b, uchar c, uchar d, uchar e, 
						 uchar f, uchar g, uchar h, uchar i, uchar &min, uchar &max)
{
	uchar A[9];

	A[0] = a;  // capture 9 values:
	A[1] = b;
	A[2] = c;
	A[3] = d;
	A[4] = e;
	A[5] = f;
	A[6] = g;
	A[7] = h;
	A[8] = i;

	// selection sort:
	for(int i=0; i < 8; i++)
	{
		int min = i;

		// select smallest in subset:
		for (int j=i+1; j < 9; j++)
		{
			if (A[j] < A[min])
				min = j;
		}

		// swap:
		uchar temp = A[i];
		A[i] = A[min];
		A[min] = temp;
	}//for

	// return min, max & median of 9 elements:
	min = A[0];
	max = A[8];

	return A[4];
}


//
// Given a pixel value (a single RGB value) P and it's 8 neighbors, returns the new
// pixel value P' based on min & max values of neighbors.
//
uchar NewPixelValue(uchar UL, uchar UP, uchar UR,
										uchar L,  uchar P,  uchar R,
										uchar DL, uchar DW, uchar DR, 
	                  int stepby)
{
	uchar newp, med, min, max;
	double ratio;

	med = median(UL, UP, UR, L, P, R, DL, DW, DR, min, max);

	if (min == max)  // pixels are all the same:
		newp = min;
	else
	{
		// are neighbors darker than pixel, or lighter?
		ratio = (P - min) / (double) (max - min);

		if (ratio < 0.5)  // darker, so make pixel darker if possible:
		{
			if (P > stepby) // beware of underflow:
				newp = P - stepby;
			else  
				newp = 0;
		}
		else if (ratio > 0.5)  // lighter, so make pixel lighter if possible:
		{
			if (P < 255-stepby)  // beware of overflow:
				newp = P + stepby;
			else  
				newp = 255;
		}
		else  // neither, so leave pixel alone:
			newp = P;
	}

	// done:
	return newp;
}


//
// Copy over boundary rows to 2nd image matrix:
//
void copy_boundary(uchar** image2, uchar** image, int rows, int cols)
{
	int basecol = 0;

	for (int col = 0; col < cols; col++, basecol += 3)
	{
		image2[0][basecol] = image[0][basecol];
		image2[0][basecol + 1] = image[0][basecol + 1];
		image2[0][basecol + 2] = image[0][basecol + 2];
	}

	basecol = 0;

	for (int col = 0; col < cols; col++, basecol += 3)
	{
		image2[rows - 1][basecol] = image[rows - 1][basecol];
		image2[rows - 1][basecol + 1] = image[rows - 1][basecol + 1];
		image2[rows - 1][basecol + 2] = image[rows - 1][basecol + 2];
	}

	for (int row = 0; row < rows; row++)
	{
		image2[row][0] = image[row][0];
		image2[row][1] = image[row][1];
		image2[row][2] = image[row][2];
	}

	for (int row = 0; row < rows; row++)
	{
		image2[row][cols * 3 - 3] = image[row][cols * 3 - 3];
		image2[row][cols * 3 - 2] = image[row][cols * 3 - 2];
		image2[row][cols * 3 - 1] = image[row][cols * 3 - 1];
	}

}


//
// stretch_one_pixel:
//
void stretch_one_pixel(uchar** image2, uchar** image, int baserow, int basecol)
{
	int prevrow = baserow - 1;  // row above
	int nextrow = baserow + 1;  // row below

	// columns are a little trickier, since a "column" is really 3 physical cols: RGB
	int prevcol = basecol - 3;  // previous column start:
	int nextcol = basecol + 3;  // next column start:

	//
	// now update each pixel based on nearest neighbors around that pixel:
	//

	// Blue:
	image2[baserow][basecol] = NewPixelValue(image[prevrow][prevcol],
		image[prevrow][basecol],
		image[prevrow][nextcol],
		image[baserow][prevcol],
		image[baserow][basecol],
		image[baserow][nextcol],
		image[nextrow][prevcol],
		image[nextrow][basecol],
		image[nextrow][nextcol],
		1 /*stepby*/);

	// Green:
	basecol++;
	prevcol++;
	nextcol++;

	image2[baserow][basecol] = NewPixelValue(image[prevrow][prevcol],
		image[prevrow][basecol],
		image[prevrow][nextcol],
		image[baserow][prevcol],
		image[baserow][basecol],
		image[baserow][nextcol],
		image[nextrow][prevcol],
		image[nextrow][basecol],
		image[nextrow][nextcol],
		1 /*stepby*/);

	// Red:
	basecol++;
	prevcol++;
	nextcol++;

	image2[baserow][basecol] = NewPixelValue(image[prevrow][prevcol],
		image[prevrow][basecol],
		image[prevrow][nextcol],
		image[baserow][prevcol],
		image[baserow][basecol],
		image[baserow][nextcol],
		image[nextrow][prevcol],
		image[nextrow][basecol],
		image[nextrow][nextcol],
		1 /*stepby*/);
}


//
// Performs contrast stetch over the given image --- i.e. makes lighter colors lighter and
// darker colors darker.
//
// NOTE: the image matrix is a 2D matrix of pixels.  However, each pixel is 3 colors RGB,
// which means a "column" in the matrix is actually 3 columns: Blue, Green, and Red.  So
// as we loop over the matrix, we loop by 3 as we go along the column.
//
uchar **ContrastStretch(uchar **image, int rows, int cols, int startRow, int endRow, int totalRows)
{
	//
	// First, we need a temporary 2D matrix of the same size:
	//
	uchar **image2 = New2dMatrix<uchar>(totalRows, cols*3);

	//
	// copy the boundary rows into the temp image so we can swap
	// pointers after each step:
	//
	copy_boundary(image2, image, totalRows, cols);

	//
	// Okay, now perform contrast stretching, one step at a time:
	//
	int step = 1;

		// cout << "** Step " << step << "..." << endl;

		//
		// Okay, for each row (except boundary rows), lighten/darken pixel:
		//
		for (int row = startRow + 1; row < endRow - 1; row++)
		{
			//
			// And for each column (except boundary columns), lighten/darken pixel:
			//
			// columns are a little trickier, since a "column" is really 
			// 3 physical cols: RGB
			//
			int basecol = 3;  // start of column (skip boundary column 0):

			for (int col = 1; col < cols-1; col++, basecol += 3)
			{
				stretch_one_pixel(image2, image, row, basecol);
			}
		}

		//
		// flip the image pointers and step:
		//
		uchar** tempi = image;
		image = image2;
		image2 = tempi;

		step++;

	//
	// done!
	//
	Delete2dMatrix(image2);

	return image;
}


uchar **main_process(uchar **image, int rows, int cols, int steps, int numProcs) {
  cout << "main starting..." << endl;
  cout.flush();

  
  for (int w = 1; w < numProcs; w++) {
    int dest = w;
    int count = 1;
    int tag = 0;
    MPI_Send(&steps, count, MPI_INT, dest, tag, MPI_COMM_WORLD);
  }

  for (int i = 0; i < steps; i++) {
    cout << "Step " << i << endl;
    cout.flush();
    
    // int myRank = 0;
    int chunkSize = rows / numProcs; //possible issue here
    int startRow = chunkSize * (numProcs - 1);
    
    //send to workers
    for (int w = 1; w < numProcs; w++) {
      // cout << "Sending to worker " << w << endl;
      // cout.flush();

      int dest = w;
      int count = 1;
      int tag = 0;

      MPI_Send(&chunkSize, count, MPI_INT, dest, tag, MPI_COMM_WORLD);
      MPI_Send(&rows, count, MPI_INT, dest, tag, MPI_COMM_WORLD);
      MPI_Send(&cols, count, MPI_INT, dest, tag, MPI_COMM_WORLD);
      
      count = rows * cols * 3;
      //eventually only send chunks of the image
      // MPI_Send(&image[(w - 1) * chunkSize], count + 2, MPI_UNSIGNED_CHAR, dest, tag, MPI_COMM_WORLD);
      
      MPI_Ssend(image[0], count, MPI_UNSIGNED_CHAR, dest, tag, MPI_COMM_WORLD);
      
    }
    
    //WORK
    image = ContrastStretch(image, chunkSize, cols, startRow, rows, rows);
    
    //recv from workers
    for (int w = 1; w < numProcs; w++) {

      int src = w;
      int count = chunkSize * cols * 3;
      int tag = 0;
      
      int row = (w - 1) * chunkSize;
      
      MPI_Recv(image[row], count, MPI_UNSIGNED_CHAR, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
    }
  }
  
  return image;
}

void worker_process(int myRank, int numProcs) {
  cout << "worker " << myRank << " starting..." << endl;
  cout.flush();

  int src = 0;
  int count = 1;
  int tag = 0;

  int steps = 0;

  MPI_Recv(&steps, count, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int step = 0; step < steps; step++) {
    src = 0;
    count = 1;
    tag = 0;
    
    int chunkSize, rows, cols;
    
    MPI_Recv(&chunkSize, count, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&rows, count, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&cols, count, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    count = rows * cols * 3;
    uchar **image = New2dMatrix<uchar>(rows, cols*3);
    
    MPI_Recv(image[0], count, MPI_UNSIGNED_CHAR, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    int startRow = chunkSize * (myRank - 1);

    if (myRank == 1) startRow = 0;

    int endRow = chunkSize + startRow + 1;
    
    //WORK
    image = ContrastStretch(image, chunkSize, cols, startRow, endRow, rows);

    int dest = 0;
    count = chunkSize * cols * 3;
    tag = 0;

    MPI_Send(image[startRow], count, MPI_UNSIGNED_CHAR, dest, tag, MPI_COMM_WORLD);

    Delete2dMatrix(image);
  }
}