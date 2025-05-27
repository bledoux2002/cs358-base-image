/* cs.cpp */

//
// Performs the actual contrast stretching.
//
// << YOUR NAME >>
//
// Initial author:
//   Prof. Joe Hummel
//   Northwestern University
//

#include "app.h"


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
// NOTE: every process is responsible for processing only a CHUNK of the image.  This
// chunk is rows x cols in size.  We assume that the image passed in contains not only
// the pixel data for the CHUNK, but also memory set aside for ghost row storage.
//
uchar **ContrastStretch(uchar **image, int rows, int cols, int steps)
{
	int tag = 0;
	int myRank;
	int numProcs;
	MPI_Status status;

	uchar** original_image = image;

	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);  // number of processes involved in run:
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);    // my proc id: 0 <= myRank < numProcs:

	//
	// First, EVERYONE needs a temporary 2D matrix of the same size as THEIR CHUNK (include
	// memory for ghost rows so that temp matrix is the same size, which makes the processing
	// loop below easier to write since indices line up one-for-one):
	//
	uchar **image2 = New2dMatrix<uchar>(rows+2, cols*3);  // worst-case: +2 ghost rows

	//
	// Okay, processes on the boundary of the image (the master and last worker, i.e.
	// processes that own top and bottom chunks of image) have one fewer row to process
	// since we don't process the boundary rows.  If we adjust now, this makes the 
	// processing loop MUCH cleaner:
	//
	if (myRank == 0)  // main:
		rows--;
	if (myRank == numProcs-1)  // last worker:
		rows--;

	//
	// Okay, now perform contrast stretching, one step at a time:
	//
	int step = 1;

	while (step <= steps)
	{
		if (myRank == 0)
		  cout << "** Step " << step << "..." << endl;

		//cout << myRank << " (" << host << "): Step " << step << "..." << endl;

		//
		// Exchange "ghost" rows, i.e. we need rows above/below our matrix, and our
		// neighbors likewise need our first/last rows:
		//
		// 1 of 2: everyone send their last data row *DOWN* to the next proc, and receive 
		// that row from the process above them and store into first (top) ghost row:
		//
		int dest = (myRank < numProcs-1) ? myRank + 1 : MPI_PROC_NULL;  // next process, or 0 if last process
		int src = (myRank > 0) ? myRank - 1 : MPI_PROC_NULL;  // previous process, or 0 if first process
		
		MPI_Sendrecv(chunk[rows], cols, MPI_UNSIGNED_CHAR, dest, tag,
		    image[0], cols, MPI_UNSIGNED_CHAR, src, tag, MPI_COMM_WORLD, &status);
		
		//
			if (myRank < numProcs-1)  // all send except the last process:
			MPI_Send(image[rows], cols*3, MPI_UNSIGNED_CHAR, myRank+1, tag, MPI_COMM_WORLD);

		if (myRank > 0)  // all receive except the first process, store into top row:
			MPI_Recv(image[0], cols*3, MPI_UNSIGNED_CHAR, myRank-1, tag, MPI_COMM_WORLD, &status);

		// 
		// 2 of 2: everyone send their first data row *UP* to the previous proc, and receive
		// that row from the process below them and store into their last (bottom) ghost row:
		//
		if (myRank > 0)  // all send except the first process:
			MPI_Send(image[1], cols*3, MPI_UNSIGNED_CHAR, myRank-1, tag, MPI_COMM_WORLD);

		if (myRank < numProcs-1)  // all receive except the last process, store in bottom row:
			MPI_Recv(image[rows+1], cols*3, MPI_UNSIGNED_CHAR, myRank+1, tag, MPI_COMM_WORLD, &status);

		//
		// Okay, for each row in OUR CHUNK, lighten/darken pixel:
		//
		for (int row = 1; row <= rows; row++)
		{
			//
			// And for each column (except boundary columns), lighten/darken pixel:
			//
			// columns are a little trickier, since a "column" is really 
			// 3 physical cols: RGB
			//
			int basecol = 3;  // start of column (skip boundary column 0):

			for (int col = 1; col < cols - 1; col++, basecol += 3)
			{
				stretch_one_pixel(image2, image, row, basecol);
			}
		}

		//
		// copy the boundary rows into the temp image so we can swap
		// pointers after each step:
		//
		copy_boundary(image2, image, rows + 2, cols);

		//
		// flip the image pointers and step:
		//
		uchar** tempi = image;
		image = image2;
		image2 = tempi;

		step++;

	}//while-each-step

	//
	// Okay, we're done, let's make sure the final results are in 
	// original image matrix so it's returned in the same memory
	// that was passed to us. We need to do this because the main
	// process passed in the ENTIRE image, and so it needs that
	// memory back --- we can't return the temporary image2 matrix
	// allocated here since it's a different size for main:
	//
	if (image != original_image) {

		for (int row = 1; row <= rows; row++)
		{
			int basecol = 3;  // start of column (skip boundary column 0):

			for (int col = 1; col < cols - 1; col++, basecol += 3)
			{
				original_image[row][basecol] = image[row][basecol];
				original_image[row][basecol + 1] = image[row][basecol + 1];
				original_image[row][basecol + 2] = image[row][basecol + 2];
			}
		}

		Delete2dMatrix(image);
	}
	else {
		//
		// results are already in original memory, so nothing to copy over:
		//

		Delete2dMatrix(image2);
	}

	//
	// Done! Return updated CHUNK:
	//
	return original_image;
}
