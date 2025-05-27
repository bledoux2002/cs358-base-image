/* debug.cpp */

//
// Debug function(s) for the contrast stretching app
//
// Initial author:
//   Prof. Joe Hummel
//   Northwestern University
//

#include "app.h"


//
// sequential algorithm:
//
static uchar median(uchar a, uchar b, uchar c, uchar d, uchar e,
	uchar f, uchar g, uchar h, uchar i, uchar& min, uchar& max)
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
	for (int i = 0; i < 8; i++)
	{
		int min = i;

		// select smallest in subset:
		for (int j = i + 1; j < 9; j++)
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
static uchar NewPixelValue(uchar UL, uchar UP, uchar UR,
	uchar L, uchar P, uchar R,
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
		ratio = (P - min) / (double)(max - min);

		if (ratio < 0.5)  // darker, so make pixel darker if possible:
		{
			if (P > stepby) // beware of underflow:
				newp = P - stepby;
			else
				newp = 0;
		}
		else if (ratio > 0.5)  // lighter, so make pixel lighter if possible:
		{
			if (P < 255 - stepby)  // beware of overflow:
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
static void copy_boundary(uchar** image2, uchar** image, int rows, int cols)
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
static void stretch_one_pixel(uchar** image2, uchar** image, int baserow, int basecol)
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
static uchar** __cs(uchar** image, int rows, int cols, int steps)
{
	//
	// First, we need a temporary 2D matrix of the same size:
	//
	uchar** image2 = New2dMatrix<uchar>(rows, cols * 3);

	//
	// copy the boundary rows into the temp image so we can swap
	// pointers after each step:
	//
	copy_boundary(image2, image, rows, cols);

	//
	// Okay, now perform contrast stretching, one step at a time:
	//
	int step = 1;

	while (step <= steps)
	{
		//
		// Okay, for each row (except boundary rows), lighten/darken pixel:
		//
		for (int row = 1; row < rows - 1; row++)
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
		// flip the image pointers and step:
		//
		uchar** tempi = image;
		image = image2;
		image2 = tempi;

		step++;

	}//while-each-step

	//
	// done!
	//
	Delete2dMatrix(image2);

	return image;
}


//
// debug_compare_image:
//
// Opens and reads the image from the given file, runs the sequential
// algorithm for given steps, and then compares that result to the image
// you passed in. You control the rows and cols that you want compared
// via the [firstRol][firstCol] and [lastRow][lastCol] parameters. Note
// that firstCol and lastCol are in pixels, not actual column numbers in
// the image matrix. For example, if the image is 100x200, then to compare
// the entire image pass firstCol = 0 and lastCol = 199, even though there
// are technically 600 columns given that each pixel has 3 values RGB.s
// 
// Returns true if the image is correct, false if not.
//
// If verbose is false then the output is simply whether the image is
// the same or different. If verbose is true then you get output on 
// each pixel that is different.
//
bool debug_compare_image(
	char* filename,
	int steps,
	bool verbose,
	uchar** image,
	int firstRow, /*0-based index*/
	int lastRow,  /*0-based index*/
	int firstCol, /*0-based index*/
	int lastCol   /*0-based index*/
)
{
	cout << "** debug_compare_image **" << endl;

	BITMAPFILEHEADER bitmapFileHeader;
	BITMAPINFOHEADER bitmapInfoHeader;
	uchar** correct_image = NULL;
	int rows = 0, cols = 0;

	correct_image = ReadBitmapFile(filename, bitmapFileHeader, bitmapInfoHeader, rows, cols);
	if (correct_image == NULL) {
		cout << "** debug_compare_image: unable to open image file for comparison" << endl;
		return false;
	}
	 
	//
	// run sequential algorithm:
	//
	if (steps > 0)
		correct_image = __cs(correct_image, rows, cols, steps);

	//
	// now compare:
	//
	int diffs = 0;

	for (int r = firstRow; r <= lastRow; r++)
	{
		for (int c = firstCol; c <= lastCol; c++)
		{
			//
			// each colulmn expands into 3 values BGR:
			//
			int col = c * 3;

			if (image[r][col] != correct_image[r][col]) {
				diffs++;

				if (verbose)
					cout << "pixel diff (blue):  ["
					<< r << "][" << col << "] has value "
					<< (int) image[r][col] << " (should be "
					<< (int) correct_image[r][col] << ")" << endl;
			}

			col++;

			if (image[r][col] != correct_image[r][col]) {
				diffs++;

				if (verbose)
					cout << "pixel diff (green): ["
					<< r << "][" << col << "] has value "
					<< (int) image[r][col] << " (should be "
					<< (int) correct_image[r][col] << ")" << endl;
			}

			col++;

			if (image[r][col] != correct_image[r][col]) {
				diffs++;

				if (verbose)
					cout << "pixel diff (red):   ["
					<< r << "][" << col << "] has value "
					<< (int) image[r][col] << " (should be "
					<< (int) correct_image[r][col] << ")" << endl;
			}
		}//for
	}//for

	//
	// done!
	//
	cout << "# of pixel differences: " << diffs << endl;

	if (diffs > 0)
		cout << "summary: image is different/incorrect" << endl;
	else
		cout << "summary: image is same/correct" << endl;

	cout << "** DEBUG END **" << endl;

	Delete2dMatrix(correct_image);

	return (diffs == 0);
}
