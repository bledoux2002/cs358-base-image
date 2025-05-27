/* debug.h */

//
// Debug function(s) for the contrast stretching app
//
// Initial author:
//   Prof. Joe Hummel
//   Northwestern University
//


// #include "app.h"
// #include "matrix.h"


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
);
