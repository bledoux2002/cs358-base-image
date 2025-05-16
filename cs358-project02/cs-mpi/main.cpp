/* main.cpp */

//
// Performs a contrast stretch over a Windows bitmap (.bmp) file, making lighter pixels
// lighter and darker pixels darker.
//
// Usage: cs infile.bmp outfile.bmp steps
//
// Benjamin Ledoux
//
// Initial author:
//   Prof. Joe Hummel
//   Northwestern University
//

#include "app.h"
#include <mpi.h>


//
// main:
//
int main(int argc, char* argv[])
{
	// MPI
	int myRank, numProcs;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	if (myRank > 0) {

		worker_process(myRank, numProcs);

	}
	else {
		char *infile;
		char *outfile;
		int   steps;

		//
		// process command-line args to program:
		//
		if (argc != 4)
		{
			cout << endl;
			cout << "Usage: cs infile.bmp outfile.bmp steps" << endl;
			cout << endl;
			return 0;
		}

		infile = argv[1];
		outfile = argv[2];
		steps = atoi(argv[3]);

		cout << endl;
		cout << "** Starting Contrast Stretch **" << endl;
		cout << "   Input file:  " << infile << endl;
		cout << "   Output file: " << outfile << endl;
		cout << "   Steps:       " << steps << endl;
		cout << endl;

		//
		// now let's input bitmap file:
		//
		BITMAPFILEHEADER bitmapFileHeader;
		BITMAPINFOHEADER bitmapInfoHeader;
		uchar **image;
		int rows, cols;

		cout << "** Reading bitmap..." << endl;
		image = ReadBitmapFile(infile, bitmapFileHeader, bitmapInfoHeader, rows, cols);
		if (image == NULL)
		{
			cout << "** Failed to open image file, halting..." << endl;
			return 1;
		}

		//
		// okay, perform contrast stretching:
		//
		cout << "** Processing..." << endl;

	  auto start = chrono::high_resolution_clock::now();

	  // image = ContrastStretch(image, rows, cols, steps);
	  image = main_process(image, rows, cols, steps, numProcs);

	  auto stop = chrono::high_resolution_clock::now();
	  auto diff = stop - start;
	  auto duration = chrono::duration_cast<chrono::milliseconds>(diff);

		//
		// Done, save image and output exec time:
		//
		cout << endl;
		cout << "** Done!  Time: " << duration.count() / 1000.0 << " secs" << endl;

		cout << "** Writing bitmap..." << endl;
		WriteBitmapFile(outfile, bitmapFileHeader, bitmapInfoHeader, image);

		cout << "** Execution complete." << endl;
		cout << endl;
	}

	MPI_Finalize();
	return 0;
}
