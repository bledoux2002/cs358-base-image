/* main.cpp */

//
// Performs a contrast stretch over a Windows bitmap (.bmp) file, making lighter pixels
// lighter and darker pixels darker.
//
// Usage: cs infile.bmp outfile.bmp steps
//
// << YOUR NAME >>
//
// Initial author:
//   Prof. Joe Hummel
//   Northwestern University
//

#include "app.h"


//
// Function prototypes:
//
uchar** DistributeImage(int myRank, int numProcs, uchar** image, int& rows, int& cols, int& rowsPerProc, int& leftOverRows);
uchar** CollectImage(int myRank, int numProcs, uchar** image, int rows, int cols, int rowsPerProc, int leftOverRows);


//
// main:
//
int main(int argc, char* argv[])
{
	char *infile;
	char *outfile;
	int   steps, myRank, numProcs;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);  // number of processes involved in run:
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);  	 // my proc id: 0 <= myRank < numProcs:

	//
	// process command-line args to program:
	//
	if (argc != 4)
	{
		cout << endl;
		cout << "Usage: mpiexec ... cs infile.bmp outfile.bmp steps" << endl;
		cout << endl;
		MPI_Abort(MPI_COMM_WORLD, 0 /*return code*/);
	}

	infile = argv[1];
	outfile = argv[2];
	steps = atoi(argv[3]);

	char host[128];

	gethostname(host, sizeof(host)/sizeof(host[0]));  // machine we are running on:

	cout << "process " << myRank 
	     << " starting on node '" << host << "'..." 
		 << endl;
	cout.flush();

	if (myRank == 0)
	{
		cout << endl;
		cout << "** Starting Contrast Stretch **" << endl;
		cout << "   Input file:  " << infile << endl;
		cout << "   Output file: " << outfile << endl;
		cout << "   Steps:       " << steps << endl;
		// cout << endl;
	}

	//
	// now let's input bitmap file:
	//
	BITMAPFILEHEADER bitmapFileHeader;
	BITMAPINFOHEADER bitmapInfoHeader;
	uchar** image = nullptr;
	int rows = 0, cols = 0, rowsPerProc = 0, leftOverRows = 0;

	if (myRank == 0)
	{
		//cout << "** Reading bitmap..." << endl;
		image = ReadBitmapFile(infile, bitmapFileHeader, bitmapInfoHeader, rows, cols);
		if (image == NULL)
		{
			cout << "** Failed to open image file, halting..." << endl;
			return 0;
		}

		cout << "   Image size:  "
		     << rows << " rows, "
			 << cols << " columns" << endl;
	    cout << endl;
	}

	//
	// okay, perform contrast stretching:
	//
	if (myRank == 0)
	{
		cout << "** Processing..." << endl;
	}

    auto start = chrono::high_resolution_clock::now();

	//
	// MASTER distributes matrix, WORKERS receives their chunk of matrix:
	//
	image = DistributeImage(myRank, numProcs, image, rows, cols, rowsPerProc, leftOverRows);

	//
	// Okay, everyone performs constrast-stretching on their chunk:
	//
	image = ContrastStretch(image, rows, cols, steps);

	//
	// Collect the results: WORKERS send, MASTER receives and puts image back together:
	//
	image = CollectImage(myRank, numProcs, image, rows, cols, rowsPerProc, leftOverRows);

    auto stop = chrono::high_resolution_clock::now();
    auto diff = stop - start;
    auto duration = chrono::duration_cast<chrono::milliseconds>(diff);

	//
	// Done, save image and output exec time:
	//
	if (myRank == 0)
	{
		cout << endl;
		cout << "** Done!  Time: " << duration.count() / 1000.0 << " secs" << endl;

		cout << "** Writing bitmap..." << endl;
		WriteBitmapFile(outfile, bitmapFileHeader, bitmapInfoHeader, image);

		cout << "** Execution complete." << endl;
		cout << endl;
	}

	//
	// done:
	//
	MPI_Finalize();

	return 0;
}


//
// DistributeImage: given the original image, rows and cols (from the master process),
// the master distributes this data to the worker processes.
//
// Upon return, the master has the entire image, but now rows x cols reflects the
// CHUNK the master should process.  For the workers, the image matrix contains
// their CHUNK of the matrix to process (this matrix includes room for ghost rows),
// along with the size of the matrix chunk (in rows and cols).
//
// NOTE: any extra rows (due to uneven split across processes) are kept by the master
// process; these extra rows are viewed as the start of the image.
//
uchar** DistributeImage(int myRank, int numProcs,
	                    uchar** image, int& rows, int& cols, 
						int& rowsPerProc, int& leftOverRows)
{
	int  params[2];
	int  src, dest, tag = 0;
	MPI_Status  status;

	//cout << myRank << " (" << host << "): Distributing image..." << endl;

	//
	// Master: distribute size of each worker's CHUNK (rows x cols), then the image
	// data itself:
	//
	if (myRank == 0)  // Master:
	{
		rowsPerProc = rows / numProcs;
		leftOverRows = rows % numProcs;

		params[0] = rowsPerProc;
		params[1] = cols;

		// for all workers, each gets an equal chunk (master keeps extra rows):
		for (dest = 1; dest < numProcs; dest++)
			MPI_Send(params, sizeof(params) / sizeof(params[0]), MPI_INT, dest, tag, MPI_COMM_WORLD);

		//
		// Note that for simplicity we are *not* sending the ghost rows along with the chunks, 
		// we are just sending the image chunks owned by each process.  This will require an 
		// extra send/recv later on, but the code is much cleaner here as a result.
		//

		// send chunk to each worker (skipping over extra rows owned by master):
		for (dest = 1; dest < numProcs; dest++)
			MPI_Send(image[leftOverRows + dest * rowsPerProc], rowsPerProc * cols * 3, MPI_UNSIGNED_CHAR, dest, tag, MPI_COMM_WORLD);

		// okay, master is now only responsible for their own (the first) chunk:
		rows = rowsPerProc + leftOverRows;
	}

	/* All processes execute broadcast */
	int sender = 0;
	MPI_Bcast(params, 2 /*count*/, MPI_INT, sender, MPI_COMM_WORLD);

	rows = params[0];  // rowsPerProc for workers, or rowsPerProc + leftOverRows for master
	cols = params[1];  // cols is the same for all processes

	// allocate memory for the image chunk:
	chunk = New2dMatrix<uchar>(rows + 2, cols);  // worst-case: +2 ghost rows

	uchar* sendbuf = (myRank == 0) ? image[leftOverRows] : NULL;  // master sends their chunk, workers receive their chunk

	sender = 0;  // master sends, workers receive
	MPI_Scatter(sendbuf, rowsPerProc * cols, MPI_UNSIGNED_CHAR, chunk[1], rowsPerProc * cols, MPI_UNSIGNED_CHAR, sender, MPI_COMM_WORLD);

	else  // Workers:
	{
		//
		// Workers: receive size of our CHUNK in rows x cols, allocate data for such a 
		// chunk, and then receive the data itself:
		//
		src = 0;  // from master
		MPI_Recv(params, sizeof(params) / sizeof(params[0]), MPI_INT, src, tag, MPI_COMM_WORLD, &status);

		rows = params[0];
		cols = params[1];

		//
		// Next, workers need to create image matrix for CHUNK they will own, including ghost
		// rows they will need during processing:
		//
		image = New2dMatrix<uchar>(rows + 2, cols * 3);  // worst-case: 2 ghost rows (+2)

		//
		// okay workers, receive data into newly-allocated matrix, skipping the first row
		// since it will be used for ghost row storage (which is why it's image[1] below 
		// and not image[0]):
		//
		src = 0;  // from master
		MPI_Recv(image[1], rows * cols * 3, MPI_UNSIGNED_CHAR, src, tag, MPI_COMM_WORLD, &status);
	}

	//
	// Done!  Everyone returns back a matrix to process... (rows and cols should already
	// be set for both master and workers)
	//
	return image;
}


//
// CollectImage: each worker sends in their image chunk (of size rows x cols), and 
// sends this chunk to the master.  The master collects these chunks and writes them
// back to the original image matrix.  Note that for all processes, rows and cols 
// denotes the size of their matrix CHUNK, not the original matrix size.
//
uchar** CollectImage(int myRank, int numProcs,
	                 uchar** image, int rows, int cols, 
	                 int rowsPerProc, int leftOverRows)
{
	receiver = 0;  // master receives, workers send
	uchar* recvbuf = (myRank == 0) ? image[leftOverRows] : NULL;  // workers send their chunk, master receives

	MPI_Gather(chunk[1], rowsPerProc * cols, MPI_UNSIGNED_CHAR, 
	    recvbuf, rowsPerProc * cols, MPI_UNSIGNED_CHAR, receiver, MPI_COMM_WORLD);

	int src, dest, tag;
	MPI_Status status;

	//cout << myRank << " (" << host << "): Collecting image..." << endl;

	//
	// WORKERS send, MASTER receives:
	//
	tag = 0;

	if (myRank > 0)  // workers:
	{
		dest = 0;  // to master
		// NOTE: skip over the first (ghost) row when sending in our final results:
		MPI_Send(image[1], rows * cols * 3, MPI_UNSIGNED_CHAR, dest, tag, MPI_COMM_WORLD);

		// workers are done with CHUNK, so free associated memory:
		Delete2dMatrix<uchar>(image);
		image = NULL;
	}
	else  // master:
	{
		//
		// we could allocate a temp buffer and receive msgs in any order, but there doesn't seem
		// much point since we are done after this.  So we'll keep the code simple and just
		// receive in the order the chunks are copied back into the image matrix:
		//

		// for all workers, receive their chunk and store back into matrix:
		for (src = 1; src < numProcs; src++)
			MPI_Recv(image[leftOverRows + src * rowsPerProc], rowsPerProc * cols * 3, MPI_UNSIGNED_CHAR, src, tag, MPI_COMM_WORLD, &status);
	}

	// 
	// Done, return final image:
	//
	return image;
}
