/* bitmap.cpp */

//
// Functions to read/write bitmap (.bmp) files.
//
// Initial author:
//   Prof. Joe Hummel
//   Northwestern University
//

#include "app.h"


//
// ReadBitmapFile: opens a given .bmp file and returns header info, # of rows and cols,
// and a ptr to a 2D matrix of pixel data.  The returned 2D matrix is really a 1D array
// of row pointers into a large, contiguous 1D raw image array.
//
// NOTE that each "pixel" is 3 bytes of the matrix --- blue, green, and red.  This implies
// that in order to iterate across the image, you need to do something like this:
//
//  for (int row = 0; row < rows; row++)
//    for (int col = 0; col < cols; col++)
//      cout << (int) image[row][col*3]     << " "     // blue:
//           << (int) image[row][col*3 + 1] << " "     // green:
//           << (int) image[row][col*3 + 2] << endl;   // red:
//
uchar **ReadBitmapFile(char *filename, BITMAPFILEHEADER &bitmapFileHeader, BITMAPINFOHEADER &bitmapInfoHeader, int &rows, int &cols)
{
	FILE *filePtr;

	// open filename in read binary mode
	filePtr = fopen(filename, "rb");
	if (filePtr == NULL)
	{
		cout << ">> Error in ReadBitmapFile: unable to open '" << filename << "'." << endl;
		return NULL;
	}

	//read the bitmap file header
	fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

	// verify that this is a bmp file by check bitmap id
	if (bitmapFileHeader.file_type !=0x4D42)
	{
		cout << ">> Error in ReadBitmapFile: '" << filename << "' is not a .bmp file." << endl;
		fclose(filePtr);
		return NULL;
	}

	// read the bitmap info header
	fread(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

	// move file point to the beginning of bitmap data:
	fseek(filePtr, bitmapFileHeader.image_data_offset, SEEK_SET);

	// allocate memory to hold image as a 2d matrix:
	if (bitmapInfoHeader.image_size == 0)  // sometimes size is 0?  Then compute ourselves:
		bitmapInfoHeader.image_size = bitmapInfoHeader.height * bitmapInfoHeader.width * 3 /*RGB*/;

	rows = bitmapInfoHeader.height;
	cols = bitmapInfoHeader.width;

	assert(bitmapInfoHeader.image_size == (uint32_t)rows * cols * 3);  // pixels * RGB == # of bytes
  assert(24 == bitmapInfoHeader.bits_per_pixel);

	uchar **image = New2dMatrix<uchar>(rows, cols*3);

	// read in the raw bitmap image data
	fread(image[0], bitmapInfoHeader.image_size, 1, filePtr);

	// close file and return bitmap iamge data
	fclose(filePtr);
	return image;
}


//
// WriteBitmapFile: assuming you've read the image in using ReadBitmapFile, just pass
// the headers and image here to write it back out to a .bmp file.  After the write,
// the memory associated with the 2D image array is freed (this memory was allocated
// when you called ReadBitmapFile).
//
void WriteBitmapFile(char *filename, BITMAPFILEHEADER bitmapFileHeader, BITMAPINFOHEADER bitmapInfoHeader, uchar **image)
{
	FILE *filePtr;

	// open filename in write binary mode, erasing previous file contents if necessary:
	filePtr = fopen(filename, "wb");
	if (filePtr == NULL)
	{
		cout << ">> Error in WriteBitmapFile: unable to open '" << filename << "'." << endl;
		return;
	}

	// write the bitmap file header
	fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

	// write the bitmap info header
	fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

	// move file point to the beginning of bitmap data:
	fseek(filePtr, bitmapFileHeader.image_data_offset, SEEK_SET);

	// write out the bitmap image data:
	uchar *bitmapImage = image[0];  // first row ptr == block of image data:
	fwrite(bitmapImage, bitmapInfoHeader.image_size, 1, filePtr);

	// free memory objects and close file:
	Delete2dMatrix(image);
	fclose(filePtr);
}
