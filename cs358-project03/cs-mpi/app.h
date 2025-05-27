/*app.h*/

//
// Application-wide header file.
//
// Initial author:
//   Prof. Joe Hummel
//   Northwestern University
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include <cstdint>
#include <math.h>

using namespace std;


//
// types:
//
typedef unsigned char uchar;


//
// bitmap-related types:
//
#pragma pack(push, 1) // Ensure no padding
struct BITMAPFILEHEADER {
  uint16_t file_type{ 0x4D42 }; // "BM" in ASCII
  uint32_t file_size;
  uint16_t reserved1{ 0 };
  uint16_t reserved2{ 0 };
  uint32_t image_data_offset;
};

struct BITMAPINFOHEADER {
  uint32_t header_size{ 40 };
  int32_t width;
  int32_t height;
  uint16_t planes{ 1 };
  uint16_t bits_per_pixel;
  uint32_t compression{ 0 };
  uint32_t image_size;
  int32_t x_pixels_per_meter{ 0 };
  int32_t y_pixels_per_meter{ 0 };
  uint32_t colors_in_palette{ 0 };
  uint32_t important_colors{ 0 };
};
#pragma pack(pop)


//
// function prototypes:
//
/* Bitmap.cpp */
uchar **ReadBitmapFile(char *filename, BITMAPFILEHEADER &bitmapFileHeader, BITMAPINFOHEADER &bitmapInfoHeader, int &rows, int &cols);
void    WriteBitmapFile(char *filename, BITMAPFILEHEADER bitmapFileHeader, BITMAPINFOHEADER bitmapInfoHeader, uchar **image);

/* ContrastStretch.cpp */
uchar **ContrastStretch(uchar **image, int rows, int cols, int steps);

/* Main and Worker Processes */
uchar **main_process(uchar **image, int rows, int cols, int steps, int numProcs);
void worker_process(int myRank, int numProcs);