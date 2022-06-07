#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

//aproximare kernel Gaussian 3x3
int kernel3[3][3] = { 1, 2, 1,
                   2, 4, 2,
                   1, 2, 1 };

int pixel3(unsigned char* arr, int col, int row, int k, int width, int height)
{
    int sum = 0;
    int Kernel = 0;

    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            if ((row + j) >= 0 && (row + j) < height && (col + i) >= 0 && (col + i) < width)
            {
                int color = arr[(row + j) * 3 * width + (col + i) * 3 + k];
                sum += color * kernel3[i + 1][j + 1];
                Kernel += kernel3[i + 1][j + 1];
            }
        }
    }

    return sum / Kernel;
}

void gauss_3x3(unsigned char* arr, unsigned char* rez, int width, int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            for (int k = 0; k < 3; k++)
            {
                rez[3 * row * width + 3 * col + k] = pixel3(arr, col, row, k, width, height);
            }
        }
    }
}

int main()
{
    Mat img = imread("lena512.bmp");
    Mat out3 = img.clone();
    Mat out5 = img.clone();
    Mat res = img.clone();

    if (img.empty())
    {
        cout << "Eroare imagine" << endl;
        return 0;
    }
    
    //Blurarea imaginii cu 3x3 kernel Gaussian
    gauss_3x3(img.data, out3.data, 512, 512);

    //Blurarea imaginii cu 5x5 kernel Gaussian, folosind OpenCV
    GaussianBlur(img, out5, Size(5, 5), 0);

    //Diferenta absoluta intre cele doua variante de gaussian blur
    absdiff(out5, out3, res);

    //Afisarea inputului si a rezultatelor
    imshow("Imaginea originala", img);
    imshow("Gaussian filter 3x3", out3);
    imshow("Gaussian filter 5x5", out5);
    imshow("absdiff", res);

    waitKey(0);
    destroyAllWindows();

    return 0;
}