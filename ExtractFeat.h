#pragma once

#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream> // cout
#include <fstream> // Open files
#include <math.h>

using namespace cv;
using namespace std;

struct Fillet {
	String name;
	float hist_mean[3] = { 0 };						// BGR
	double area = 0,rectangularity=0,hullarity;						// Contour area + convexity(squarity) which is contour area divided by boundrect area.
	float width = 0, height = 0;					// of RotatedRect


	

	
	Rect boundRect;									// The img is generated from the original image using this boundingRect
	Point2f contour_center_mass;					//coordinate of contour center of mass
	vector<Point> contour,hull;							// Coordinates of the fillet 
	vector<vector<Point>> bloodstain_contours, notches,skin_contour;		// Coordinates of the bloodstains detected + Coordinates of the detected notches				
	Mat img, bin;									// Only the boundingRect image from original image + Binary image of fillet							
};

class ExtractFeat
{
public:
	String data_file_path = "C:\\Users\\Axel\\source\\repos\\GitHub_07_05\\GitHub_07_05\\Data\\features.dat";

	//------------Uden-For-Loop----------------------
	void clearFileContent();
	void displayImg(const String &name, const Mat &img);
	void makeBinary(const Mat &img, Mat &bin);
	//------------Nuværende-fisk----------------------
	void getMeanHist(Fillet &fillet);
	void getDimensions(Fillet &fillet);
	void getBloodStains(Fillet &fillet);
	void getnotches(Fillet &fillet);
	void getShape(Fillet &fillet);
	void getSkin(Fillet &fillet);

	//------------Efter-Features-------------------------------
	void saveFeatures(const Fillet &fillet);
	void drawFeatures(Mat &img, Fillet &fillet);
	//------------Main----------------------
	void run(vector<Mat> &images);
};

