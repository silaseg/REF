#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat frame);

// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber = 1; // Number of file to be saved
string filename;

// Function main
int main(void)
{
	VideoCapture capture(0);

	if (!capture.isOpened())  // check if we succeeded
		return -1;

	// Load the cascade
	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading\n");
		return (-1);
	};

	// Read the video stream
	Mat frame;

	for (;;)
	{
		capture >> frame;

		// Apply the classifier to the frame
		if (!frame.empty())
		{
			detectAndDisplay(frame);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		int c = waitKey(3);

		if (27 == char(c))
		{
			break;
		}
	}

	return 0;
}

// Function detectAndDisplay
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	Mat crop;
	Mat res;
    Mat gray;
	string text;
	stringstream sstm;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Détecter les faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 3| CASCADE_SCALE_IMAGE, Size(0, 0), Size(255,255));

	// Définir la région d'intérêt
	cv::Rect roi_b;
	cv::Rect roi_c;

	size_t ic = 0; // ic est l'indice de l'élément courant
	int ac = 0; //ac est la zone de l'élément courant
	size_t ib = 0; //ib est l'index du plus grand élément
	int ab = 0; // ab est la zone du plus grand élément

	for (ic = 0; ic < faces.size(); ic++) // Itération à travers tous les éléments courants (faces détectées)
	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height; // Obtenir la zone de l'élément courant (visage détecté)

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);

		ab = roi_b.width * roi_b.height; // Obtenir la zone de plus grand élément, au début, il est le même que "courant" élément

		if (ac > ab)
		{
			ib = ic;
			roi_b.x = faces[ib].x;
			roi_b.y = faces[ib].y;
			roi_b.width = (faces[ib].width);
			roi_b.height = (faces[ib].height);
		}

		crop = frame(roi_b);
		resize(crop, res, Size(255,255), 0, 0, INTER_LINEAR); // Cela sera nécessaire plus tard lors de l'enregistrement des images
		cvtColor(crop, gray, CV_BGR2GRAY); // Convertir l'image recadrée en niveaux de gris

										   // Form un nom de fichier
		filename = "";
		stringstream ssfn;
		ssfn << filenumber << ".png";
		filename = ssfn.str();
		filenumber;

		imwrite(filename, gray);
		//imwrite("C:/Users/User/Documents/Visual Studio 2012/image",crop);
		Point pt1(faces[ic].x, faces[ic].y); // Afficher les visages détectés sur la fenêtre principale - flux en direct de la caméra
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0,200,0), 6, 8, 0);

	}

	// Show image
	sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
	text = sstm.str();

	putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	imshow("original", frame);

	if (!crop.empty())
	{
		imshow("detected", crop);
	}
	else
		destroyWindow("detected");
}