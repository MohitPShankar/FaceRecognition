//
//  main.cpp
//  Stat-Face_Rec
//
//  Created by Avinash Bhaskaran on 4/8/15.
//  Copyright (c) 2015 Avinash Bhaskaran. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <sstream>
#include <dirent.h>
#include <glob.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <string.h>
#include <cstring>
#include <stdlib.h> // for atoi function
#include <fstream>

using namespace cv;
using namespace std;
// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "/Users/Mohit/Downloads/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved
string filename;
Mat cropped_image;
typedef struct pdmtype  {
    short pdm_patch[196][8];
} pdm_patch;
typedef struct pdm_desc {
    double desc[64][30];
} descriptor;
std::vector <pdm_patch> pdm;
string root_path = "/users/Mohit/Desktop/Stat-Face_Rec/build/Debug/Database";
int number_of_classes;
unordered_map<string, int> numImagesperClass;
unordered_map< string,std::vector < std::vector <pdm_patch> > > megamap;
unordered_map< string,std::vector < descriptor > > descriptorMap;
unordered_map<string,pdm_patch> mean_class_pdm;
pdm_patch mean_global_pdm;
Mat w(196,10,CV_64F),v(8,3,CV_64F);//,sw1(196,196,CV_64F),





void Create_PDM( )  {
    pdm_patch patch;
//    std::cout<<cropped_image.rows<<" "<<cropped_image.cols<<std::endl;
//    int count = 0;
//    short patch[196][8];
    for ( int i = 0; i< 128; i+=16) {
        for ( int j = 0; j<128; j+=16 )  {
            int patch_id = 0;
            for ( int pixely = 1; pixely< 15; pixely++) {
                for (int pixelx = 1; pixelx <15; pixelx++)   {
                    patch.pdm_patch[patch_id][0] = cropped_image.ptr(i + pixely-1)[j + pixelx -1] - cropped_image.ptr(i + pixely)[j + pixelx];
                    patch.pdm_patch[patch_id][1] = cropped_image.ptr(i + pixely-1)[j + pixelx] - cropped_image.ptr(i + pixely)[j + pixelx];
                    patch.pdm_patch[patch_id][2] = cropped_image.ptr(i + pixely-1)[j + pixelx +1] - cropped_image.ptr(i + pixely)[j + pixelx];
                    patch.pdm_patch[patch_id][3] = cropped_image.ptr(i + pixely)[j + pixelx-1] - cropped_image.ptr(i + pixely)[j + pixelx];
                    patch.pdm_patch[patch_id][4] = cropped_image.ptr(i + pixely)[j + pixelx +1] - cropped_image.ptr(i + pixely)[j + pixelx];
                    patch.pdm_patch[patch_id][5] = cropped_image.ptr(i + pixely+1)[j + pixelx -1] - cropped_image.ptr(i + pixely)[j + pixelx];
                    patch.pdm_patch[patch_id][6] = cropped_image.ptr(i + pixely+1)[j + pixelx] - cropped_image.ptr(i + pixely)[j + pixelx];
                    patch.pdm_patch[patch_id][7] = cropped_image.ptr(i + pixely+1)[j + pixelx +1] - cropped_image.ptr(i + pixely)[j + pixelx];
                    patch_id++;
                }
            }
            pdm.push_back(patch);
        }
    }
    std::cout<<pdm.size()<<" is size of pdm" << std::endl;
}

void Get_ClassMeanPDM() {
    DIR* dir = opendir(root_path.c_str());
    struct dirent *ent = readdir(dir);
    if ( ent != NULL)  {
        cout<<"I don't think I've seen you before. Please enter your name"<<std::endl;
        string dirname;
        cin>>dirname;
        dirname = root_path + "/" + dirname;
#if defined(_WIN32)
        mkdir(dirname.c_str());
#else
        mkdir(dirname.c_str(),0777);
#endif
        string filename = dirname + "/" + "0.png";
        imwrite(filename, cropped_image);
        
    }
    while (true)    {
        
    }

}

void Get_GlobalPDM()    {
    Mat sb1(196,196,CV_64F),sw2(8,8,CV_64F),sb2(8,8,CV_64F);
    setIdentity(w,1);
    setIdentity(v,1);
    double norm_thresh_v = 0.01, norm_thresh_w = 0.1;
    double vnorm = 1, wnorm = 1;
    int iter = 0;
    while (iter<100 && ((vnorm<norm_thresh_v) || (wnorm<norm_thresh_w)))  {
        Mat sw1 = Mat::zeros(196, 196, CV_64F);
        Mat sb1 = Mat::zeros(196, 196, CV_64F);
        Mat sw2 = Mat::zeros(8, 8, CV_64F);
        Mat sb2 = Mat::zeros(8, 8, CV_64F);
        for ( auto i = megamap.begin();i!=megamap.end(); i++) {
            for (int j = 0; j< 64; j++) {
                for ( int k = 0; k< i->second.size(); k++)  {
                    Mat temp1,temp1T,vT,temp2,temp3,temp4;
                    Mat pdm1 = Mat(196,8,CV_64F,(i->second[k])[j].pdm_patch);
                    Mat pdm2 = Mat(196,8,CV_64F,mean_class_pdm[i->first].pdm_patch);
//                    std::cout<<(i->second[k])[j].pdm_patch[10][6];
                    subtract(pdm1,pdm2,temp1);
                    transpose(temp1,temp1T);
                    transpose(v,vT);
                    temp2 = v*vT;
//                    gemm(v, vT,1,new Mat(),0,temp2,0);
                    temp3 = temp1*temp2;
//                    gemm(temp1,temp2,temp3);
                    temp4 = temp3*temp1T;
//                    gemm(temp3,temp1T,temp4);
                    sw1+=temp4;
                }
                Mat temp1,temp1T,vT,temp2,temp3,temp4;
                Mat pdmmean = Mat(196,8,CV_64F,mean_class_pdm[i->first].pdm_patch);
                Mat pdmglobal = Mat(196,8,CV_64F,mean_global_pdm.pdm_patch);
                subtract(pdmmean,pdmglobal,temp1);
                transpose(temp1, temp1T);
                transpose(v,vT);
                temp2 = v*vT;
//                gemm(v, vT, temp2);
                temp1 = temp1*temp2;
//                gemm(temp1,temp2,temp1);
                temp4 = temp1*temp3;
//                gemm(temp1,temp3,temp4);
                sb1+=temp4;
            }
        }
        Mat temp;
        invert(sw1, temp);
        Mat temp2;
        temp2 = temp*sb1;
        //gemm(temp, sb1, temp2);
        Mat X,Lambda;
        eigen(temp2,Lambda,X,0,9);
        subtract(X,w,temp);
        wnorm = norm(temp,NORM_L2);
        w=X;

        
        for ( auto i = megamap.begin();i!=megamap.end(); i++) {
            for (int j = 0; j< 64; j++) {
                for ( int k = 0; k< i->second.size(); k++)  {
                    Mat temp1,temp1T,wT,temp2,temp3,temp4;
                    Mat pdm1 = Mat(196,8,CV_64F,(i->second[k])[j].pdm_patch);
                    Mat pdm2 = Mat(196,8,CV_64F,mean_class_pdm[i->first].pdm_patch);
//                    std::cout<<(i->second[k])[j].pdm_patch[10][6];
                    subtract(pdm1,pdm2,temp1);
                    transpose(temp1,temp1T);
                    transpose(w,wT);
                    temp2 = w*wT;
//                    gemm(w, wT,temp2);
                    temp3 = temp1T*temp2;
//                    gemm(temp1T,temp2,temp3);
                    temp4 = temp3*temp1;
//                    gemm(temp3,temp1,temp4);
                    sw2+=temp4;
                }
                Mat temp1,temp1T,vT,temp2,temp3,temp4;
                Mat pdmmean = Mat(196,8,CV_64F,mean_class_pdm[i->first].pdm_patch);
                Mat pdmglobal = Mat(196,8,CV_64F,mean_global_pdm.pdm_patch);
                subtract(pdmmean,pdmglobal,temp1);
                transpose(temp1, temp1T);
                transpose(v,vT);
                temp2 = v*vT;
//                gemm(v, vT, temp2);
                temp3 = temp1T*temp2;
//                gemm(temp1T,temp2,temp3);
                temp4 = temp3*temp1;
//                gemm(temp3,temp1,temp4);
                sw2+=temp4;
            }
        }
        invert(sw2, temp);
        temp2 = temp*sb2;
//        gemm(temp, sb2, temp2);
        eigen(temp2,Lambda,X,0,9);
        subtract(X,v,temp);
        vnorm = norm(temp,NORM_L2);
        v=X;
    }
    iter++;
}


void getFeatures()  {
    Mat wT,temp,temp1;
    transpose(w,wT);
    for ( auto i = megamap.begin(); i!= megamap.end();i++)   {
        for ( int j = 0; j<i->second.size(); j++ )  {
            descriptor d;
            for ( int k = 0; k< 64; k++)    {
                Mat pdm = Mat(196,8,CV_64F,(i->second[j])[k].pdm_patch);
                temp1 = wT*pdm*v;
                for ( int ii = 0; ii < 10;ii++)    {
                    for ( int jj = 0; jj<2; jj++ )  {
                        d.desc[k][ii*3+jj] = temp.ptr(ii)[jj];
                    }
                }
            }
            descriptorMap[i->first].push_back(d);
        }
    }
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
    std::stringstream sstm;
    
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    
    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    
    // Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;
    
    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element
    
    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element
    
    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)
        
    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);
        
        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)
        
        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);
        
        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element
        
        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }
        
        crop = frame(roi_b);
        resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
        cvtColor(res, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale
        
        // Form a filename
        filename = "";
        stringstream ssfn;
        ssfn << filenumber << ".png";
        filename = ssfn.str();
        filenumber++;
        
        imwrite(filename, gray);
        cropped_image = gray.clone();
        
        Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    }
    
    // Show image
    std::cout<<roi_b.x<<" "<<roi_b.y<<" "<<roi_b.height<<" "<<roi_b.width<<std::endl;
    sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
    text = sstm.str();
    
    putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
    while (true)    {
        namedWindow("original",0);
        imshow("original", frame);
        if (waitKey(1)>=0)
            break;
    }
//    std::cout<<crop.empty();
    while (true)    {
        cv::namedWindow("detected",1);
        if (!crop.empty())
        {
            imshow("detected", crop);
        }
        else
            destroyWindow("detected");
        if (waitKey(1)>=0)
            break;
    }
}

void mult( Mat a, Mat b, Mat& c)
{
    c = Mat::ones( a.rows, b.cols, CV_64F);
    for (double iter = 0; iter < b.cols; iter++)    {
        for ( double i = 0; i< a.rows; i++)  {
            for ( double j = 0; j<a.cols; j++)  {
                c.at<double>(i,iter) +=(double)(a.at<double>(i,j) * b.at<double>(j,iter));
            }
        }
    }
}

void read_pdm(string filename, std::vector < std::vector <pdm_patch> > &class_pdm)
{
    
    pdm_patch patch;
    ifstream myfile;
    myfile.open(filename);
//    FILE* fr = fopen(filename.c_str(), "rt");
    string line;
//    char line[20];
    char* temp;
    int num_im;
    
    getline(myfile,line);
//    fgets(line, 16, fr);
//    {
//        while ( getline(myfile,line) )
//        {
//            cout << line << '\n';
//        }
//        myfile.close();
//    }
    
    sscanf(line.c_str(), "%d", &num_im); // get number of images
    cout << num_im;
    int k = 0;
    while ( k < num_im)
    {
        std::vector <pdm_patch> image_pdm;
        for (int patch_id = 0; patch_id < 64; ++patch_id)
        {
            for(int i = 0; i < 196; ++i)
            {
                int j = 0;
                getline(myfile,line);
                //                temp = string(line);
                char* dup = strdup(line.c_str());
                temp = strtok(dup," ");
                free(dup);
                while ((temp != NULL)&&(j<8))
                {
                    patch.pdm_patch[i][j] = atoi(temp);
                    j++;
                    temp = strtok (NULL," ");
                }
            }
            image_pdm.push_back(patch);
        }
        class_pdm.push_back(pdm);
        ++k;
    }
    //    return 0;
}

void write_pdm(string filename, std::vector < std::vector <pdm_patch> > class_pdm)
{
    std::cout<<filename<<std::endl;
    ofstream myfile;
    myfile.open(filename);
//    FILE* fw = fopen(filename.c_str(), "w");
    if (myfile.is_open())   {
//        fprintf(fw, "%d\n", class_pdm.size());
        myfile<<class_pdm.size()<<std::endl;
        for(auto it = class_pdm.begin(); it != class_pdm.end(); ++it)
        {
            std::vector<pdm_patch> image_pdm= *it;
            for(auto it1 = image_pdm.begin(); it1!= image_pdm.end(); ++it1)
            {
                pdm_patch patch = *it1;
                for(int i = 0; i < 196; ++i)
                {
                    for(int j = 0; j < 8; ++j)
                    {
                        myfile<<patch.pdm_patch[i][j]<< " ";
//                        fprintf(fw, "%d ", patch.pdm_patch[i][j]);
                    }
                    myfile<<std::endl;
//                    fprintf(fw, "\n");
                }
            }
        }
        myfile.close();
    }
    else    {
        std::cout<<"File open Error"<<std::endl;
    }
}


//returns 1 for failure
int load_data()
{
    string root_path2 = root_path;
    string dat_file = "pdm.dat";
    DIR* dir = opendir(root_path2.c_str());
    struct dirent *ent = readdir(dir);
    while( ent != NULL )  {
        if ( ent->d_type == DT_DIR && !strncmp(ent->d_name,".",1) && !strncmp(ent->d_name,"..",2))    {
            string temp = ent->d_name;
            temp = temp + "/" + dat_file;//root_path + "/" +
            std::vector < std::vector <pdm_patch> > class_pdm;
            read_pdm(temp,class_pdm);
            megamap[temp] = class_pdm;
            //            count++;
            //            std::cout<<"path is :"<<ent->d_name;
        }
        ent = readdir(dir);
        std::cout<<"ha";
    }
    return 0;
}


int write_data()    {
    for(auto it = megamap.begin(); it != megamap.end(); ++it)
    {
        string label = it->first;
        std::vector < std::vector <pdm_patch> > class_pdm = it->second;
        std::cout<<"vector size is "<<class_pdm.size()<<std::endl;
        string temp = label + "/" + "pdm.dat";//root_path + "/" +
        write_pdm(temp, class_pdm);
    }
    return 0;
}



int main(int argc, const char * argv[])
{
    cv::Mat Image;
    cv::VideoCapture capture(0); // open default camera
    if ( capture.isOpened() == false )
        return -1;
    
    //    cv::namedWindow("Test OpenCV 2.4.9",1);
    cv::Mat frame;
    //    cv::namedWindow("Image clicked",0);
    char confirm;
    while ( true )
    {
        capture >> frame;
        cv::namedWindow("Test OpenCV 2.4.9",1);
        cv::imshow("Test OpenCV 2.4.9", frame );
        int key = cv::waitKey(1);
        if ( key >=0 ) {
            Image = frame;
            cv::namedWindow("Image clicked",1);
            cv::imshow("Image clicked", Image);
            std::cout<<" Confirm Image : y/n"<<std::endl;
            std::cin>>confirm;
            if (confirm == 'y') {
                break;
            }
            else    {
                cv::destroyWindow("Image clicked");
                //                key = cv::waitKey(1);
            }
        }
    }
    while (true)    {
        cv::namedWindow("Image clicked",0);
        cv::imshow("Image clicked", Image);
        //    std::cout<<Image.size();
        if (cv::waitKey(1)>=0)  break;
    }
    if (!face_cascade.load(face_cascade_name))
    {
        printf("--(!)Error loading\n");
        return (-1);
    };
    
    // Read the image file
    frame = Image;
    // Apply the classifier to the frame
    if (!frame.empty())
    {
        detectAndDisplay(frame);
        
    }
    else
    {
        printf(" --(!) No captured frame -- Break!");
        //        break;
    }
    
    //    int c = waitKey(10);
    
    //    if (27 == char(c))
    //    {
    //        break;
    //    }
    
    int count = 0;
    
    //-- Show what you got
    namedWindow("Gray Image",0);
    imshow( "Gray Image", frame );
    Create_PDM();
    DIR* dir = opendir(root_path.c_str());
    struct dirent *ent = readdir(dir);
    while( ent != NULL)  {
        if ( ent->d_type == DT_DIR )    {
            count++;
            std::cout<<"count is :"<<count;
        }
        ent = readdir(dir);
        std::cout<<"    ";//<<std::flush;
    }
    
        
    if ( count<3 )    {
        cout<<"I don't think I've seen you before. Please enter your name"<<std::endl;
        string dirname;
        cin>>dirname;
        dirname = root_path + "/" + dirname;
#if defined(_WIN32)
        mkdir(dirname.c_str());
#else
        mkdir(dirname.c_str(),0777);
#endif
        string filename = dirname + "/" + "0.png";
        imwrite(filename, cropped_image);
        megamap[dirname].push_back(pdm);
        write_data();
    }
    else    {
        std::cout<<"reached1"<<std::cout;
        load_data();
        std::cout<<megamap.size()<<" is size of megamap"<<std::endl;
    }
    
//    std::cout<<"You are :  AVINASH" <<std::endl;
//    int a;
//    cin>>a;
    
    
    return 0;
}

//bharath

