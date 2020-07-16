#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

/* Credit correlation() - Paul Bourke */
void correlation(double* x, double* y, int n, double* r/*out*/)
{
    int i, j, delay, maxdelay;
    double mx, my, sx, sy, sxy, denom;
    maxdelay = n / 2;

    /* Calculate the mean of the two series x[], y[] */
    mx = 0;
    my = 0;
    for (i = 0;i < n;i++) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;

    /* Calculate the denominator */
    sx = 0;
    sy = 0;
    for (i = 0;i < n;i++) {
        sx += (x[i] - mx) * (x[i] - mx);
        sy += (y[i] - my) * (y[i] - my);
    }
    denom = sqrt(sx * sy);

    /* Calculate the correlation series */
    for (delay = -maxdelay;delay < maxdelay;delay++) {
        sxy = 0;
        for (i = 0;i < n;i++) {
            j = i + delay;
            if (j < 0 || j >= n)
                continue;
            else
                sxy += (x[i] - mx) * (y[j] - my);
            /* Or should it be (?)
            if (j < 0 || j >= n)
               sxy += (x[i] - mx) * (-my);
            else
               sxy += (x[i] - mx) * (y[j] - my);
            */
        }

        r[delay + maxdelay] = sxy / denom;
        /* r is the correlation coefficient at "delay" */

    }
}

std::string exec(const std::string strcmd) {
    int n = strcmd.length();
    char* charcmd = new char[n + 1];
    strcpy(charcmd, strcmd.c_str());
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(charcmd, "r"), _pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int main(int argc, const char* argv[])
{
    if (argc > 1) {
        int arg = argc - 1;
        std::cout << "V2: " << std::endl;
        std::cout << "Arguments: " << argc - 1 << std::endl;        

        const long window = 300; //Fenêtre de recherche intercorrelation

        std::vector< std::vector<float> > corr;
        corr.reserve(argc - 1);
        for (int i = 0; i < argc - 1; i++) {
            corr.push_back(std::vector<float>(window));
        }

        //Get mean luma of each frame for future cross-correlation
        cv::VideoCapture cap;
        for (int i = 0; i < argc - 1; i++) {
            cap = cv::VideoCapture(argv[i + 1]);
            for (int j = 0; cap.isOpened() && j < window; j++) {
                cv::Mat img;
                cap >> img;
                cv::Scalar m = cv::mean(img);
                //std::cout <<"n°"<<j<<": " << m[0] << std::endl;
                corr[i][j] = (double)m[0]; // A faire: Vérifier qu'on utilise le vert.
            }
        }
        //Luma mean for reference
        double ref[window];
        for (int j = 0; j < window; j++) {
            ref[j] = corr[0][j];
        }

        std::vector<int> delay;
        delay.reserve(argc - 1);
        delay.push_back(0); //Set delay for ref
        //Cross-correlation for each comps
        for (int i = 0; i < argc - 2; i++) {
            double comps[window]; //Luma mean for comps
            double coefcorr[window];   //Coefficients of cross-corr
            for (int j = 0; j < window; j++) {
                comps[j] = corr[i + 1][j];
                coefcorr[j] = 0; //Initialize array
            }
            correlation(ref, comps, window, coefcorr); //See cross-correlation function

            long index;
            double maxcorr = -2; // correlaration between -1 and 1
            for (int j = 0; j < window; j++) {
                if (coefcorr[j] > maxcorr) {
                    maxcorr = coefcorr[j];
                    index = j;
                    //std::cout << "Max: " << coefcorr[j] << ", Index: " << index << std::endl;
                }
            }
            //std::cout << "-----------------------------" << std::endl;
            std::cout << "Delay: " << index - window / 2 << ", MaxCoef: " << maxcorr << std::endl;
            delay.push_back(index - (window / 2));
        }
        std::vector<std::string> pathsynced;
        for (int i = 0; i < delay.size(); i++) {
            std::string str = argv[i+1];
            size_t found = str.find_last_of(".ts");
            std::string Name = str.substr(0, found-2) + "synced.ts";
            pathsynced.push_back(Name);
            //std::string strcmd = "ffmpeg -y -ss " + std::to_string((float) delay[i]/25) + " -i "+ argv[i+1] + " -c copy "+ "-f mpegts "+ Name;
            std::string strcmd = "ffmpeg -y -ss " + std::to_string((float) delay[i]/25) + " -i "+ argv[i+1] + " -c:v libx264 -preset ultrafast -crf 0 "+ "-f mpegts "+ Name;
            std::cout << strcmd << std::endl;
            exec(strcmd);
        }

        for (int i = 0; i < pathsynced.size(); i++) {
            std::string ref = pathsynced[0];
            std::string comps = pathsynced[i];
            size_t found = comps.find_last_of("synced.ts");
            std::string logpath = comps.substr(0, found - 2) + ".log";
            logpath.insert(1, "\\");
            std::string strcmd = "ffmpeg -i "+ comps  +" -i " + ref +  " -lavfi  psnr=\"stats_file='" + logpath +"'\" -f null -";
            std::cout << strcmd << std::endl;
            exec(strcmd);
        }

        cv::VideoCapture capture;
        capture = cv::VideoCapture(argv[2]);
        std::cout << "Pos: " <<capture.get(cv::CAP_PROP_POS_FRAMES) << std::endl;
        std::cout << "Length: " <<capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
        std::cout << capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
        capture = cv::VideoCapture(pathsynced[1]);
        std::cout << "Pos: " << capture.get(cv::CAP_PROP_POS_FRAMES) << std::endl;
        std::cout << "Length: " << capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    }
    return 0;
}