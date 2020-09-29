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
        auto start = std::chrono::system_clock::now();
        int arg = argc - 1;
        std::cout << "V2: " << std::endl;
        std::cout << "Arguments: " << argc - 1 << std::endl;

        const long window = 250; //Fenêtre de recherche intercorrelation

        std::vector< std::vector<float> > corr;
        corr.reserve(argc - 1);
        for (int i = 0; i < argc - 1; i++) {
            corr.push_back(std::vector<float>(window));
        }

        //Get mean luma of each frame for future cross-correlation
        int fps;
        cv::VideoCapture cap;
        for (int i = 0; i < argc - 1; i++) {
            cap = cv::VideoCapture(argv[i + 1]);
            if (i == 0) {
                fps = cap.get(cv::CAP_PROP_FPS);
            }

            for (int j = 0; cap.isOpened() && j < window; j++) {
                cv::Mat img;
                cap >> img;
                //cv::imshow("test", img);
                //cv::waitKey(0);
                if (j == 0) {
                    std::cout << "Pos: " << cap.get(cv::CAP_PROP_POS_FRAMES) << std::endl;
                }
                cv::Scalar m = cv::sum(img);
                //std::cout <<"n°"<<j<<": " << m[0] << std::endl;
                corr[i][j] = m[0] + m[1] + m[2]; // A faire: Vérifier qu'on utilise le vert.
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
                    //std::cout << "Max: " << coefcorr[j] << std::endl;
                }
                //std::cout << "Index: "<< j <<" , Value: " << coefcorr[j] << std::endl;
            }
            //std::cout << "-----------------------------" << std::endl;
            std::cout << "Delay: " << index - window / 2 << ", MaxCoef: " << maxcorr << std::endl;
            delay.push_back(index - (window / 2));
        }
        std::vector<std::string> pathsynced;
        for (int i = 1; i < delay.size(); i++) {
            std::string comps = argv[i + 1];
            std::string ref = argv[1];
            std::string str = argv[i + 1];
            std::string name = comps.substr(comps.find_last_of("/\\") + 1);
            size_t p = name.find_last_of('.');
            name = name.substr(0, p);
            name = name  + ".log";
            float delay_pts = (float) delay[i] / fps;
            std::cout << fps << std::endl;
            std::cout << delay_pts << std::endl;
            std::string strcmd = "ffmpeg -i " + comps + " -i " + ref + " -lavfi \"[0:v]settb=AVTB,setpts=PTS-" + std::to_string(delay_pts) +"/TB[main];[1:v]settb=AVTB,setpts=PTS[ref];[main][ref] psnr=\"stats_file='[PSNR]" + name + "'\"\" -f null -";
            std::cout << strcmd << std::endl;
            exec(strcmd);
            strcmd = "ffmpeg -i " + comps + " -i " + ref + " -lavfi \"[0:v]settb=AVTB,setpts=PTS-" + std::to_string(delay_pts) + "/TB[main];[1:v]settb=AVTB,setpts=PTS[ref];[main][ref] ssim=\"stats_file='[SSIM]" + name + "'\"\" -f null -";
            std::cout << strcmd << std::endl;
            exec(strcmd);
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << std::endl;
    }
    return 0;
}