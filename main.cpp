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

#include <nlohmann/json.hpp>
using json = nlohmann::json;

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
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(charcmd, "r"), pclose);
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
    auto start = std::chrono::system_clock::now(); //QStart Chrono
    std::cout << "V3: " << std::endl;

    if (argc == 2) {
        json json = json::parse(argv[1]);  // Parse json passed as argument
        std::string ref = json["ref"];     //
        //std::string converted = json["converted"];
        unsigned int window = 1200;
        if (!json["window"].is_null()){
            window = json["window"];
        }
        std::cout <<"Search window: " << window << std::endl;

        std::vector<std::string> comps = json["comps"];
        std::vector< std::vector<float> > corr;
        corr.reserve(json["comps"].size()+1);
        for (unsigned int i = 0; i < json["comps"].size()+1; i++) {
            corr.push_back(std::vector<float>(window));
        }

        std::cout << "Start delay synchronize" << std::endl;
        //Get mean luma of each frame for future cross-correlation
        uint8_t fps;
        cv::VideoCapture cap;

        for (unsigned int i = 0; i < json["comps"].size()+1; i++) {
            if (i == 0) {
                cap = cv::VideoCapture(ref);
                fps = cap.get(cv::CAP_PROP_FPS); //Get framerate. Needed later.
            } else {
                cap = cv::VideoCapture(comps[i-1]);
            }
            for (unsigned int j = 0; cap.isOpened() && j < window; j++) {
                cv::Mat img;
                cap >> img;
                cv::Mat cropimg;
                //img(cv::Rect(cv::Point(0, 0), cv::Point(100, 100))).copyTo(cropimg);
                //cv::imshow("test", img);
                //cv::waitKey(0);

                cv::Scalar m = cv::sum(img);
                //std::cout <<"n°"<<j<<": " << m[0] << std::endl;
                //corr[i][j] = m[0] + m[1] + m[2]; // A faire: Vérifier qu'on utilise le vert.
                corr[i][j] = m[0]; // A faire: Vérifier qu'on utilise le vert.
            }
        }
        //Luma mean for reference
        double corref[window];
        for (unsigned int j = 0; j < window; j++) {
            corref[j] = corr[0][j];
        }

        std::cout <<"Results:" << std::endl;
        std::vector<int> delay;
        delay.reserve(json["comps"].size()+1);
        delay.push_back(0); //Set delay for ref at [0]
        //Cross-correlation for each comps
        for (unsigned int i = 0; i < json["comps"].size(); i++) {
            double comps[window]; //Luma mean for comps
            double coefcorr[window];   //Coefficients of cross-corr
            for (unsigned int j = 0; j < window; j++) {
                comps[j] = corr[i + 1][j];
                coefcorr[j] = 0; //Initialize array
            }
            correlation(corref, comps, window, coefcorr); //See cross-correlation function

            long index;
            double maxcorr = -2; // correlaration between -1 and 1
            for (unsigned int j = 0; j < window; j++) {
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

        std::cout << "Start compute PSNR&SSIM by ffmpeg" << std::endl;
        for (unsigned long i = 1; i < delay.size(); i++) {
            std::string comp = comps[i-1];
            std::string name = comp.substr(comp.find_last_of("/\\") + 1);
            size_t p = name.find_last_of('.');
            name = name.substr(0, p);
            name = name  + ".log";
            float delaycomp_pts = 0;
            float delayref_pts =  0;
            if ((float) delay[i] / fps >=0) {
                delaycomp_pts = (float) delay[i] / fps;
                delayref_pts = 0;
            } else {
                delaycomp_pts = 0;
                delayref_pts = std::abs((float) delay[i] / fps);
            }

            //std::cout <<"Framerate: " << fps << std::endl;
            //std::cout <<"Delay compspts: " << delaycomp_pts << ", delay refpst: " << delayref_pts << std::endl;
            //std::string strcmd = "ffmpeg -y -nostats -loglevel 0 -i " + comp + " -i " + ref + " -lavfi \"[0:v]settb=AVTB,setpts=PTS-" + std::to_string(delaycomp_pts) +"/TB[main];[1:v]settb=AVTB,setpts=PTS-" + std::to_string(delayref_pts) +"/TB[ref];[main][ref] psnr=stats_file='[PSNR]" + name + "'\" -f null -";
            std::string strcmd = "ffmpeg -i " + comp + " -i " + ref + " -lavfi \"[0:v]settb=AVTB,setpts=PTS-" + std::to_string(delaycomp_pts) +"/TB[main];[1:v]settb=AVTB,setpts=PTS-" + std::to_string(delayref_pts) +"/TB[ref];[main][ref] psnr=stats_file='[PSNR]" + name + "'\" -f null -";
            //std::cout << strcmd << std::endl;
            exec(strcmd);
            strcmd = "ffmpeg -y -nostats -loglevel 0 -i " + comp + " -i " + ref + " -lavfi \"[0:v]settb=AVTB,setpts=PTS-" + std::to_string(delaycomp_pts) + "/TB[main];[1:v]settb=AVTB,setpts=PTS-" + std::to_string(delayref_pts) +"/TB[ref];[main][ref] ssim=stats_file='[SSIM]" + name + "'\" -f null -";
            //std::cout << strcmd << std::endl;
            exec(strcmd);
            std::cout << std::to_string(i)+"/" + std::to_string(delay.size()-1) << " completed." <<std::endl;
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << std::endl;

        for (int i=0; i < (int) delay.size();i++){
            if(i==0) cv::VideoCapture cap = cv::VideoCapture(ref);
            else cv::VideoCapture cap = cv::VideoCapture(comps[i-1]);
            cv::Mat img;
            for(int j=0; j < delay[i]; j++){
                cap >> img;
            }
            if(i==0){
            std::string comp = ref;
            std::string name = comp.substr(comp.find_last_of("/\\") + 1);
            size_t p = name.find_last_of('.');
            name = name.substr(0, p);
            name = name  + ".jpg";
            cv::imwrite(name, img );
            } else{
            std::string comp = comps[i-1];
            std::string name = comp.substr(comp.find_last_of("/\\") + 1);
            size_t p = name.find_last_of('.');
            name = name.substr(0, p);
            name = name  + ".jpg";
            cv::imwrite(name, img );
            }
        }
    }

    if (argc == 4) {
        std::cout << "No json sended" << std::endl;
        std::string ref = argv[1];     //
        unsigned int window = std::stoi(argv[3]);

        std::cout <<"Search window: " << window << std::endl;
        //const char comp[] = argv[2];
        std::vector<std::string> comps = {argv[2]};
        std::vector< std::vector<float> > corr;
        corr.reserve(2);
        for (unsigned int i = 0; i < 2; i++) {
            corr.push_back(std::vector<float>(window));
        }

        std::cout << "Start delay synchronize" << std::endl;
        //Get mean luma of each frame for future cross-correlation
        uint8_t fps;
        cv::VideoCapture cap;

        for (unsigned int i = 0; i < 2; i++) {
            if (i == 0) {
                cap = cv::VideoCapture(ref);
                fps = cap.get(cv::CAP_PROP_FPS); //Get framerate. Needed later.
            } else {
                cap = cv::VideoCapture(comps[i-1]);
            }
            for (unsigned int j = 0; cap.isOpened() && j < window; j++) {
                cv::Mat img;
                cap >> img;
                cv::Mat cropimg;
                //img(cv::Rect(cv::Point(0, 0), cv::Point(100, 100))).copyTo(cropimg);
                //cv::imshow("test", img);
                //cv::waitKey(0);

                cv::Scalar m = cv::sum(img);
                //std::cout <<"n°"<<j<<": " << m[0] << std::endl;
                corr[i][j] = m[0] + m[1] + m[2]; // A faire: Vérifier qu'on utilise le vert.
                //corr[i][j] = m[0]; // A faire: Vérifier qu'on utilise le vert.
            }
        }
        //Luma mean for reference
        double corref[window];
        for (unsigned int j = 0; j < window; j++) {
            corref[j] = corr[0][j];
        }

        std::cout <<"Results:" << std::endl;
        std::vector<int> delay;
        delay.reserve(2+1);
        delay.push_back(0); //Set delay for ref at [0]
        //Cross-correlation for each comps
        for (unsigned int i = 0; i < 2; i++) {
            double comps[window]; //Luma mean for comps
            double coefcorr[window];   //Coefficients of cross-corr
            for (unsigned int j = 0; j < window; j++) {
                comps[j] = corr[i + 1][j];
                coefcorr[j] = 0; //Initialize array
            }
            correlation(corref, comps, window, coefcorr); //See cross-correlation function

            long index;
            double maxcorr = -2; // correlaration between -1 and 1
            for (unsigned int j = 0; j < window; j++) {
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

        std::cout << "Start compute PSNR&SSIM by ffmpeg" << std::endl;
        for (unsigned long i = 1; i < delay.size(); i++) {
            std::string comp = comps[i-1];
            std::string name = comp.substr(comp.find_last_of("/\\") + 1);
            size_t p = name.find_last_of('.');
            name = name.substr(0, p);
            name = name  + ".log";
            float delaycomp_pts = 0;
            float delayref_pts =  0;
            if ((float) delay[i] / fps >=0) {
                delaycomp_pts = (float) delay[i] / fps;
                delayref_pts = 0;
            } else {
                delaycomp_pts = 0;
                delayref_pts = std::abs((float) delay[i] / fps);
            }

            //std::cout <<"Framerate: " << fps << std::endl;
            //std::cout <<"Delay compspts: " << delaycomp_pts << ", delay refpst: " << delayref_pts << std::endl;
            //std::string strcmd = "ffmpeg -y -nostats -loglevel 0 -i " + comp + " -i " + ref + " -lavfi \"[0:v]settb=AVTB,setpts=PTS-" + std::to_string(delaycomp_pts) +"/TB[main];[1:v]settb=AVTB,setpts=PTS-" + std::to_string(delayref_pts) +"/TB[ref];[main][ref] psnr=stats_file='[PSNR]" + name + "'\" -f null -";
            std::string strcmd = "ffmpeg -i " + comp + " -i " + ref + " -lavfi \"[0:v]settb=AVTB,setpts=PTS-" + std::to_string(delaycomp_pts) +"/TB[main];[1:v]settb=AVTB,setpts=PTS-" + std::to_string(delayref_pts) +"/TB[ref];[main][ref] psnr=stats_file='[PSNR]" + name + "'\" -f null -";
            //std::cout << strcmd << std::endl;
            exec(strcmd);
            strcmd = "ffmpeg -y -nostats -loglevel 0 -i " + comp + " -i " + ref + " -lavfi \"[0:v]settb=AVTB,setpts=PTS-" + std::to_string(delaycomp_pts) + "/TB[main];[1:v]settb=AVTB,setpts=PTS-" + std::to_string(delayref_pts) +"/TB[ref];[main][ref] ssim=stats_file='[SSIM]" + name + "'\" -f null -";
            //std::cout << strcmd << std::endl;
            exec(strcmd);
            std::cout << std::to_string(i)+"/" + std::to_string(delay.size()-1) << " completed." <<std::endl;
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << std::endl;

        for (int i=0; i < (int) delay.size();i++){
            if(i==0) cv::VideoCapture cap = cv::VideoCapture(ref);
            else cv::VideoCapture cap = cv::VideoCapture(comps[i-1]);
            cv::Mat img;
            for(int j=0; j < delay[i]; j++){
                cap >> img;
            }
            if(i==0){
            std::string comp = ref;
            std::string name = comp.substr(comp.find_last_of("/\\") + 1);
            size_t p = name.find_last_of('.');
            name = name.substr(0, p);
            name = name  + ".jpg";
            cv::imwrite(name, img );
            } else{
            std::string comp = comps[i-1];
            std::string name = comp.substr(comp.find_last_of("/\\") + 1);
            size_t p = name.find_last_of('.');
            name = name.substr(0, p);
            name = name  + ".jpg";
            cv::imwrite(name, img );
            }
        }
    }


    else {
        std::cout << "Missing ref's path and comps's path as arguments" << std::endl;
    }
    return 0;
}
