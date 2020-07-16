#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>

std::vector<std::vector<float>> res(1000, std::vector<float>(3)); // global 2d vector
std::mutex myMutex;

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

double getPSNR(const cv::Mat& I1, const cv::Mat& I2, const int x, const int y)
{
    cv::Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    cv::Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) { // for small values return zero
        res[x][y] = 0;
        return 0;
    }
    else
    {
        double  mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        std::cout << "PSNR" << ": " << psnr << std::endl;
        std::lock_guard<std::mutex> guard(myMutex);
        res[x][y] = psnr;
        return psnr;
    }
}

cv::Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2 = I2.mul(I2);        // I2^2
    cv::Mat I1_2 = I1.mul(I1);        // I1^2
    cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2

    /***********************PRELIMINARY COMPUTING ******************************/

    cv::Mat mu1, mu2;   //
    GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    cv::Scalar mssim = mean(ssim_map); // mssim = average of ssim map
    return mssim;
}

int main(int argc, const char* argv[])
{
    if (argc > 1) {
        int arg = argc - 1;
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

        std::vector<long> delay;
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
        //Synchro Ref and comps
        std::vector<cv::VideoCapture> vcap;
        //vcap.reserve(10000);
        vcap.reserve(argc - 1);
        for (int i = 0; i < argc - 1; i++) {
            std::cout << "I vcap: " << argc - 1 << std::endl;
            cv::VideoCapture tmpcap = cv::VideoCapture(argv[i + 1]);
            std::cout << "sync delay" << i << ": " << delay[i] << std::endl;
            vcap.push_back(tmpcap);
            for (int j = 0; j <= delay[i];j++) {
                cv::Mat img;
                vcap[i] >> img;
            }
        }

        auto start = std::chrono::system_clock::now();

        bool isplaying = true;
        //std::vector< std::vector<float> > res;
        int framecount = 0;
        while (isplaying) {
            std::vector<cv::Mat> image;
            image.reserve(argc - 1);
            for (int i = 0; i < argc - 1; i++) {
                cv::Mat img;
                vcap[i] >> img;
                //if (img.empty() || framecount == 30) {
                if (img.empty()) {      //detect the end of a file
                    isplaying = false;
                }
                image.push_back(img);
            }
            //std::vector<float> resrow;
            for (int i = 1; i < argc - 1 && isplaying; i++) {
                //cv::imshow(std::to_string(i),image[i]);
                std::thread{ getPSNR, image[0], image[i], framecount, i }.detach(); //compute psnr through thread for more efficiency.
             //float resPSNR =  getPSNR(image[0], image[i]);
             //resrow.push_back(resPSNR);
            }
            //res.push_back(resrow);
            framecount++;
        }
        /*int reslength = res.size();
        std::cout << std::to_string(reslength) << std::endl;
        std::cout << std::to_string(res[10].size()) << std::endl;
        std::cout << "writting CSV" << std::endl;
        std::ofstream myfile;
        myfile.open("example.csv");
        std::string firstline;
        for (int i = 0; i < argc - 2;i++) {
            firstline = firstline + "Frame,PSNR,";
        }
        firstline = firstline + "\n";
        myfile << firstline;

        std::cout << reslength << std::endl;
        for (int i = 0; i < reslength-1;i++) {
            std::string line;
            for (int j = 0; j < argc - 2;j++) {
                line = line + std::to_string(i) +","+ std::to_string(res[i][j]) + ",";
            }
            line = line + "\n";
            myfile << line;
        }
        myfile.close();*/
    }
    return 0;
}