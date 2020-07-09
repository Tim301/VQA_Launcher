#include <iostream>
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>

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

bool isopen(std::vector<cv::VideoCapture> media) {
    for (int i = 0; i < media.size(); i++) {
        if (media[i].isOpened() != true ) {
            return false;
        }
    }
    return true;
}

void applyGaussianBlur(const cv::Mat& src, cv::Mat& dst, int ksize, double sigma)
{
    int invalid = (ksize - 1) / 2;
    cv::Mat tmp(src.rows, src.cols, CV_32F);
    cv::GaussianBlur(src, tmp, cv::Size(ksize, ksize), sigma);
    tmp(cv::Range(invalid, tmp.rows - invalid), cv::Range(invalid, tmp.cols - invalid)).copyTo(dst);
}

double getPSNR(const cv::Mat& I1, const cv::Mat& I2)
{
    cv::Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    cv::Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
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

cv::Scalar computeSSIM(const cv::Mat& img1, const cv::Mat& img2)
{
    const float C1 = 6.5025f;
    const float C2 = 58.5225f;

    int ht = img1.rows;
    int wt = img1.cols;
    int w = wt - 10;
    int h = ht - 10;

    cv::Mat mu1(h, w, CV_32F), mu2(h, w, CV_32F);
    cv::Mat mu1_sq(h, w, CV_32F), mu2_sq(h, w, CV_32F), mu1_mu2(h, w, CV_32F);
    cv::Mat img1_sq(ht, wt, CV_32F), img2_sq(ht, wt, CV_32F), img1_img2(ht, wt, CV_32F);
    cv::Mat sigma1_sq(h, w, CV_32F), sigma2_sq(h, w, CV_32F), sigma12(h, w, CV_32F);
    cv::Mat tmp1(h, w, CV_32F), tmp2(h, w, CV_32F), tmp3(h, w, CV_32F);
    cv::Mat ssim_map(h, w, CV_32F), cs_map(h, w, CV_32F);

    // mu1 = filter2(window, img1, 'valid');
    applyGaussianBlur(img1, mu1, 11, 1.5);

    // mu2 = filter2(window, img2, 'valid');
    applyGaussianBlur(img2, mu2, 11, 1.5);

    // mu1_sq = mu1.*mu1;
    cv::multiply(mu1, mu1, mu1_sq);
    // mu2_sq = mu2.*mu2;
    cv::multiply(mu2, mu2, mu2_sq);
    // mu1_mu2 = mu1.*mu2;
    cv::multiply(mu1, mu2, mu1_mu2);

    cv::multiply(img1, img1, img1_sq);
    cv::multiply(img2, img2, img2_sq);
    cv::multiply(img1, img2, img1_img2);

    // sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
    applyGaussianBlur(img1_sq, sigma1_sq, 11, 1.5);
    sigma1_sq -= mu1_sq;

    // sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
    applyGaussianBlur(img2_sq, sigma2_sq, 11, 1.5);
    sigma2_sq -= mu2_sq;

    // sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
    applyGaussianBlur(img1_img2, sigma12, 11, 1.5);
    sigma12 -= mu1_mu2;

    // cs_map = (2*sigma12 + C2)./(sigma1_sq + sigma2_sq + C2);
    tmp1 = 2 * sigma12 + C2;
    tmp2 = sigma1_sq + sigma2_sq + C2;
    cv::divide(tmp1, tmp2, cs_map);
    // ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
    tmp3 = 2 * mu1_mu2 + C1;
    cv::multiply(tmp1, tmp3, tmp1);
    tmp3 = mu1_sq + mu2_sq + C1;
    cv::multiply(tmp2, tmp3, tmp2);
    cv::divide(tmp1, tmp2, ssim_map);

    // mssim = mean2(ssim_map);
    double mssim = cv::mean(ssim_map).val[0];
    // mcs = mean2(cs_map);
    double mcs = cv::mean(cs_map).val[0];

    cv::Scalar res(mssim, mcs);

    return res;
}

double cov(cv::Mat& m0, cv::Mat& m1, int i, int j, int block_size) {
    cv::Mat m3 = cv::Mat::zeros(block_size, block_size, m0.depth());
    cv::Mat m0_tmp = m0(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
    cv::Mat m1_tmp = m1(cv::Range(i, i + block_size), cv::Range(j, j + block_size));

    multiply(m0_tmp, m1_tmp, m3);

    double avg_ro = mean(m3)[0]; // E(XY)
    double avg_r = mean(m0_tmp)[0]; // E(X)
    double avg_o = mean(m1_tmp)[0]; // E(Y)

    double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)
    return sd_ro;
}

double sigma(cv::Mat& m, int i, int j, int block_size) {
    double sd = 0;

    cv::Mat m_tmp = m(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
    cv::Mat m_squared(block_size, block_size, CV_64F);

    multiply(m_tmp, m_tmp, m_squared);

    // E(x)
    double avg = mean(m_tmp)[0];
    // E(x²)
    double avg_2 = cv::mean(m_squared)[0];

    sd = sqrt(avg_2 - avg * avg);
    return sd;
}

double ssim(cv::Mat& m0, cv::Mat& m1, int block_size) {
#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 255)
    bool mask = false;
    double ssim = 0;

    int nbBlockPerHeight = m0.rows / block_size;
    int nbBlockPerWidth = m0.cols / block_size;
    double ssim_total = 0;

#pragma omp parallel for
    for (int k = 0; k < nbBlockPerHeight; k++) {
        for (int l = 0; l < nbBlockPerWidth; l++) {
            int m = k * block_size;
            int n = l * block_size;

            double avg_o = cv::mean(m0(cv::Range(k, k + block_size), cv::Range(l, l + block_size)))[0];
            double avg_r = cv::mean(m1(cv::Range(k, k + block_size), cv::Range(l, l + block_size)))[0];
            if (mask) {
                if (avg_o > 254 || avg_o < 1) {
                    continue;
                }
                ssim_total += 1;
            }
            double sigma_o = sigma(m0, m, n, block_size);
            double sigma_r = sigma(m1, m, n, block_size);
            double sigma_ro = cov(m0, m1, m, n, block_size);

            ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));
        }
    }

    ssim /= mask ? ssim_total : nbBlockPerHeight * nbBlockPerWidth;
    return ssim;
}

int main(int argc, const char* argv[])
{
    if (argc > 1) {
        int arg = argc - 1;
        std::cout << "Arguments: " << argc - 1 << std::endl;
        const long window = 1200; //Fenêtre de recherche intercorrelation

        //Initialize 2D Array [nb_media][fenetrederecherche]
        /*float corry = (float) argc - 1;
        float**corr = new float*[corry];
        for (int i = 0; i < argc - 1; i++) {
            corr[i] = new float[window];
        }*/

        std::vector< std::vector<float> > corr;
        corr.reserve(argc - 1);
        for (int i = 0; i < argc - 1; i++) {
            corr.push_back(std::vector<float>(window));
        }

        //Get mean luma of each frame for future cross-correlation
        cv::VideoCapture cap;
        for (int i = 0; i < argc-1; i++) {
            cap = cv::VideoCapture(argv[i+1]);
            for (int j = 0; cap.isOpened() && j< window; j++) {
                cv::Mat img;
                cap >> img;
                cv::Scalar m = cv::mean(img);
                //std::cout <<"n°"<<j<<": " << m[0] << std::endl;
                corr[i][j] = (double) m[0]; //Definir channal de luma à utiliser.
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
                comps[j] = corr[i+1][j];
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
            std::cout << "Delay: " << index - window/2 << ", MaxCoef: " << maxcorr << std::endl;
            delay.push_back(index - (window / 2));  
        }
        //Synchro Ref and comps
        std::vector<cv::VideoCapture> vcap;
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

        while (isopen(vcap)) {
            std::vector<cv::Mat> image;
            std::cout << "IN" << std::endl;
            image.reserve(argc - 1);
            for (int i = 0; i < argc - 1; i++) {
                cv::Mat img;
                vcap[i] >> img;
                image.push_back(img);
            }
            for (int i = 1; i < argc - 1; i++) {
                //cv::imshow(std::to_string(i),image[i]);
                //float res = computeSSIM(image[0], image[i]).val[0];
                float resSSIM =  getMSSIM(image[0], image[i]).val[0];
                float resPSNR =  getPSNR(image[0], image[i]);
                //double resSSIM =  ssim(image[0], image[i], 1);
                //std::cout << "SSIM" << std::to_string(i) << ": " << std::to_string(res) << std::endl;
                std::cout << "SSIM" << std::to_string(i) << ": " << resSSIM << std::endl;
                std::cout << "PSNR" << std::to_string(i) << ": " << resPSNR << std::endl;
            }
            //cv::waitKey(1);
        }

    }
    return 0;
}
