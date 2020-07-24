/*

     FFTE: A FAST FOURIER TRANSFORM PACKAGE

     (C) COPYRIGHT SOFTWARE, 2000-2004, 2008-2014, ALL RIGHTS RESERVED
                BY
         DAISUKE TAKAHASHI
         FACULTY OF ENGINEERING, INFORMATION AND SYSTEMS
         UNIVERSITY OF TSUKUBA
         1-1-1 TENNODAI, TSUKUBA, IBARAKI 305-8573, JAPAN
         E-MAIL: daisuke@cs.tsukuba.ac.jp


     WRITTEN BY DAISUKE TAKAHASHI

     THIS KERNEL WAS GENERATED BY SPIRAL 8.2.0a03
*/


void dft6b_(double  *Y, double  *X, double  *TW1, int  *lp1, int  *mp1) {
    double a689, a690, a691, a692, a693, a694, a695, a696, 
            a697, a698, s211, s212, s213, s214, s215, s216, 
            s217, s218, s219, s220, s221, s222, s223, s224, 
            s225, s226, s227, s228, s229, s230, s231, s232, 
            s233, s234, s235, s236, s237, s238, s239, s240, 
            t494, t495, t496, t497, t498, t499, t500, t501, 
            t502, t503, t504, t505, t506, t507, t508, t509, 
            t510, t511, t512, t513;
    int a680, a681, a682, a683, a684, a685, a686, a687, 
            a688, a699, a700, a701, a702, a703, a704, a705, 
            b90, j1, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        for(int k1 = 0; k1 < m1; k1++) {
            a680 = (2*k1);
            a681 = (j1*m1);
            a682 = (a680 + (2*a681));
            s211 = X[a682];
            s212 = X[(a682 + 1)];
            b90 = (l1*m1);
            a683 = (a682 + (2*b90));
            s213 = X[a683];
            s214 = X[(a683 + 1)];
            a684 = (a682 + (4*b90));
            s215 = X[a684];
            s216 = X[(a684 + 1)];
            a685 = (a682 + (6*b90));
            s217 = X[a685];
            s218 = X[(a685 + 1)];
            a686 = (a682 + (8*b90));
            s219 = X[a686];
            s220 = X[(a686 + 1)];
            a687 = (a682 + (10*b90));
            s221 = X[a687];
            s222 = X[(a687 + 1)];
            t494 = (s215 + s219);
            t495 = (s216 + s220);
            t496 = (s211 + t494);
            t497 = (s212 + t495);
            t498 = (s211 - (0.5*t494));
            t499 = (s212 - (0.5*t495));
            s223 = (0.8660254037844386*(s216 - s220));
            s224 = (0.8660254037844386*(s215 - s219));
            t500 = (t498 + s223);
            t501 = (t499 - s224);
            t502 = (t498 - s223);
            t503 = (t499 + s224);
            t504 = (s217 + s221);
            t505 = (s218 + s222);
            t506 = (s213 + t504);
            t507 = (s214 + t505);
            t508 = (s213 - (0.5*t504));
            t509 = (s214 - (0.5*t505));
            s225 = (0.8660254037844386*(s218 - s222));
            s226 = (0.8660254037844386*(s217 - s221));
            t510 = (t508 + s225);
            t511 = (t509 - s226);
            t512 = (t508 - s225);
            t513 = (t509 + s226);
            s227 = ((0.5*t510) + (0.8660254037844386*t511));
            s228 = ((0.5*t511) - (0.8660254037844386*t510));
            s229 = ((0.8660254037844386*t513) - (0.5*t512));
            s230 = ((0.8660254037844386*t512) + (0.5*t513));
            s231 = (t496 - t506);
            s232 = (t497 - t507);
            s233 = (t500 + s227);
            s234 = (t501 + s228);
            s235 = (t500 - s227);
            s236 = (t501 - s228);
            s237 = (t502 + s229);
            s238 = (t503 - s230);
            s239 = (t502 - s229);
            s240 = (t503 + s230);
            a688 = (10*j1);
            a689 = TW1[a688];
            a690 = TW1[(a688 + 1)];
            a691 = TW1[(a688 + 2)];
            a692 = TW1[(a688 + 3)];
            a693 = TW1[(a688 + 4)];
            a694 = TW1[(a688 + 5)];
            a695 = TW1[(a688 + 6)];
            a696 = TW1[(a688 + 7)];
            a697 = TW1[(a688 + 8)];
            a698 = TW1[(a688 + 9)];
            a699 = (12*a681);
            a700 = (a680 + a699);
            Y[a700] = (t496 + t506);
            Y[(a700 + 1)] = (t497 + t507);
            a701 = (a680 + (2*m1) + a699);
            Y[a701] = ((a689*s233) - (a690*s234));
            Y[(a701 + 1)] = ((a690*s233) + (a689*s234));
            a702 = (a680 + (4*m1) + a699);
            Y[a702] = ((a691*s237) - (a692*s238));
            Y[(a702 + 1)] = ((a692*s237) + (a691*s238));
            a703 = (a680 + (6*m1) + a699);
            Y[a703] = ((a693*s231) - (a694*s232));
            Y[(a703 + 1)] = ((a694*s231) + (a693*s232));
            a704 = (a680 + (8*m1) + a699);
            Y[a704] = ((a695*s235) - (a696*s236));
            Y[(a704 + 1)] = ((a696*s235) + (a695*s236));
            a705 = (a680 + (10*m1) + a699);
            Y[a705] = ((a697*s239) - (a698*s240));
            Y[(a705 + 1)] = ((a698*s239) + (a697*s240));
        }
    }
}
