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


void dft3b_(double  *Y, double  *X, double  *TW1, int  *lp1, int  *mp1) {
    double a279, a280, a281, a282, s65, s66, s67, s68, 
            s69, s70, s71, s72, s73, s74, s75, s76, 
            t125, t126, t127, t128;
    int a273, a274, a275, a276, a277, a278, a283, a284, 
            a285, a286, b45, j1, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        for(int k1 = 0; k1 < m1; k1++) {
            a273 = (2*k1);
            a274 = (j1*m1);
            a275 = (a273 + (2*a274));
            s65 = X[a275];
            s66 = X[(a275 + 1)];
            b45 = (l1*m1);
            a276 = (a275 + (2*b45));
            s67 = X[a276];
            s68 = X[(a276 + 1)];
            a277 = (a275 + (4*b45));
            s69 = X[a277];
            s70 = X[(a277 + 1)];
            t125 = (s67 + s69);
            t126 = (s68 + s70);
            t127 = (s65 - (0.5*t125));
            t128 = (s66 - (0.5*t126));
            s71 = (0.8660254037844386*(s68 - s70));
            s72 = (0.8660254037844386*(s67 - s69));
            s73 = (t127 + s71);
            s74 = (t128 - s72);
            s75 = (t127 - s71);
            s76 = (t128 + s72);
            a278 = (4*j1);
            a279 = TW1[a278];
            a280 = TW1[(a278 + 1)];
            a281 = TW1[(a278 + 2)];
            a282 = TW1[(a278 + 3)];
            a283 = (6*a274);
            a284 = (a273 + a283);
            Y[a284] = (s65 + t125);
            Y[(a284 + 1)] = (s66 + t126);
            a285 = (a273 + (2*m1) + a283);
            Y[a285] = ((a279*s73) - (a280*s74));
            Y[(a285 + 1)] = ((a280*s73) + (a279*s74));
            a286 = (a273 + (4*m1) + a283);
            Y[a286] = ((a281*s75) - (a282*s76));
            Y[(a286 + 1)] = ((a282*s75) + (a281*s76));
        }
    }
}