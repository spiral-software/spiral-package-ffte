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


void dft5a_(double  *Y, double  *X, double  *TW1, int  *lp1) {
    double a449, a450, a451, a452, a453, a454, a455, a456, 
            s141, s142, s143, s144, s145, s146, s147, s148, 
            s149, s150, s151, s152, s153, s154, s155, s156, 
            s157, s158, s159, s160, s161, s162, s163, s164, 
            t394, t395, t396, t397, t398, t399, t400, t401, 
            t402, t403, t404, t405, t406, t407, t408, t409, 
            t410, t411, t412, t413, t414, t415, t416, t417;
    int a443, a444, a445, a446, a447, a448, a457, l1;
    l1 = *(lp1);
    for(int j1 = 0; j1 < l1; j1++) {
        l1 = *(lp1);
        a443 = (2*j1);
        s141 = X[a443];
        s142 = X[(a443 + 1)];
        a444 = (a443 + (2*l1));
        s143 = X[a444];
        s144 = X[(a444 + 1)];
        a445 = (a443 + (4*l1));
        s145 = X[a445];
        s146 = X[(a445 + 1)];
        a446 = (a443 + (6*l1));
        s147 = X[a446];
        s148 = X[(a446 + 1)];
        a447 = (a443 + (8*l1));
        s149 = X[a447];
        s150 = X[(a447 + 1)];
        t394 = (s143 + s149);
        t395 = (s144 + s150);
        t396 = (s143 - s149);
        t397 = (s144 - s150);
        t398 = (s145 + s147);
        t399 = (s146 + s148);
        t400 = (s145 - s147);
        t401 = (s146 - s148);
        t402 = (t394 + t398);
        t403 = (t395 + t399);
        t404 = (t396 + t401);
        t405 = (t397 - t400);
        t406 = (t396 - t401);
        t407 = (t397 + t400);
        t408 = (s141 - (0.25*t402));
        t409 = (s142 - (0.25*t403));
        s151 = ((0.29389262614623657*t404) + (0.47552825814757677*t405));
        s152 = ((0.29389262614623657*t405) - (0.47552825814757677*t404));
        s153 = (0.55901699437494745*(t394 - t398));
        s154 = (0.55901699437494745*(t395 - t399));
        s155 = ((0.47552825814757682*t407) - (0.29389262614623657*t406));
        s156 = ((0.47552825814757682*t406) + (0.29389262614623657*t407));
        t410 = (t408 + s153);
        t411 = (t409 + s154);
        t412 = (t408 - s153);
        t413 = (t409 - s154);
        t414 = (s151 + s155);
        t415 = (s152 - s156);
        t416 = (s151 - s155);
        t417 = (s152 + s156);
        s157 = (t410 + t414);
        s158 = (t411 + t415);
        s159 = (t410 - t414);
        s160 = (t411 - t415);
        s161 = (t412 + t417);
        s162 = (t413 - t416);
        s163 = (t412 - t417);
        s164 = (t413 + t416);
        a448 = (8*j1);
        a449 = TW1[a448];
        a450 = TW1[(a448 + 1)];
        a451 = TW1[(a448 + 2)];
        a452 = TW1[(a448 + 3)];
        a453 = TW1[(a448 + 4)];
        a454 = TW1[(a448 + 5)];
        a455 = TW1[(a448 + 6)];
        a456 = TW1[(a448 + 7)];
        a457 = (10*j1);
        Y[a457] = (s141 + t402);
        Y[(a457 + 1)] = (s142 + t403);
        Y[(a457 + 2)] = ((a449*s157) - (a450*s158));
        Y[(a457 + 3)] = ((a450*s157) + (a449*s158));
        Y[(a457 + 4)] = ((a451*s161) - (a452*s162));
        Y[(a457 + 5)] = ((a452*s161) + (a451*s162));
        Y[(a457 + 6)] = ((a453*s163) - (a454*s164));
        Y[(a457 + 7)] = ((a454*s163) + (a453*s164));
        Y[(a457 + 8)] = ((a455*s159) - (a456*s160));
        Y[(a457 + 9)] = ((a456*s159) + (a455*s160));
    }
}