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

#include <arm_sve.h>

void dft4b_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1, int  *mp1) {
    float64_t a264, a265, a266, a267, a268, a269;
    int a262, a263, j1, l1, m1;
    svfloat64x2_t s103, s106, s109, s112, svex2_5, svex2_6, svex2_7, svex2_8;
    svfloat64_t s104, s105, s107, s108, s110, s111, s113, s114, 
            s115, s116, s117, s118, s119, s120, t154, t155, 
            t156, t157, t158, t159, t160, t161;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        int k1 = 0;
        pg1 = svwhilelt_b64(k1, m1);
        do {
            a262 = (k1 + ((j1)*(m1)));
            s103 = svld2_f64(pg1, (X + ((2)*(a262))));
            s104 = s103.v0;
            s105 = s103.v1;
            s106 = svld2_f64(pg1, (X + ((2)*((a262 + ((l1)*(m1)))))));
            s107 = s106.v0;
            s108 = s106.v1;
            s109 = svld2_f64(pg1, (X + ((2)*((a262 + ((((2)*(l1)))*(m1)))))));
            s110 = s109.v0;
            s111 = s109.v1;
            s112 = svld2_f64(pg1, (X + ((2)*((a262 + ((((3)*(l1)))*(m1)))))));
            s113 = s112.v0;
            s114 = s112.v1;
            t154 = svadd_f64_x(pg1, s104, s110);
            t155 = svadd_f64_x(pg1, s105, s111);
            t156 = svsub_f64_x(pg1, s104, s110);
            t157 = svsub_f64_x(pg1, s105, s111);
            t158 = svadd_f64_x(pg1, s107, s113);
            t159 = svadd_f64_x(pg1, s108, s114);
            t160 = svsub_f64_x(pg1, s107, s113);
            t161 = svsub_f64_x(pg1, s108, s114);
            s115 = svsub_f64_x(pg1, t154, t158);
            s116 = svsub_f64_x(pg1, t155, t159);
            s117 = svadd_f64_x(pg1, t156, t161);
            s118 = svsub_f64_x(pg1, t157, t160);
            s119 = svsub_f64_x(pg1, t156, t161);
            s120 = svadd_f64_x(pg1, t157, t160);
            a263 = ((6)*(j1));
            a264 = TW1[a263];
            a265 = TW1[(a263 + 1)];
            a266 = TW1[(a263 + 2)];
            a267 = TW1[(a263 + 3)];
            a268 = TW1[(a263 + 4)];
            a269 = TW1[(a263 + 5)];
            svex2_5.v0 = svadd_f64_x(pg1, t154, t158);
            svex2_5.v1 = svadd_f64_x(pg1, t155, t159);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((4)*(j1)))*(m1)))))), svex2_5);
            svex2_6.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s118, a265), s117, a264);
            svex2_6.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s118, a264), s117, a265);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((4)*(j1)))*(m1)) + m1)))), svex2_6);
            svex2_7.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s116, a267), s115, a266);
            svex2_7.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s116, a266), s115, a267);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((4)*(j1)))*(m1)) + ((2)*(m1)))))), svex2_7);
            svex2_8.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s120, a269), s119, a268);
            svex2_8.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s120, a268), s119, a269);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((4)*(j1)))*(m1)) + ((3)*(m1)))))), svex2_8);
            k1 += svcntd();
            pg1 = svwhilelt_b64(k1, m1);
        } while(svptest_any(svptrue_b64(), pg1));
    }
}
