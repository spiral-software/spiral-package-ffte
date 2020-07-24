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

void dft5c_(float64_t  *Y, float64_t  *X, int  *lp1, int  *mp1) {
    int l1, m1;
    svfloat64x2_t s172, s175, s178, s181, s184, svex2_10, svex2_6, svex2_7, 
            svex2_8, svex2_9;
    svfloat64_t s173, s174, s176, s177, s179, s180, s182, s183, 
            s185, s186, s187, s188, s189, s190, s191, s192, 
            s193, s194, s195, s196, t394, t395, t396, t397, 
            t398, t399, t400, t401, t402, t403, t404, t405, 
            t406, t407, t408, t409, t410, t411, t412, t413, 
            t414, t415, t416, t417;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    int k1 = 0;
    pg1 = svwhilelt_b64(k1, m1);
    do {
        s172 = svld2_f64(pg1, (X + ((2)*(k1))));
        s173 = s172.v0;
        s174 = s172.v1;
        s175 = svld2_f64(pg1, (X + ((2)*((k1 + ((l1)*(m1)))))));
        s176 = s175.v0;
        s177 = s175.v1;
        s178 = svld2_f64(pg1, (X + ((2)*((k1 + ((((2)*(l1)))*(m1)))))));
        s179 = s178.v0;
        s180 = s178.v1;
        s181 = svld2_f64(pg1, (X + ((2)*((k1 + ((((3)*(l1)))*(m1)))))));
        s182 = s181.v0;
        s183 = s181.v1;
        s184 = svld2_f64(pg1, (X + ((2)*((k1 + ((((4)*(l1)))*(m1)))))));
        s185 = s184.v0;
        s186 = s184.v1;
        t394 = svadd_f64_x(pg1, s176, s185);
        t395 = svadd_f64_x(pg1, s177, s186);
        t396 = svsub_f64_x(pg1, s176, s185);
        t397 = svsub_f64_x(pg1, s177, s186);
        t398 = svadd_f64_x(pg1, s179, s182);
        t399 = svadd_f64_x(pg1, s180, s183);
        t400 = svsub_f64_x(pg1, s179, s182);
        t401 = svsub_f64_x(pg1, s180, s183);
        t402 = svadd_f64_x(pg1, t394, t398);
        t403 = svadd_f64_x(pg1, t395, t399);
        t404 = svadd_f64_x(pg1, t396, t401);
        t405 = svsub_f64_x(pg1, t397, t400);
        t406 = svsub_f64_x(pg1, t396, t401);
        t407 = svadd_f64_x(pg1, t397, t400);
        t408 = svmls_n_f64_x(pg1, s173, t402, 0.25);
        t409 = svmls_n_f64_x(pg1, s174, t403, 0.25);
        s193 = svmla_n_f64_x(pg1, t404, t405, 1.6180339887498947);
        s187 = svmul_n_f64_x(pg1, s193, 0.29389262614623657);
        s194 = svmls_n_f64_x(pg1, t405, t404, 1.6180339887498947);
        s188 = svmul_n_f64_x(pg1, s194, 0.29389262614623657);
        s189 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t394, t398), 0.55901699437494745);
        s190 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t395, t399), 0.55901699437494745);
        s195 = svmls_n_f64_x(pg1, t407, t406, 0.61803398874989479);
        s191 = svmul_n_f64_x(pg1, s195, 0.47552825814757682);
        s196 = svmla_n_f64_x(pg1, t406, t407, 0.61803398874989479);
        s192 = svmul_n_f64_x(pg1, s196, 0.47552825814757682);
        t410 = svadd_f64_x(pg1, t408, s189);
        t411 = svadd_f64_x(pg1, t409, s190);
        t412 = svsub_f64_x(pg1, t408, s189);
        t413 = svsub_f64_x(pg1, t409, s190);
        t414 = svadd_f64_x(pg1, s187, s191);
        t415 = svsub_f64_x(pg1, s188, s192);
        t416 = svsub_f64_x(pg1, s187, s191);
        t417 = svadd_f64_x(pg1, s188, s192);
        svex2_6.v0 = svadd_f64_x(pg1, s173, t402);
        svex2_6.v1 = svadd_f64_x(pg1, s174, t403);
        svst2_f64(pg1, (Y + ((2)*(k1))), svex2_6);
        svex2_7.v0 = svadd_f64_x(pg1, t410, t414);
        svex2_7.v1 = svadd_f64_x(pg1, t411, t415);
        svst2_f64(pg1, (Y + ((2)*((k1 + m1)))), svex2_7);
        svex2_8.v0 = svadd_f64_x(pg1, t412, t417);
        svex2_8.v1 = svsub_f64_x(pg1, t413, t416);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((2)*(m1)))))), svex2_8);
        svex2_9.v0 = svsub_f64_x(pg1, t412, t417);
        svex2_9.v1 = svadd_f64_x(pg1, t413, t416);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((3)*(m1)))))), svex2_9);
        svex2_10.v0 = svsub_f64_x(pg1, t410, t414);
        svex2_10.v1 = svsub_f64_x(pg1, t411, t415);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((4)*(m1)))))), svex2_10);
        k1 += svcntd();
        pg1 = svwhilelt_b64(k1, m1);
    } while(svptest_any(svptrue_b64(), pg1));
}