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

void dft6a_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1) {
    int l1;
    float64_t  *a518;
    svfloat64x2_t s289, s292, s295, s298, s301, s304;
    svfloat64_t a519, a520, a521, a522, a523, a524, a525, a526, 
            a527, a528, s290, s291, s293, s294, s296, s297, 
            s299, s300, s302, s303, s305, s306, s307, s308, 
            s309, s310, s311, s312, s313, s314, s315, s316, 
            s317, s318, s319, s320, s321, s322, s323, s324, 
            s325, s326, s327, s328, s329, s330, s331, s332, 
            s333, s334, s335, s336, s337, s338, s339, s340, 
            t495, t496, t497, t498, t499, t500, t501, t502, 
            t503, t504, t505, t506, t507, t508, t509, t510, 
            t511, t512, t513, t514;
    svbool_t pg1;
    l1 = *(lp1);
    int j1 = 0;
    pg1 = svwhilelt_b64(j1, l1);
    do {
        s289 = svld2_f64(pg1, (X + ((2)*(j1))));
        s290 = s289.v0;
        s291 = s289.v1;
        s292 = svld2_f64(pg1, (X + ((2)*((j1 + l1)))));
        s293 = s292.v0;
        s294 = s292.v1;
        s295 = svld2_f64(pg1, (X + ((2)*((j1 + ((2)*(l1)))))));
        s296 = s295.v0;
        s297 = s295.v1;
        s298 = svld2_f64(pg1, (X + ((2)*((j1 + ((3)*(l1)))))));
        s299 = s298.v0;
        s300 = s298.v1;
        s301 = svld2_f64(pg1, (X + ((2)*((j1 + ((4)*(l1)))))));
        s302 = s301.v0;
        s303 = s301.v1;
        s304 = svld2_f64(pg1, (X + ((2)*((j1 + ((5)*(l1)))))));
        s305 = s304.v0;
        s306 = s304.v1;
        t495 = svadd_f64_x(pg1, s296, s302);
        t496 = svadd_f64_x(pg1, s297, s303);
        t497 = svadd_f64_x(pg1, s290, t495);
        t498 = svadd_f64_x(pg1, s291, t496);
        t499 = svmls_n_f64_x(pg1, s290, t495, 0.5);
        t500 = svmls_n_f64_x(pg1, s291, t496, 0.5);
        s307 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s297, s303), 0.8660254037844386);
        s308 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s296, s302), 0.8660254037844386);
        t501 = svadd_f64_x(pg1, t499, s307);
        t502 = svsub_f64_x(pg1, t500, s308);
        t503 = svsub_f64_x(pg1, t499, s307);
        t504 = svadd_f64_x(pg1, t500, s308);
        t505 = svadd_f64_x(pg1, s299, s305);
        t506 = svadd_f64_x(pg1, s300, s306);
        t507 = svadd_f64_x(pg1, s293, t505);
        t508 = svadd_f64_x(pg1, s294, t506);
        t509 = svmls_n_f64_x(pg1, s293, t505, 0.5);
        t510 = svmls_n_f64_x(pg1, s294, t506, 0.5);
        s309 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s300, s306), 0.8660254037844386);
        s310 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s299, s305), 0.8660254037844386);
        t511 = svadd_f64_x(pg1, t509, s309);
        t512 = svsub_f64_x(pg1, t510, s310);
        t513 = svsub_f64_x(pg1, t509, s309);
        t514 = svadd_f64_x(pg1, t510, s310);
        s337 = svmla_n_f64_x(pg1, t511, t512, 1.7320508075688772);
        s311 = svmul_n_f64_x(pg1, s337, 0.5);
        s338 = svmls_n_f64_x(pg1, t512, t511, 1.7320508075688772);
        s312 = svmul_n_f64_x(pg1, s338, 0.5);
        s339 = svmls_n_f64_x(pg1, t514, t513, 0.57735026918962584);
        s313 = svmul_n_f64_x(pg1, s339, 0.8660254037844386);
        s340 = svmla_n_f64_x(pg1, t513, t514, 0.57735026918962584);
        s314 = svmul_n_f64_x(pg1, s340, 0.8660254037844386);
        s315 = svadd_f64_x(pg1, t497, t507);
        s316 = svadd_f64_x(pg1, t498, t508);
        s317 = svsub_f64_x(pg1, t497, t507);
        s318 = svsub_f64_x(pg1, t498, t508);
        s319 = svadd_f64_x(pg1, t501, s311);
        s320 = svadd_f64_x(pg1, t502, s312);
        s321 = svsub_f64_x(pg1, t501, s311);
        s322 = svsub_f64_x(pg1, t502, s312);
        s323 = svadd_f64_x(pg1, t503, s313);
        s324 = svsub_f64_x(pg1, t504, s314);
        s325 = svsub_f64_x(pg1, t503, s313);
        s326 = svadd_f64_x(pg1, t504, s314);
        a518 = (TW1 + ((10)*(j1)));
        a519 = svld1_gather_u64offset_f64(pg1, a518, svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        a520 = svld1_gather_u64offset_f64(pg1, (a518 + 1), svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        s327 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a520, s320), a519, s319);
        s328 = svmla_f64_x(pg1, svmul_f64_x(pg1, a519, s320), a520, s319);
        a521 = svld1_gather_u64offset_f64(pg1, (a518 + 2), svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        a522 = svld1_gather_u64offset_f64(pg1, (a518 + 3), svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        s329 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a522, s324), a521, s323);
        s330 = svmla_f64_x(pg1, svmul_f64_x(pg1, a521, s324), a522, s323);
        a523 = svld1_gather_u64offset_f64(pg1, (a518 + 4), svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        a524 = svld1_gather_u64offset_f64(pg1, (a518 + 5), svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        s331 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a524, s318), a523, s317);
        s332 = svmla_f64_x(pg1, svmul_f64_x(pg1, a523, s318), a524, s317);
        a525 = svld1_gather_u64offset_f64(pg1, (a518 + 6), svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        a526 = svld1_gather_u64offset_f64(pg1, (a518 + 7), svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        s333 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a526, s322), a525, s321);
        s334 = svmla_f64_x(pg1, svmul_f64_x(pg1, a525, s322), a526, s321);
        a527 = svld1_gather_u64offset_f64(pg1, (a518 + 8), svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        a528 = svld1_gather_u64offset_f64(pg1, (a518 + 9), svindex_u64(0, (int64_t)(10 * sizeof(float64_t))));
        s335 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a528, s326), a527, s325);
        s336 = svmla_f64_x(pg1, svmul_f64_x(pg1, a527, s326), a528, s325);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s315);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(1 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s316);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(2 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s327);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(3 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s328);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(4 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s329);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(5 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s330);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(6 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s331);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(7 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s332);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(8 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s333);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(9 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s334);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(10 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s335);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(11 + Y + ((12)*(j1)))), svindex_u64(0, (int64_t)(12 * sizeof(float64_t))), s336);
        j1 += svcntd();
        pg1 = svwhilelt_b64(j1, l1);
    } while(svptest_any(svptrue_b64(), pg1));
}
