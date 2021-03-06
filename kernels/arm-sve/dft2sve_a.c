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

void dft2a_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1) {
    int a93, l1;
    float64_t  *a94;
    svfloat64x2_t s53, s56;
    svfloat64_t a95, a96, s54, s55, s57, s58, s59, s60, 
            s61, s62, s63, s64;
    svbool_t pg1;
    l1 = *(lp1);
    int j1 = 0;
    pg1 = svwhilelt_b64(j1, l1);
    do {
        a93 = ((2)*(j1));
        s53 = svld2_f64(pg1, (X + a93));
        s54 = s53.v0;
        s55 = s53.v1;
        s56 = svld2_f64(pg1, (X + ((2)*((j1 + l1)))));
        s57 = s56.v0;
        s58 = s56.v1;
        s59 = svadd_f64_x(pg1, s54, s57);
        s60 = svadd_f64_x(pg1, s55, s58);
        s61 = svsub_f64_x(pg1, s54, s57);
        s62 = svsub_f64_x(pg1, s55, s58);
        a94 = (TW1 + a93);
        a95 = svld1_gather_u64offset_f64(pg1, a94, svindex_u64(0, (int64_t)(2 * sizeof(float64_t))));
        a96 = svld1_gather_u64offset_f64(pg1, (a94 + 1), svindex_u64(0, (int64_t)(2 * sizeof(float64_t))));
        s63 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a96, s62), a95, s61);
        s64 = svmla_f64_x(pg1, svmul_f64_x(pg1, a95, s62), a96, s61);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(Y + ((4)*(j1)))), svindex_u64(0, (int64_t)(4 * sizeof(float64_t))), s59);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(1 + Y + ((4)*(j1)))), svindex_u64(0, (int64_t)(4 * sizeof(float64_t))), s60);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(2 + Y + ((4)*(j1)))), svindex_u64(0, (int64_t)(4 * sizeof(float64_t))), s63);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(3 + Y + ((4)*(j1)))), svindex_u64(0, (int64_t)(4 * sizeof(float64_t))), s64);
        j1 += svcntd();
        pg1 = svwhilelt_b64(j1, l1);
    } while(svptest_any(svptrue_b64(), pg1));
}
