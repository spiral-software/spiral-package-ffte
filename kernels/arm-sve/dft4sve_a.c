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

void dft4a_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1) {
    int l1;
    float64_t  *a241;
    svfloat64x2_t s111, s114, s117, s120;
    svfloat64_t a242, a243, a244, a245, a246, a247, s112, s113, 
            s115, s116, s118, s119, s121, s122, s123, s124, 
            s125, s126, s127, s128, s129, s130, s131, s132, 
            s133, s134, s135, s136, t154, t155, t156, t157, 
            t158, t159, t160, t161;
    svbool_t pg1;
    l1 = *(lp1);
    int j1 = 0;
    pg1 = svwhilelt_b64(j1, l1);
    do {
        s111 = svld2_f64(pg1, (X + ((2)*(j1))));
        s112 = s111.v0;
        s113 = s111.v1;
        s114 = svld2_f64(pg1, (X + ((2)*((j1 + l1)))));
        s115 = s114.v0;
        s116 = s114.v1;
        s117 = svld2_f64(pg1, (X + ((2)*((j1 + ((2)*(l1)))))));
        s118 = s117.v0;
        s119 = s117.v1;
        s120 = svld2_f64(pg1, (X + ((2)*((j1 + ((3)*(l1)))))));
        s121 = s120.v0;
        s122 = s120.v1;
        t154 = svadd_f64_x(pg1, s112, s118);
        t155 = svadd_f64_x(pg1, s113, s119);
        t156 = svsub_f64_x(pg1, s112, s118);
        t157 = svsub_f64_x(pg1, s113, s119);
        t158 = svadd_f64_x(pg1, s115, s121);
        t159 = svadd_f64_x(pg1, s116, s122);
        t160 = svsub_f64_x(pg1, s115, s121);
        t161 = svsub_f64_x(pg1, s116, s122);
        s123 = svadd_f64_x(pg1, t154, t158);
        s124 = svadd_f64_x(pg1, t155, t159);
        s125 = svsub_f64_x(pg1, t154, t158);
        s126 = svsub_f64_x(pg1, t155, t159);
        s127 = svadd_f64_x(pg1, t156, t161);
        s128 = svsub_f64_x(pg1, t157, t160);
        s129 = svsub_f64_x(pg1, t156, t161);
        s130 = svadd_f64_x(pg1, t157, t160);
        a241 = (TW1 + ((6)*(j1)));
        a242 = svld1_gather_u64offset_f64(pg1, a241, svindex_u64(0, (int64_t)(6 * sizeof(float64_t))));
        a243 = svld1_gather_u64offset_f64(pg1, (a241 + 1), svindex_u64(0, (int64_t)(6 * sizeof(float64_t))));
        s131 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a243, s128), a242, s127);
        s132 = svmla_f64_x(pg1, svmul_f64_x(pg1, a242, s128), a243, s127);
        a244 = svld1_gather_u64offset_f64(pg1, (a241 + 2), svindex_u64(0, (int64_t)(6 * sizeof(float64_t))));
        a245 = svld1_gather_u64offset_f64(pg1, (a241 + 3), svindex_u64(0, (int64_t)(6 * sizeof(float64_t))));
        s133 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a245, s126), a244, s125);
        s134 = svmla_f64_x(pg1, svmul_f64_x(pg1, a244, s126), a245, s125);
        a246 = svld1_gather_u64offset_f64(pg1, (a241 + 4), svindex_u64(0, (int64_t)(6 * sizeof(float64_t))));
        a247 = svld1_gather_u64offset_f64(pg1, (a241 + 5), svindex_u64(0, (int64_t)(6 * sizeof(float64_t))));
        s135 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a247, s130), a246, s129);
        s136 = svmla_f64_x(pg1, svmul_f64_x(pg1, a246, s130), a247, s129);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(Y + ((8)*(j1)))), svindex_u64(0, (int64_t)(8 * sizeof(float64_t))), s123);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(1 + Y + ((8)*(j1)))), svindex_u64(0, (int64_t)(8 * sizeof(float64_t))), s124);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(2 + Y + ((8)*(j1)))), svindex_u64(0, (int64_t)(8 * sizeof(float64_t))), s131);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(3 + Y + ((8)*(j1)))), svindex_u64(0, (int64_t)(8 * sizeof(float64_t))), s132);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(4 + Y + ((8)*(j1)))), svindex_u64(0, (int64_t)(8 * sizeof(float64_t))), s133);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(5 + Y + ((8)*(j1)))), svindex_u64(0, (int64_t)(8 * sizeof(float64_t))), s134);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(6 + Y + ((8)*(j1)))), svindex_u64(0, (int64_t)(8 * sizeof(float64_t))), s135);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(7 + Y + ((8)*(j1)))), svindex_u64(0, (int64_t)(8 * sizeof(float64_t))), s136);
        j1 += svcntd();
        pg1 = svwhilelt_b64(j1, l1);
    } while(svptest_any(svptrue_b64(), pg1));
}