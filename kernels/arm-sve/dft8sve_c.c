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

void dft8c_(float64_t  *Y, float64_t  *X, int  *lp1, int  *mp1) {
    int l1, m1;
    svfloat64x2_t s229, s232, s235, s238, s241, s244, s247, s250, 
            svex2_10, svex2_11, svex2_12, svex2_13, svex2_14, svex2_15, svex2_16, svex2_9;
    svfloat64_t a321, a322, a323, a324, s230, s231, s233, s234, 
            s236, s237, s239, s240, s242, s243, s245, s246, 
            s248, s249, s251, s252, s253, s254, s255, s256, 
            t614, t615, t616, t617, t618, t619, t620, t621, 
            t622, t623, t624, t625, t626, t627, t628, t629, 
            t630, t631, t632, t633, t634, t635, t636, t637, 
            t638, t639, t640, t641;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    int k1 = 0;
    pg1 = svwhilelt_b64(k1, m1);
    do {
        s229 = svld2_f64(pg1, (X + ((2)*(k1))));
        s230 = s229.v0;
        s231 = s229.v1;
        s232 = svld2_f64(pg1, (X + ((2)*((k1 + ((l1)*(m1)))))));
        s233 = s232.v0;
        s234 = s232.v1;
        s235 = svld2_f64(pg1, (X + ((2)*((k1 + ((((2)*(l1)))*(m1)))))));
        s236 = s235.v0;
        s237 = s235.v1;
        s238 = svld2_f64(pg1, (X + ((2)*((k1 + ((((3)*(l1)))*(m1)))))));
        s239 = s238.v0;
        s240 = s238.v1;
        s241 = svld2_f64(pg1, (X + ((2)*((k1 + ((((4)*(l1)))*(m1)))))));
        s242 = s241.v0;
        s243 = s241.v1;
        s244 = svld2_f64(pg1, (X + ((2)*((k1 + ((((5)*(l1)))*(m1)))))));
        s245 = s244.v0;
        s246 = s244.v1;
        s247 = svld2_f64(pg1, (X + ((2)*((k1 + ((((6)*(l1)))*(m1)))))));
        s248 = s247.v0;
        s249 = s247.v1;
        s250 = svld2_f64(pg1, (X + ((2)*((k1 + ((((7)*(l1)))*(m1)))))));
        s251 = s250.v0;
        s252 = s250.v1;
        t614 = svadd_f64_x(pg1, s230, s242);
        t615 = svadd_f64_x(pg1, s231, s243);
        t616 = svsub_f64_x(pg1, s230, s242);
        t617 = svsub_f64_x(pg1, s231, s243);
        t618 = svadd_f64_x(pg1, s236, s248);
        t619 = svadd_f64_x(pg1, s237, s249);
        t620 = svsub_f64_x(pg1, s236, s248);
        t621 = svsub_f64_x(pg1, s237, s249);
        t622 = svadd_f64_x(pg1, t614, t618);
        t623 = svadd_f64_x(pg1, t615, t619);
        t624 = svsub_f64_x(pg1, t614, t618);
        t625 = svsub_f64_x(pg1, t615, t619);
        t626 = svadd_f64_x(pg1, t616, t621);
        t627 = svsub_f64_x(pg1, t617, t620);
        t628 = svsub_f64_x(pg1, t616, t621);
        t629 = svadd_f64_x(pg1, t617, t620);
        t630 = svadd_f64_x(pg1, s233, s245);
        t631 = svadd_f64_x(pg1, s234, s246);
        t632 = svsub_f64_x(pg1, s233, s245);
        t633 = svsub_f64_x(pg1, s234, s246);
        t634 = svadd_f64_x(pg1, s239, s251);
        t635 = svadd_f64_x(pg1, s240, s252);
        t636 = svsub_f64_x(pg1, s239, s251);
        t637 = svsub_f64_x(pg1, s240, s252);
        t638 = svadd_f64_x(pg1, t630, t634);
        t639 = svadd_f64_x(pg1, t631, t635);
        t640 = svsub_f64_x(pg1, t630, t634);
        t641 = svsub_f64_x(pg1, t631, t635);
        a321 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, t632, t637), 0.70710678118654757);
        a322 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t633, t636), 0.70710678118654757);
        s253 = svadd_f64_x(pg1, a321, a322);
        s254 = svsub_f64_x(pg1, a322, a321);
        a323 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, t633, t636), 0.70710678118654757);
        a324 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t632, t637), 0.70710678118654757);
        s255 = svsub_f64_x(pg1, a323, a324);
        s256 = svadd_f64_x(pg1, a324, a323);
        svex2_9.v0 = svadd_f64_x(pg1, t622, t638);
        svex2_9.v1 = svadd_f64_x(pg1, t623, t639);
        svst2_f64(pg1, (Y + ((2)*(k1))), svex2_9);
        svex2_10.v0 = svadd_f64_x(pg1, t626, s253);
        svex2_10.v1 = svadd_f64_x(pg1, t627, s254);
        svst2_f64(pg1, (Y + ((2)*((k1 + m1)))), svex2_10);
        svex2_11.v0 = svadd_f64_x(pg1, t624, t641);
        svex2_11.v1 = svsub_f64_x(pg1, t625, t640);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((2)*(m1)))))), svex2_11);
        svex2_12.v0 = svadd_f64_x(pg1, t628, s255);
        svex2_12.v1 = svsub_f64_x(pg1, t629, s256);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((3)*(m1)))))), svex2_12);
        svex2_13.v0 = svsub_f64_x(pg1, t622, t638);
        svex2_13.v1 = svsub_f64_x(pg1, t623, t639);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((4)*(m1)))))), svex2_13);
        svex2_14.v0 = svsub_f64_x(pg1, t626, s253);
        svex2_14.v1 = svsub_f64_x(pg1, t627, s254);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((5)*(m1)))))), svex2_14);
        svex2_15.v0 = svsub_f64_x(pg1, t624, t641);
        svex2_15.v1 = svadd_f64_x(pg1, t625, t640);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((6)*(m1)))))), svex2_15);
        svex2_16.v0 = svsub_f64_x(pg1, t628, s255);
        svex2_16.v1 = svadd_f64_x(pg1, t629, s256);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((7)*(m1)))))), svex2_16);
        k1 += svcntd();
        pg1 = svwhilelt_b64(k1, m1);
    } while(svptest_any(svptrue_b64(), pg1));
}
