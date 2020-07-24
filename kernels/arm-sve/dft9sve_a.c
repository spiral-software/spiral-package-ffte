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

void dft9a_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1) {
    int l1;
    float64_t  *a804;
    svfloat64x2_t s477, s480, s483, s486, s489, s492, s495, s498, 
            s501;
    svfloat64_t a805, a806, a807, a808, a809, a810, a811, a812, 
            a813, a814, a815, a816, a817, a818, a819, a820, 
            s478, s479, s481, s482, s484, s485, s487, s488, 
            s490, s491, s493, s494, s496, s497, s499, s500, 
            s502, s503, s504, s505, s506, s507, s508, s509, 
            s510, s511, s512, s513, s514, s515, s516, s517, 
            s518, s519, s520, s521, s522, s523, s524, s525, 
            s526, s527, s528, s529, s530, s531, s532, s533, 
            s534, s535, s536, s537, s538, s539, s540, s541, 
            s542, s543, s544, s545, s546, s547, s548, s549, 
            s550, s551, s552, s553, s554, s555, s556, s557, 
            s558, s559, s560, s561, s562, s563, s564, s565, 
            t681, t682, t683, t684, t685, t686, t687, t688, 
            t689, t690, t691, t692, t693, t694, t695, t696, 
            t697, t698, t699, t700, t701, t702, t703, t704, 
            t705, t706, t707, t708, t709, t710, t711, t712, 
            t713, t714, t715, t716, t717, t718, t719, t720, 
            t721, t722;
    svbool_t pg1;
    l1 = *(lp1);
    int j1 = 0;
    pg1 = svwhilelt_b64(j1, l1);
    do {
        s477 = svld2_f64(pg1, (X + ((2)*(j1))));
        s478 = s477.v0;
        s479 = s477.v1;
        s480 = svld2_f64(pg1, (X + ((2)*((j1 + l1)))));
        s481 = s480.v0;
        s482 = s480.v1;
        s483 = svld2_f64(pg1, (X + ((2)*((j1 + ((2)*(l1)))))));
        s484 = s483.v0;
        s485 = s483.v1;
        s486 = svld2_f64(pg1, (X + ((2)*((j1 + ((3)*(l1)))))));
        s487 = s486.v0;
        s488 = s486.v1;
        s489 = svld2_f64(pg1, (X + ((2)*((j1 + ((4)*(l1)))))));
        s490 = s489.v0;
        s491 = s489.v1;
        s492 = svld2_f64(pg1, (X + ((2)*((j1 + ((5)*(l1)))))));
        s493 = s492.v0;
        s494 = s492.v1;
        s495 = svld2_f64(pg1, (X + ((2)*((j1 + ((6)*(l1)))))));
        s496 = s495.v0;
        s497 = s495.v1;
        s498 = svld2_f64(pg1, (X + ((2)*((j1 + ((7)*(l1)))))));
        s499 = s498.v0;
        s500 = s498.v1;
        s501 = svld2_f64(pg1, (X + ((2)*((j1 + ((8)*(l1)))))));
        s502 = s501.v0;
        s503 = s501.v1;
        t681 = svadd_f64_x(pg1, s487, s496);
        t682 = svadd_f64_x(pg1, s488, s497);
        t683 = svadd_f64_x(pg1, s478, t681);
        t684 = svadd_f64_x(pg1, s479, t682);
        t685 = svmls_n_f64_x(pg1, s478, t681, 0.5);
        t686 = svmls_n_f64_x(pg1, s479, t682, 0.5);
        s504 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s488, s497), 0.8660254037844386);
        s505 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s487, s496), 0.8660254037844386);
        t687 = svadd_f64_x(pg1, t685, s504);
        t688 = svsub_f64_x(pg1, t686, s505);
        t689 = svsub_f64_x(pg1, t685, s504);
        t690 = svadd_f64_x(pg1, t686, s505);
        t691 = svadd_f64_x(pg1, s490, s499);
        t692 = svadd_f64_x(pg1, s491, s500);
        t693 = svadd_f64_x(pg1, s481, t691);
        t694 = svadd_f64_x(pg1, s482, t692);
        t695 = svmls_n_f64_x(pg1, s481, t691, 0.5);
        t696 = svmls_n_f64_x(pg1, s482, t692, 0.5);
        s506 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s491, s500), 0.8660254037844386);
        s507 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s490, s499), 0.8660254037844386);
        t697 = svadd_f64_x(pg1, t695, s506);
        t698 = svsub_f64_x(pg1, t696, s507);
        t699 = svsub_f64_x(pg1, t695, s506);
        t700 = svadd_f64_x(pg1, t696, s507);
        s558 = svmla_n_f64_x(pg1, t697, t698, 0.83909963117727981);
        s508 = svmul_n_f64_x(pg1, s558, 0.76604444311897812);
        s559 = svmls_n_f64_x(pg1, t698, t697, 0.83909963117727981);
        s509 = svmul_n_f64_x(pg1, s559, 0.76604444311897812);
        s560 = svmla_n_f64_x(pg1, t699, t700, 5.6712818196177102);
        s510 = svmul_n_f64_x(pg1, s560, 0.17364817766693033);
        s561 = svmls_n_f64_x(pg1, t700, t699, 5.6712818196177102);
        s511 = svmul_n_f64_x(pg1, s561, 0.17364817766693033);
        t701 = svadd_f64_x(pg1, s493, s502);
        t702 = svadd_f64_x(pg1, s494, s503);
        t703 = svadd_f64_x(pg1, s484, t701);
        t704 = svadd_f64_x(pg1, s485, t702);
        t705 = svmls_n_f64_x(pg1, s484, t701, 0.5);
        t706 = svmls_n_f64_x(pg1, s485, t702, 0.5);
        s512 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s494, s503), 0.8660254037844386);
        s513 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s493, s502), 0.8660254037844386);
        t707 = svadd_f64_x(pg1, t705, s512);
        t708 = svsub_f64_x(pg1, t706, s513);
        t709 = svsub_f64_x(pg1, t705, s512);
        t710 = svadd_f64_x(pg1, t706, s513);
        s562 = svmla_n_f64_x(pg1, t707, t708, 5.6712818196177102);
        s514 = svmul_n_f64_x(pg1, s562, 0.17364817766693033);
        s563 = svmls_n_f64_x(pg1, t708, t707, 5.6712818196177102);
        s515 = svmul_n_f64_x(pg1, s563, 0.17364817766693033);
        s564 = svmls_n_f64_x(pg1, t710, t709, 2.7474774194546225);
        s516 = svmul_n_f64_x(pg1, s564, 0.34202014332566871);
        s565 = svmla_n_f64_x(pg1, t709, t710, 2.7474774194546225);
        s517 = svmul_n_f64_x(pg1, s565, 0.34202014332566871);
        t711 = svadd_f64_x(pg1, t693, t703);
        t712 = svadd_f64_x(pg1, t694, t704);
        t713 = svmls_n_f64_x(pg1, t683, t711, 0.5);
        t714 = svmls_n_f64_x(pg1, t684, t712, 0.5);
        s518 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t694, t704), 0.8660254037844386);
        s519 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t693, t703), 0.8660254037844386);
        s520 = svadd_f64_x(pg1, t683, t711);
        s521 = svadd_f64_x(pg1, t684, t712);
        s522 = svadd_f64_x(pg1, t713, s518);
        s523 = svsub_f64_x(pg1, t714, s519);
        s524 = svsub_f64_x(pg1, t713, s518);
        s525 = svadd_f64_x(pg1, t714, s519);
        t715 = svadd_f64_x(pg1, s508, s514);
        t716 = svadd_f64_x(pg1, s509, s515);
        t717 = svmls_n_f64_x(pg1, t687, t715, 0.5);
        t718 = svmls_n_f64_x(pg1, t688, t716, 0.5);
        s526 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s509, s515), 0.8660254037844386);
        s527 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s508, s514), 0.8660254037844386);
        s528 = svadd_f64_x(pg1, t687, t715);
        s529 = svadd_f64_x(pg1, t688, t716);
        s530 = svadd_f64_x(pg1, t717, s526);
        s531 = svsub_f64_x(pg1, t718, s527);
        s532 = svsub_f64_x(pg1, t717, s526);
        s533 = svadd_f64_x(pg1, t718, s527);
        t719 = svadd_f64_x(pg1, s510, s516);
        t720 = svsub_f64_x(pg1, s511, s517);
        t721 = svmls_n_f64_x(pg1, t689, t719, 0.5);
        t722 = svmls_n_f64_x(pg1, t690, t720, 0.5);
        s534 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s511, s517), 0.8660254037844386);
        s535 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s510, s516), 0.8660254037844386);
        s536 = svadd_f64_x(pg1, t689, t719);
        s537 = svadd_f64_x(pg1, t690, t720);
        s538 = svadd_f64_x(pg1, t721, s534);
        s539 = svsub_f64_x(pg1, t722, s535);
        s540 = svsub_f64_x(pg1, t721, s534);
        s541 = svadd_f64_x(pg1, t722, s535);
        a804 = (TW1 + ((16)*(j1)));
        a805 = svld1_gather_u64offset_f64(pg1, a804, svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        a806 = svld1_gather_u64offset_f64(pg1, (a804 + 1), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        s542 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a806, s529), a805, s528);
        s543 = svmla_f64_x(pg1, svmul_f64_x(pg1, a805, s529), a806, s528);
        a807 = svld1_gather_u64offset_f64(pg1, (a804 + 2), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        a808 = svld1_gather_u64offset_f64(pg1, (a804 + 3), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        s544 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a808, s537), a807, s536);
        s545 = svmla_f64_x(pg1, svmul_f64_x(pg1, a807, s537), a808, s536);
        a809 = svld1_gather_u64offset_f64(pg1, (a804 + 4), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        a810 = svld1_gather_u64offset_f64(pg1, (a804 + 5), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        s546 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a810, s523), a809, s522);
        s547 = svmla_f64_x(pg1, svmul_f64_x(pg1, a809, s523), a810, s522);
        a811 = svld1_gather_u64offset_f64(pg1, (a804 + 6), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        a812 = svld1_gather_u64offset_f64(pg1, (a804 + 7), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        s548 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a812, s531), a811, s530);
        s549 = svmla_f64_x(pg1, svmul_f64_x(pg1, a811, s531), a812, s530);
        a813 = svld1_gather_u64offset_f64(pg1, (a804 + 8), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        a814 = svld1_gather_u64offset_f64(pg1, (a804 + 9), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        s550 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a814, s539), a813, s538);
        s551 = svmla_f64_x(pg1, svmul_f64_x(pg1, a813, s539), a814, s538);
        a815 = svld1_gather_u64offset_f64(pg1, (a804 + 10), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        a816 = svld1_gather_u64offset_f64(pg1, (a804 + 11), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        s552 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a816, s525), a815, s524);
        s553 = svmla_f64_x(pg1, svmul_f64_x(pg1, a815, s525), a816, s524);
        a817 = svld1_gather_u64offset_f64(pg1, (a804 + 12), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        a818 = svld1_gather_u64offset_f64(pg1, (a804 + 13), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        s554 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a818, s533), a817, s532);
        s555 = svmla_f64_x(pg1, svmul_f64_x(pg1, a817, s533), a818, s532);
        a819 = svld1_gather_u64offset_f64(pg1, (a804 + 14), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        a820 = svld1_gather_u64offset_f64(pg1, (a804 + 15), svindex_u64(0, (int64_t)(16 * sizeof(float64_t))));
        s556 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a820, s541), a819, s540);
        s557 = svmla_f64_x(pg1, svmul_f64_x(pg1, a819, s541), a820, s540);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s520);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(1 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s521);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(2 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s542);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(3 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s543);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(4 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s544);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(5 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s545);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(6 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s546);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(7 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s547);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(8 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s548);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(9 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s549);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(10 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s550);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(11 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s551);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(12 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s552);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(13 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s553);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(14 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s554);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(15 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s555);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(16 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s556);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(17 + Y + ((18)*(j1)))), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))), s557);
        j1 += svcntd();
        pg1 = svwhilelt_b64(j1, l1);
    } while(svptest_any(svptrue_b64(), pg1));
}