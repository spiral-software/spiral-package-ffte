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

void dft12b_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1, int  *mp1) {
    float64_t a1665, a1666, a1667, a1668, a1669, a1670, a1671, a1672, 
            a1673, a1674, a1675, a1676, a1677, a1678, a1679, a1680, 
            a1681, a1682, a1683, a1684, a1685, a1686;
    int a1663, a1664, j1, l1, m1;
    svfloat64x2_t s944, s947, s950, s953, s956, s959, s962, s965, 
            s968, s971, s974, s977, svex2_13, svex2_14, svex2_15, svex2_16, 
            svex2_17, svex2_18, svex2_19, svex2_20, svex2_21, svex2_22, svex2_23, svex2_24;
    svfloat64_t s1000, s1001, s1002, s1003, s1004, s1005, s1006, s1007, 
            s1008, s1009, s1010, s1011, s1012, s1013, s1014, s1015, 
            s1016, s1017, s1018, s1019, s1020, s1021, s1022, s1023, 
            s1024, s1025, s945, s946, s948, s949, s951, s952, 
            s954, s955, s957, s958, s960, s961, s963, s964, 
            s966, s967, s969, s970, s972, s973, s975, s976, 
            s978, s979, s980, s981, s982, s983, s984, s985, 
            s986, s987, s988, s989, s990, s991, s992, s993, 
            s994, s995, s996, s997, s998, s999, t2715, t2716, 
            t2717, t2718, t2719, t2720, t2721, t2722, t2723, t2724, 
            t2725, t2726, t2727, t2728, t2729, t2730, t2731, t2732, 
            t2733, t2734, t2735, t2736, t2737, t2738, t2739, t2740, 
            t2741, t2742, t2743, t2744, t2745, t2746, t2747, t2748, 
            t2749, t2750, t2751, t2752, t2753, t2754, t2755, t2756, 
            t2757, t2758, t2759, t2760, t2761, t2762, t2763, t2764, 
            t2765, t2766, t2767, t2768, t2769, t2770, t2771, t2772, 
            t2773, t2774, t2775, t2776, t2777, t2778;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        int k1 = 0;
        pg1 = svwhilelt_b64(k1, m1);
        do {
            a1663 = (k1 + ((j1)*(m1)));
            s944 = svld2_f64(pg1, (X + ((2)*(a1663))));
            s945 = s944.v0;
            s946 = s944.v1;
            s947 = svld2_f64(pg1, (X + ((2)*((a1663 + ((l1)*(m1)))))));
            s948 = s947.v0;
            s949 = s947.v1;
            s950 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((2)*(l1)))*(m1)))))));
            s951 = s950.v0;
            s952 = s950.v1;
            s953 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((3)*(l1)))*(m1)))))));
            s954 = s953.v0;
            s955 = s953.v1;
            s956 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((4)*(l1)))*(m1)))))));
            s957 = s956.v0;
            s958 = s956.v1;
            s959 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((5)*(l1)))*(m1)))))));
            s960 = s959.v0;
            s961 = s959.v1;
            s962 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((6)*(l1)))*(m1)))))));
            s963 = s962.v0;
            s964 = s962.v1;
            s965 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((7)*(l1)))*(m1)))))));
            s966 = s965.v0;
            s967 = s965.v1;
            s968 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((8)*(l1)))*(m1)))))));
            s969 = s968.v0;
            s970 = s968.v1;
            s971 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((9)*(l1)))*(m1)))))));
            s972 = s971.v0;
            s973 = s971.v1;
            s974 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((10)*(l1)))*(m1)))))));
            s975 = s974.v0;
            s976 = s974.v1;
            s977 = svld2_f64(pg1, (X + ((2)*((a1663 + ((((11)*(l1)))*(m1)))))));
            s978 = s977.v0;
            s979 = s977.v1;
            t2715 = svadd_f64_x(pg1, s945, s963);
            t2716 = svadd_f64_x(pg1, s946, s964);
            t2717 = svsub_f64_x(pg1, s945, s963);
            t2718 = svsub_f64_x(pg1, s946, s964);
            t2719 = svadd_f64_x(pg1, s954, s972);
            t2720 = svadd_f64_x(pg1, s955, s973);
            t2721 = svsub_f64_x(pg1, s954, s972);
            t2722 = svsub_f64_x(pg1, s955, s973);
            t2723 = svadd_f64_x(pg1, t2715, t2719);
            t2724 = svadd_f64_x(pg1, t2716, t2720);
            t2725 = svsub_f64_x(pg1, t2715, t2719);
            t2726 = svsub_f64_x(pg1, t2716, t2720);
            t2727 = svadd_f64_x(pg1, t2717, t2722);
            t2728 = svsub_f64_x(pg1, t2718, t2721);
            t2729 = svsub_f64_x(pg1, t2717, t2722);
            t2730 = svadd_f64_x(pg1, t2718, t2721);
            t2731 = svadd_f64_x(pg1, s948, s966);
            t2732 = svadd_f64_x(pg1, s949, s967);
            t2733 = svsub_f64_x(pg1, s948, s966);
            t2734 = svsub_f64_x(pg1, s949, s967);
            t2735 = svadd_f64_x(pg1, s957, s975);
            t2736 = svadd_f64_x(pg1, s958, s976);
            t2737 = svsub_f64_x(pg1, s957, s975);
            t2738 = svsub_f64_x(pg1, s958, s976);
            t2739 = svadd_f64_x(pg1, t2731, t2735);
            t2740 = svadd_f64_x(pg1, t2732, t2736);
            t2741 = svsub_f64_x(pg1, t2731, t2735);
            t2742 = svsub_f64_x(pg1, t2732, t2736);
            s1018 = svmla_n_f64_x(pg1, t2741, t2742, 1.7320508075688772);
            s980 = svmul_n_f64_x(pg1, s1018, 0.5);
            s1019 = svmls_n_f64_x(pg1, t2742, t2741, 1.7320508075688772);
            s981 = svmul_n_f64_x(pg1, s1019, 0.5);
            t2743 = svadd_f64_x(pg1, t2733, t2738);
            t2744 = svsub_f64_x(pg1, t2734, t2737);
            t2745 = svsub_f64_x(pg1, t2733, t2738);
            t2746 = svadd_f64_x(pg1, t2734, t2737);
            s1020 = svmla_n_f64_x(pg1, t2743, t2744, 0.57735026918962584);
            s982 = svmul_n_f64_x(pg1, s1020, 0.8660254037844386);
            s1021 = svmls_n_f64_x(pg1, t2744, t2743, 0.57735026918962584);
            s983 = svmul_n_f64_x(pg1, s1021, 0.8660254037844386);
            t2747 = svadd_f64_x(pg1, s951, s969);
            t2748 = svadd_f64_x(pg1, s952, s970);
            t2749 = svsub_f64_x(pg1, s951, s969);
            t2750 = svsub_f64_x(pg1, s952, s970);
            t2751 = svadd_f64_x(pg1, s960, s978);
            t2752 = svadd_f64_x(pg1, s961, s979);
            t2753 = svsub_f64_x(pg1, s960, s978);
            t2754 = svsub_f64_x(pg1, s961, s979);
            t2755 = svadd_f64_x(pg1, t2747, t2751);
            t2756 = svadd_f64_x(pg1, t2748, t2752);
            t2757 = svsub_f64_x(pg1, t2747, t2751);
            t2758 = svsub_f64_x(pg1, t2748, t2752);
            s1022 = svmls_n_f64_x(pg1, t2758, t2757, 0.57735026918962584);
            s984 = svmul_n_f64_x(pg1, s1022, 0.8660254037844386);
            s1023 = svmla_n_f64_x(pg1, t2757, t2758, 0.57735026918962584);
            s985 = svmul_n_f64_x(pg1, s1023, 0.8660254037844386);
            t2759 = svadd_f64_x(pg1, t2749, t2754);
            t2760 = svsub_f64_x(pg1, t2750, t2753);
            t2761 = svsub_f64_x(pg1, t2749, t2754);
            t2762 = svadd_f64_x(pg1, t2750, t2753);
            s1024 = svmla_n_f64_x(pg1, t2759, t2760, 1.7320508075688772);
            s986 = svmul_n_f64_x(pg1, s1024, 0.5);
            s1025 = svmls_n_f64_x(pg1, t2760, t2759, 1.7320508075688772);
            s987 = svmul_n_f64_x(pg1, s1025, 0.5);
            t2763 = svadd_f64_x(pg1, t2739, t2755);
            t2764 = svadd_f64_x(pg1, t2740, t2756);
            t2765 = svmls_n_f64_x(pg1, t2723, t2763, 0.5);
            t2766 = svmls_n_f64_x(pg1, t2724, t2764, 0.5);
            s988 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2740, t2756), 0.8660254037844386);
            s989 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2739, t2755), 0.8660254037844386);
            s990 = svadd_f64_x(pg1, t2765, s988);
            s991 = svsub_f64_x(pg1, t2766, s989);
            s992 = svsub_f64_x(pg1, t2765, s988);
            s993 = svadd_f64_x(pg1, t2766, s989);
            t2767 = svadd_f64_x(pg1, s982, s986);
            t2768 = svadd_f64_x(pg1, s983, s987);
            t2769 = svmls_n_f64_x(pg1, t2727, t2767, 0.5);
            t2770 = svmls_n_f64_x(pg1, t2728, t2768, 0.5);
            s994 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s983, s987), 0.8660254037844386);
            s995 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s982, s986), 0.8660254037844386);
            s996 = svadd_f64_x(pg1, t2727, t2767);
            s997 = svadd_f64_x(pg1, t2728, t2768);
            s998 = svadd_f64_x(pg1, t2769, s994);
            s999 = svsub_f64_x(pg1, t2770, s995);
            s1000 = svsub_f64_x(pg1, t2769, s994);
            s1001 = svadd_f64_x(pg1, t2770, s995);
            t2771 = svadd_f64_x(pg1, s980, s984);
            t2772 = svsub_f64_x(pg1, s981, s985);
            t2773 = svmls_n_f64_x(pg1, t2725, t2771, 0.5);
            t2774 = svmls_n_f64_x(pg1, t2726, t2772, 0.5);
            s1002 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s981, s985), 0.8660254037844386);
            s1003 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s980, s984), 0.8660254037844386);
            s1004 = svadd_f64_x(pg1, t2725, t2771);
            s1005 = svadd_f64_x(pg1, t2726, t2772);
            s1006 = svadd_f64_x(pg1, t2773, s1002);
            s1007 = svsub_f64_x(pg1, t2774, s1003);
            s1008 = svsub_f64_x(pg1, t2773, s1002);
            s1009 = svadd_f64_x(pg1, t2774, s1003);
            t2775 = svsub_f64_x(pg1, t2746, t2761);
            t2776 = svadd_f64_x(pg1, t2745, t2762);
            t2777 = svmls_n_f64_x(pg1, t2729, t2775, 0.5);
            t2778 = svmla_n_f64_x(pg1, t2730, t2776, 0.5);
            s1010 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2762, t2745), 0.8660254037844386);
            s1011 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, t2746, t2761), 0.8660254037844386);
            s1012 = svadd_f64_x(pg1, t2729, t2775);
            s1013 = svsub_f64_x(pg1, t2730, t2776);
            s1014 = svadd_f64_x(pg1, t2777, s1010);
            s1015 = svsub_f64_x(pg1, t2778, s1011);
            s1016 = svsub_f64_x(pg1, t2777, s1010);
            s1017 = svadd_f64_x(pg1, t2778, s1011);
            a1664 = ((22)*(j1));
            a1665 = TW1[a1664];
            a1666 = TW1[(a1664 + 1)];
            a1667 = TW1[(a1664 + 2)];
            a1668 = TW1[(a1664 + 3)];
            a1669 = TW1[(a1664 + 4)];
            a1670 = TW1[(a1664 + 5)];
            a1671 = TW1[(a1664 + 6)];
            a1672 = TW1[(a1664 + 7)];
            a1673 = TW1[(a1664 + 8)];
            a1674 = TW1[(a1664 + 9)];
            a1675 = TW1[(a1664 + 10)];
            a1676 = TW1[(a1664 + 11)];
            a1677 = TW1[(a1664 + 12)];
            a1678 = TW1[(a1664 + 13)];
            a1679 = TW1[(a1664 + 14)];
            a1680 = TW1[(a1664 + 15)];
            a1681 = TW1[(a1664 + 16)];
            a1682 = TW1[(a1664 + 17)];
            a1683 = TW1[(a1664 + 18)];
            a1684 = TW1[(a1664 + 19)];
            a1685 = TW1[(a1664 + 20)];
            a1686 = TW1[(a1664 + 21)];
            svex2_13.v0 = svadd_f64_x(pg1, t2723, t2763);
            svex2_13.v1 = svadd_f64_x(pg1, t2724, t2764);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)))))), svex2_13);
            svex2_14.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s997, a1666), s996, a1665);
            svex2_14.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s997, a1665), s996, a1666);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + m1)))), svex2_14);
            svex2_15.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s1005, a1668), s1004, a1667);
            svex2_15.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s1005, a1667), s1004, a1668);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((2)*(m1)))))), svex2_15);
            svex2_16.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s1013, a1670), s1012, a1669);
            svex2_16.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s1013, a1669), s1012, a1670);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((3)*(m1)))))), svex2_16);
            svex2_17.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s991, a1672), s990, a1671);
            svex2_17.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s991, a1671), s990, a1672);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((4)*(m1)))))), svex2_17);
            svex2_18.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s999, a1674), s998, a1673);
            svex2_18.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s999, a1673), s998, a1674);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((5)*(m1)))))), svex2_18);
            svex2_19.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s1007, a1676), s1006, a1675);
            svex2_19.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s1007, a1675), s1006, a1676);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((6)*(m1)))))), svex2_19);
            svex2_20.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s1015, a1678), s1014, a1677);
            svex2_20.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s1015, a1677), s1014, a1678);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((7)*(m1)))))), svex2_20);
            svex2_21.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s993, a1680), s992, a1679);
            svex2_21.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s993, a1679), s992, a1680);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((8)*(m1)))))), svex2_21);
            svex2_22.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s1001, a1682), s1000, a1681);
            svex2_22.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s1001, a1681), s1000, a1682);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((9)*(m1)))))), svex2_22);
            svex2_23.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s1009, a1684), s1008, a1683);
            svex2_23.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s1009, a1683), s1008, a1684);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((10)*(m1)))))), svex2_23);
            svex2_24.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s1017, a1686), s1016, a1685);
            svex2_24.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s1017, a1685), s1016, a1686);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((12)*(j1)))*(m1)) + ((11)*(m1)))))), svex2_24);
            k1 += svcntd();
            pg1 = svwhilelt_b64(k1, m1);
        } while(svptest_any(svptrue_b64(), pg1));
    }
}