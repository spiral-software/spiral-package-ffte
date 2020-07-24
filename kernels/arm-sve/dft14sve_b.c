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

void dft14b_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1, int  *mp1) {
    float64_t a3647, a3648, a3649, a3650, a3651, a3652, a3653, a3654, 
            a3655, a3656, a3657, a3658, a3659, a3660, a3661, a3662, 
            a3663, a3664, a3665, a3666, a3667, a3668, a3669, a3670, 
            a3671, a3672;
    int a3645, a3646, j1, l1, m1;
    svfloat64x2_t s2870, s2873, s2876, s2879, s2882, s2885, s2888, s2891, 
            s2894, s2897, s2900, s2903, s2906, s2909, svex2_15, svex2_16, 
            svex2_17, svex2_18, svex2_19, svex2_20, svex2_21, svex2_22, svex2_23, svex2_24, 
            svex2_25, svex2_26, svex2_27, svex2_28;
    svfloat64_t s2871, s2872, s2874, s2875, s2877, s2878, s2880, s2881, 
            s2883, s2884, s2886, s2887, s2889, s2890, s2892, s2893, 
            s2895, s2896, s2898, s2899, s2901, s2902, s2904, s2905, 
            s2907, s2908, s2910, s2911, s2912, s2913, s2914, s2915, 
            s2916, s2917, s2918, s2919, s2920, s2921, s2922, s2923, 
            s2924, s2925, s2926, s2927, s2928, s2929, s2930, s2931, 
            s2932, s2933, s2934, s2935, s2936, s2937, s2938, s2939, 
            s2940, s2941, s2942, s2943, s2944, s2945, s2946, s2947, 
            s2948, s2949, s2950, s2951, s2952, s2953, s2954, s2955, 
            s2956, s2957, s2958, s2959, s2960, s2961, s2962, s2963, 
            s2964, s2965, s2966, s2967, s2968, s2969, s2970, s2971, 
            s2972, s2973, s2974, s2975, s2976, s2977, s2978, s2979, 
            s2980, s2981, s2982, s2983, s2984, s2985, s2986, s2987, 
            s2988, s2989, s2990, s2991, s2992, s2993, s2994, s2995, 
            s2996, s2997, s2998, s2999, s3000, s3001, s3002, s3003, 
            s3004, s3005, s3006, s3007, s3008, s3009, s3010, s3011, 
            s3012, s3013, s3014, s3015, s3016, s3017, s3018, s3019, 
            s3020, s3021, s3022, s3023, s3024, s3025, s3026, s3027, 
            s3028, s3029, s3030, s3031, s3032, s3033, s3034, s3035, 
            s3036, s3037, s3038, s3039, s3040, s3041, s3042, s3043, 
            s3044, s3045, t6735, t6736, t6737, t6738, t6739, t6740, 
            t6741, t6742, t6743, t6744, t6745, t6746, t6747, t6748, 
            t6749, t6750, t6751, t6752, t6753, t6754, t6755, t6756, 
            t6757, t6758, t6759, t6760, t6761, t6762, t6763, t6764, 
            t6765, t6766, t6767, t6768, t6769, t6770, t6771, t6772, 
            t6773, t6774, t6775, t6776, t6777, t6778, t6779, t6780, 
            t6781, t6782, t6783, t6784, t6785, t6786, t6787, t6788, 
            t6789, t6790, t6791, t6792, t6793, t6794, t6795, t6796, 
            t6797, t6798, t6799, t6800, t6801, t6802, t6803, t6804, 
            t6805, t6806, t6807, t6808, t6809, t6810, t6811, t6812, 
            t6813, t6814, t6815, t6816, t6817, t6818, t6819, t6820, 
            t6821, t6822, t6823, t6824, t6825, t6826, t6827, t6828, 
            t6829, t6830, t6831, t6832, t6833, t6834, t6835, t6836, 
            t6837, t6838, t6839, t6840, t6841, t6842, t6843, t6844, 
            t6845, t6846, t6847, t6848, t6849, t6850, t6851, t6852, 
            t6853, t6854, t6855, t6856, t6857, t6858, t6859, t6860, 
            t6861, t6862, t6863, t6864, t6865, t6866;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        int k1 = 0;
        pg1 = svwhilelt_b64(k1, m1);
        do {
            a3645 = (k1 + ((j1)*(m1)));
            s2870 = svld2_f64(pg1, (X + ((2)*(a3645))));
            s2871 = s2870.v0;
            s2872 = s2870.v1;
            s2873 = svld2_f64(pg1, (X + ((2)*((a3645 + ((l1)*(m1)))))));
            s2874 = s2873.v0;
            s2875 = s2873.v1;
            s2876 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((2)*(l1)))*(m1)))))));
            s2877 = s2876.v0;
            s2878 = s2876.v1;
            s2879 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((3)*(l1)))*(m1)))))));
            s2880 = s2879.v0;
            s2881 = s2879.v1;
            s2882 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((4)*(l1)))*(m1)))))));
            s2883 = s2882.v0;
            s2884 = s2882.v1;
            s2885 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((5)*(l1)))*(m1)))))));
            s2886 = s2885.v0;
            s2887 = s2885.v1;
            s2888 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((6)*(l1)))*(m1)))))));
            s2889 = s2888.v0;
            s2890 = s2888.v1;
            s2891 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((7)*(l1)))*(m1)))))));
            s2892 = s2891.v0;
            s2893 = s2891.v1;
            s2894 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((8)*(l1)))*(m1)))))));
            s2895 = s2894.v0;
            s2896 = s2894.v1;
            s2897 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((9)*(l1)))*(m1)))))));
            s2898 = s2897.v0;
            s2899 = s2897.v1;
            s2900 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((10)*(l1)))*(m1)))))));
            s2901 = s2900.v0;
            s2902 = s2900.v1;
            s2903 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((11)*(l1)))*(m1)))))));
            s2904 = s2903.v0;
            s2905 = s2903.v1;
            s2906 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((12)*(l1)))*(m1)))))));
            s2907 = s2906.v0;
            s2908 = s2906.v1;
            s2909 = svld2_f64(pg1, (X + ((2)*((a3645 + ((((13)*(l1)))*(m1)))))));
            s2910 = s2909.v0;
            s2911 = s2909.v1;
            t6735 = svadd_f64_x(pg1, s2883, s2895);
            t6736 = svadd_f64_x(pg1, s2884, s2896);
            t6737 = svadd_f64_x(pg1, s2877, t6735);
            t6738 = svadd_f64_x(pg1, s2878, t6736);
            t6739 = svmls_n_f64_x(pg1, s2877, t6735, 0.5);
            t6740 = svmls_n_f64_x(pg1, s2878, t6736, 0.5);
            s2912 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2884, s2896), 0.8660254037844386);
            s2913 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2883, s2895), 0.8660254037844386);
            t6741 = svadd_f64_x(pg1, t6739, s2912);
            t6742 = svsub_f64_x(pg1, t6740, s2913);
            t6743 = svsub_f64_x(pg1, t6739, s2912);
            t6744 = svadd_f64_x(pg1, t6740, s2913);
            t6745 = svadd_f64_x(pg1, s2907, s2901);
            t6746 = svadd_f64_x(pg1, s2908, s2902);
            t6747 = svadd_f64_x(pg1, s2889, t6745);
            t6748 = svadd_f64_x(pg1, s2890, t6746);
            t6749 = svmls_n_f64_x(pg1, s2889, t6745, 0.5);
            t6750 = svmls_n_f64_x(pg1, s2890, t6746, 0.5);
            s2914 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2908, s2902), 0.8660254037844386);
            s2915 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2907, s2901), 0.8660254037844386);
            t6751 = svadd_f64_x(pg1, t6749, s2914);
            t6752 = svsub_f64_x(pg1, t6750, s2915);
            t6753 = svsub_f64_x(pg1, t6749, s2914);
            t6754 = svadd_f64_x(pg1, t6750, s2915);
            s3002 = svmla_n_f64_x(pg1, t6751, t6752, 1.7320508075688772);
            s2916 = svmul_n_f64_x(pg1, s3002, 0.5);
            s3003 = svmls_n_f64_x(pg1, t6752, t6751, 1.7320508075688772);
            s2917 = svmul_n_f64_x(pg1, s3003, 0.5);
            s3004 = svmls_n_f64_x(pg1, t6754, t6753, 0.57735026918962584);
            s2918 = svmul_n_f64_x(pg1, s3004, 0.8660254037844386);
            s3005 = svmla_n_f64_x(pg1, t6753, t6754, 0.57735026918962584);
            s2919 = svmul_n_f64_x(pg1, s3005, 0.8660254037844386);
            t6755 = svadd_f64_x(pg1, t6737, t6747);
            t6756 = svadd_f64_x(pg1, t6738, t6748);
            t6757 = svadd_f64_x(pg1, t6741, s2916);
            t6758 = svadd_f64_x(pg1, t6742, s2917);
            t6759 = svsub_f64_x(pg1, t6741, s2916);
            t6760 = svsub_f64_x(pg1, t6742, s2917);
            t6761 = svadd_f64_x(pg1, t6743, s2918);
            t6762 = svsub_f64_x(pg1, t6744, s2919);
            t6763 = svsub_f64_x(pg1, t6743, s2918);
            t6764 = svadd_f64_x(pg1, t6744, s2919);
            t6765 = svadd_f64_x(pg1, s2871, t6755);
            t6766 = svadd_f64_x(pg1, s2872, t6756);
            t6767 = svmls_n_f64_x(pg1, s2871, t6755, 0.16666666666666666);
            t6768 = svmls_n_f64_x(pg1, s2872, t6756, 0.16666666666666666);
            s3006 = svmla_n_f64_x(pg1, t6757, t6758, 0.41908315722758349);
            s2920 = svmul_n_f64_x(pg1, s3006, 0.4066888930575896);
            s3007 = svmls_n_f64_x(pg1, t6758, t6757, 0.41908315722758349);
            s2921 = svmul_n_f64_x(pg1, s3007, 0.4066888930575896);
            s3008 = svmla_n_f64_x(pg1, t6761, t6762, 0.49572725516748389);
            s2922 = svmul_n_f64_x(pg1, s3008, 0.39507823426270006);
            s3009 = svmls_n_f64_x(pg1, t6762, t6761, 0.49572725516748389);
            s2923 = svmul_n_f64_x(pg1, s3009, 0.39507823426270006);
            s2924 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t6738, t6748), 0.44095855184409843);
            s2925 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t6737, t6747), 0.44095855184409843);
            s3010 = svmls_n_f64_x(pg1, t6759, t6760, 0.49572725516748389);
            s2926 = svmul_n_f64_x(pg1, s3010, 0.39507823426270006);
            s3011 = svmla_n_f64_x(pg1, t6759, t6760, 2.0172382889501304);
            s2927 = svmul_n_f64_x(pg1, s3011, 0.1958510486474645);
            s3012 = svmls_n_f64_x(pg1, t6764, t6763, 2.386161273135941);
            s2928 = svmul_n_f64_x(pg1, s3012, 0.17043646531196566);
            s3013 = svmla_n_f64_x(pg1, t6763, t6764, 2.386161273135941);
            s2929 = svmul_n_f64_x(pg1, s3013, 0.17043646531196566);
            t6769 = svadd_f64_x(pg1, s2922, s2926);
            t6770 = svadd_f64_x(pg1, s2923, s2927);
            t6771 = svadd_f64_x(pg1, t6767, t6769);
            t6772 = svadd_f64_x(pg1, t6768, t6770);
            t6773 = svmls_n_f64_x(pg1, t6767, t6769, 0.5);
            t6774 = svmls_n_f64_x(pg1, t6768, t6770, 0.5);
            s2930 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2923, s2927), 0.8660254037844386);
            s2931 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2922, s2926), 0.8660254037844386);
            t6775 = svadd_f64_x(pg1, t6773, s2930);
            t6776 = svsub_f64_x(pg1, t6774, s2931);
            t6777 = svsub_f64_x(pg1, t6773, s2930);
            t6778 = svadd_f64_x(pg1, t6774, s2931);
            t6779 = svadd_f64_x(pg1, s2924, s2928);
            t6780 = svadd_f64_x(pg1, s2925, s2929);
            t6781 = svadd_f64_x(pg1, s2920, t6779);
            t6782 = svsub_f64_x(pg1, s2921, t6780);
            t6783 = svmls_n_f64_x(pg1, s2920, t6779, 0.5);
            t6784 = svmla_n_f64_x(pg1, s2921, t6780, 0.5);
            s2932 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2929, s2925), 0.8660254037844386);
            s2933 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2924, s2928), 0.8660254037844386);
            t6785 = svadd_f64_x(pg1, t6783, s2932);
            t6786 = svsub_f64_x(pg1, t6784, s2933);
            t6787 = svsub_f64_x(pg1, t6783, s2932);
            t6788 = svadd_f64_x(pg1, t6784, s2933);
            s3014 = svmla_n_f64_x(pg1, t6785, t6786, 1.7320508075688772);
            s2934 = svmul_n_f64_x(pg1, s3014, 0.5);
            s3015 = svmls_n_f64_x(pg1, t6786, t6785, 1.7320508075688772);
            s2935 = svmul_n_f64_x(pg1, s3015, 0.5);
            s3016 = svmls_n_f64_x(pg1, t6788, t6787, 0.57735026918962584);
            s2936 = svmul_n_f64_x(pg1, s3016, 0.8660254037844386);
            s3017 = svmla_n_f64_x(pg1, t6787, t6788, 0.57735026918962584);
            s2937 = svmul_n_f64_x(pg1, s3017, 0.8660254037844386);
            t6789 = svadd_f64_x(pg1, t6771, t6781);
            t6790 = svadd_f64_x(pg1, t6772, t6782);
            t6791 = svsub_f64_x(pg1, t6771, t6781);
            t6792 = svsub_f64_x(pg1, t6772, t6782);
            t6793 = svadd_f64_x(pg1, t6775, s2934);
            t6794 = svadd_f64_x(pg1, t6776, s2935);
            t6795 = svsub_f64_x(pg1, t6775, s2934);
            t6796 = svsub_f64_x(pg1, t6776, s2935);
            t6797 = svadd_f64_x(pg1, t6777, s2936);
            t6798 = svsub_f64_x(pg1, t6778, s2937);
            t6799 = svsub_f64_x(pg1, t6777, s2936);
            t6800 = svadd_f64_x(pg1, t6778, s2937);
            t6801 = svadd_f64_x(pg1, s2886, s2898);
            t6802 = svadd_f64_x(pg1, s2887, s2899);
            t6803 = svadd_f64_x(pg1, s2880, t6801);
            t6804 = svadd_f64_x(pg1, s2881, t6802);
            t6805 = svmls_n_f64_x(pg1, s2880, t6801, 0.5);
            t6806 = svmls_n_f64_x(pg1, s2881, t6802, 0.5);
            s2938 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2887, s2899), 0.8660254037844386);
            s2939 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2886, s2898), 0.8660254037844386);
            t6807 = svadd_f64_x(pg1, t6805, s2938);
            t6808 = svsub_f64_x(pg1, t6806, s2939);
            t6809 = svsub_f64_x(pg1, t6805, s2938);
            t6810 = svadd_f64_x(pg1, t6806, s2939);
            t6811 = svadd_f64_x(pg1, s2910, s2904);
            t6812 = svadd_f64_x(pg1, s2911, s2905);
            t6813 = svadd_f64_x(pg1, s2892, t6811);
            t6814 = svadd_f64_x(pg1, s2893, t6812);
            t6815 = svmls_n_f64_x(pg1, s2892, t6811, 0.5);
            t6816 = svmls_n_f64_x(pg1, s2893, t6812, 0.5);
            s2940 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2911, s2905), 0.8660254037844386);
            s2941 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2910, s2904), 0.8660254037844386);
            t6817 = svadd_f64_x(pg1, t6815, s2940);
            t6818 = svsub_f64_x(pg1, t6816, s2941);
            t6819 = svsub_f64_x(pg1, t6815, s2940);
            t6820 = svadd_f64_x(pg1, t6816, s2941);
            s3018 = svmla_n_f64_x(pg1, t6817, t6818, 1.7320508075688772);
            s2942 = svmul_n_f64_x(pg1, s3018, 0.5);
            s3019 = svmls_n_f64_x(pg1, t6818, t6817, 1.7320508075688772);
            s2943 = svmul_n_f64_x(pg1, s3019, 0.5);
            s3020 = svmls_n_f64_x(pg1, t6820, t6819, 0.57735026918962584);
            s2944 = svmul_n_f64_x(pg1, s3020, 0.8660254037844386);
            s3021 = svmla_n_f64_x(pg1, t6819, t6820, 0.57735026918962584);
            s2945 = svmul_n_f64_x(pg1, s3021, 0.8660254037844386);
            t6821 = svadd_f64_x(pg1, t6803, t6813);
            t6822 = svadd_f64_x(pg1, t6804, t6814);
            t6823 = svadd_f64_x(pg1, t6807, s2942);
            t6824 = svadd_f64_x(pg1, t6808, s2943);
            t6825 = svsub_f64_x(pg1, t6807, s2942);
            t6826 = svsub_f64_x(pg1, t6808, s2943);
            t6827 = svadd_f64_x(pg1, t6809, s2944);
            t6828 = svsub_f64_x(pg1, t6810, s2945);
            t6829 = svsub_f64_x(pg1, t6809, s2944);
            t6830 = svadd_f64_x(pg1, t6810, s2945);
            t6831 = svadd_f64_x(pg1, s2874, t6821);
            t6832 = svadd_f64_x(pg1, s2875, t6822);
            t6833 = svmls_n_f64_x(pg1, s2874, t6821, 0.16666666666666666);
            t6834 = svmls_n_f64_x(pg1, s2875, t6822, 0.16666666666666666);
            s3022 = svmla_n_f64_x(pg1, t6823, t6824, 0.41908315722758349);
            s2946 = svmul_n_f64_x(pg1, s3022, 0.4066888930575896);
            s3023 = svmls_n_f64_x(pg1, t6824, t6823, 0.41908315722758349);
            s2947 = svmul_n_f64_x(pg1, s3023, 0.4066888930575896);
            s3024 = svmla_n_f64_x(pg1, t6827, t6828, 0.49572725516748389);
            s2948 = svmul_n_f64_x(pg1, s3024, 0.39507823426270006);
            s3025 = svmls_n_f64_x(pg1, t6828, t6827, 0.49572725516748389);
            s2949 = svmul_n_f64_x(pg1, s3025, 0.39507823426270006);
            s2950 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t6804, t6814), 0.44095855184409843);
            s2951 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t6803, t6813), 0.44095855184409843);
            s3026 = svmls_n_f64_x(pg1, t6825, t6826, 0.49572725516748389);
            s2952 = svmul_n_f64_x(pg1, s3026, 0.39507823426270006);
            s3027 = svmla_n_f64_x(pg1, t6825, t6826, 2.0172382889501304);
            s2953 = svmul_n_f64_x(pg1, s3027, 0.1958510486474645);
            s3028 = svmls_n_f64_x(pg1, t6830, t6829, 2.386161273135941);
            s2954 = svmul_n_f64_x(pg1, s3028, 0.17043646531196566);
            s3029 = svmla_n_f64_x(pg1, t6829, t6830, 2.386161273135941);
            s2955 = svmul_n_f64_x(pg1, s3029, 0.17043646531196566);
            t6835 = svadd_f64_x(pg1, s2948, s2952);
            t6836 = svadd_f64_x(pg1, s2949, s2953);
            t6837 = svadd_f64_x(pg1, t6833, t6835);
            t6838 = svadd_f64_x(pg1, t6834, t6836);
            t6839 = svmls_n_f64_x(pg1, t6833, t6835, 0.5);
            t6840 = svmls_n_f64_x(pg1, t6834, t6836, 0.5);
            s2956 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2949, s2953), 0.8660254037844386);
            s2957 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2948, s2952), 0.8660254037844386);
            t6841 = svadd_f64_x(pg1, t6839, s2956);
            t6842 = svsub_f64_x(pg1, t6840, s2957);
            t6843 = svsub_f64_x(pg1, t6839, s2956);
            t6844 = svadd_f64_x(pg1, t6840, s2957);
            t6845 = svadd_f64_x(pg1, s2950, s2954);
            t6846 = svadd_f64_x(pg1, s2951, s2955);
            t6847 = svadd_f64_x(pg1, s2946, t6845);
            t6848 = svsub_f64_x(pg1, s2947, t6846);
            t6849 = svmls_n_f64_x(pg1, s2946, t6845, 0.5);
            t6850 = svmla_n_f64_x(pg1, s2947, t6846, 0.5);
            s2958 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2955, s2951), 0.8660254037844386);
            s2959 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2950, s2954), 0.8660254037844386);
            t6851 = svadd_f64_x(pg1, t6849, s2958);
            t6852 = svsub_f64_x(pg1, t6850, s2959);
            t6853 = svsub_f64_x(pg1, t6849, s2958);
            t6854 = svadd_f64_x(pg1, t6850, s2959);
            s3030 = svmla_n_f64_x(pg1, t6851, t6852, 1.7320508075688772);
            s2960 = svmul_n_f64_x(pg1, s3030, 0.5);
            s3031 = svmls_n_f64_x(pg1, t6852, t6851, 1.7320508075688772);
            s2961 = svmul_n_f64_x(pg1, s3031, 0.5);
            s3032 = svmls_n_f64_x(pg1, t6854, t6853, 0.57735026918962584);
            s2962 = svmul_n_f64_x(pg1, s3032, 0.8660254037844386);
            s3033 = svmla_n_f64_x(pg1, t6853, t6854, 0.57735026918962584);
            s2963 = svmul_n_f64_x(pg1, s3033, 0.8660254037844386);
            t6855 = svadd_f64_x(pg1, t6837, t6847);
            t6856 = svadd_f64_x(pg1, t6838, t6848);
            t6857 = svsub_f64_x(pg1, t6837, t6847);
            t6858 = svsub_f64_x(pg1, t6838, t6848);
            s3034 = svmla_n_f64_x(pg1, t6855, t6856, 0.48157461880752861);
            s2964 = svmul_n_f64_x(pg1, s3034, 0.90096886790241915);
            s3035 = svmls_n_f64_x(pg1, t6856, t6855, 0.48157461880752861);
            s2965 = svmul_n_f64_x(pg1, s3035, 0.90096886790241915);
            s3036 = svmls_n_f64_x(pg1, t6858, t6857, 2.0765213965723368);
            s2966 = svmul_n_f64_x(pg1, s3036, 0.43388373911755812);
            s3037 = svmla_n_f64_x(pg1, t6857, t6858, 2.0765213965723368);
            s2967 = svmul_n_f64_x(pg1, s3037, 0.43388373911755812);
            t6859 = svadd_f64_x(pg1, t6841, s2960);
            t6860 = svadd_f64_x(pg1, t6842, s2961);
            t6861 = svsub_f64_x(pg1, t6841, s2960);
            t6862 = svsub_f64_x(pg1, t6842, s2961);
            s3038 = svmla_n_f64_x(pg1, t6859, t6860, 4.381286267534823);
            s2968 = svmul_n_f64_x(pg1, s3038, 0.22252093395631439);
            s3039 = svmls_n_f64_x(pg1, t6860, t6859, 4.381286267534823);
            s2969 = svmul_n_f64_x(pg1, s3039, 0.22252093395631439);
            s3040 = svmls_n_f64_x(pg1, t6862, t6861, 0.22824347439014991);
            s2970 = svmul_n_f64_x(pg1, s3040, 0.97492791218182362);
            s3041 = svmla_n_f64_x(pg1, t6861, t6862, 0.22824347439014991);
            s2971 = svmul_n_f64_x(pg1, s3041, 0.97492791218182362);
            t6863 = svadd_f64_x(pg1, t6843, s2962);
            t6864 = svsub_f64_x(pg1, t6844, s2963);
            t6865 = svsub_f64_x(pg1, t6843, s2962);
            t6866 = svadd_f64_x(pg1, t6844, s2963);
            s3042 = svmla_n_f64_x(pg1, t6863, t6864, 1.253960337662704);
            s2972 = svmul_n_f64_x(pg1, s3042, 0.62348980185873348);
            s3043 = svmls_n_f64_x(pg1, t6864, t6863, 1.253960337662704);
            s2973 = svmul_n_f64_x(pg1, s3043, 0.62348980185873348);
            s3044 = svmls_n_f64_x(pg1, t6866, t6865, 0.79747338888240393);
            s2974 = svmul_n_f64_x(pg1, s3044, 0.7818314824680298);
            s3045 = svmla_n_f64_x(pg1, t6865, t6866, 0.79747338888240393);
            s2975 = svmul_n_f64_x(pg1, s3045, 0.7818314824680298);
            s2976 = svsub_f64_x(pg1, t6765, t6831);
            s2977 = svsub_f64_x(pg1, t6766, t6832);
            s2978 = svadd_f64_x(pg1, t6789, s2964);
            s2979 = svadd_f64_x(pg1, t6790, s2965);
            s2980 = svsub_f64_x(pg1, t6789, s2964);
            s2981 = svsub_f64_x(pg1, t6790, s2965);
            s2982 = svadd_f64_x(pg1, t6797, s2972);
            s2983 = svadd_f64_x(pg1, t6798, s2973);
            s2984 = svsub_f64_x(pg1, t6797, s2972);
            s2985 = svsub_f64_x(pg1, t6798, s2973);
            s2986 = svadd_f64_x(pg1, t6793, s2968);
            s2987 = svadd_f64_x(pg1, t6794, s2969);
            s2988 = svsub_f64_x(pg1, t6793, s2968);
            s2989 = svsub_f64_x(pg1, t6794, s2969);
            s2990 = svadd_f64_x(pg1, t6795, s2970);
            s2991 = svsub_f64_x(pg1, t6796, s2971);
            s2992 = svsub_f64_x(pg1, t6795, s2970);
            s2993 = svadd_f64_x(pg1, t6796, s2971);
            s2994 = svadd_f64_x(pg1, t6799, s2974);
            s2995 = svsub_f64_x(pg1, t6800, s2975);
            s2996 = svsub_f64_x(pg1, t6799, s2974);
            s2997 = svadd_f64_x(pg1, t6800, s2975);
            s2998 = svadd_f64_x(pg1, t6791, s2966);
            s2999 = svsub_f64_x(pg1, t6792, s2967);
            s3000 = svsub_f64_x(pg1, t6791, s2966);
            s3001 = svadd_f64_x(pg1, t6792, s2967);
            a3646 = ((26)*(j1));
            a3647 = TW1[a3646];
            a3648 = TW1[(a3646 + 1)];
            a3649 = TW1[(a3646 + 2)];
            a3650 = TW1[(a3646 + 3)];
            a3651 = TW1[(a3646 + 4)];
            a3652 = TW1[(a3646 + 5)];
            a3653 = TW1[(a3646 + 6)];
            a3654 = TW1[(a3646 + 7)];
            a3655 = TW1[(a3646 + 8)];
            a3656 = TW1[(a3646 + 9)];
            a3657 = TW1[(a3646 + 10)];
            a3658 = TW1[(a3646 + 11)];
            a3659 = TW1[(a3646 + 12)];
            a3660 = TW1[(a3646 + 13)];
            a3661 = TW1[(a3646 + 14)];
            a3662 = TW1[(a3646 + 15)];
            a3663 = TW1[(a3646 + 16)];
            a3664 = TW1[(a3646 + 17)];
            a3665 = TW1[(a3646 + 18)];
            a3666 = TW1[(a3646 + 19)];
            a3667 = TW1[(a3646 + 20)];
            a3668 = TW1[(a3646 + 21)];
            a3669 = TW1[(a3646 + 22)];
            a3670 = TW1[(a3646 + 23)];
            a3671 = TW1[(a3646 + 24)];
            a3672 = TW1[(a3646 + 25)];
            svex2_15.v0 = svadd_f64_x(pg1, t6765, t6831);
            svex2_15.v1 = svadd_f64_x(pg1, t6766, t6832);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)))))), svex2_15);
            svex2_16.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2979, a3648), s2978, a3647);
            svex2_16.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2979, a3647), s2978, a3648);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + m1)))), svex2_16);
            svex2_17.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2983, a3650), s2982, a3649);
            svex2_17.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2983, a3649), s2982, a3650);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((2)*(m1)))))), svex2_17);
            svex2_18.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2987, a3652), s2986, a3651);
            svex2_18.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2987, a3651), s2986, a3652);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((3)*(m1)))))), svex2_18);
            svex2_19.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2991, a3654), s2990, a3653);
            svex2_19.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2991, a3653), s2990, a3654);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((4)*(m1)))))), svex2_19);
            svex2_20.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2995, a3656), s2994, a3655);
            svex2_20.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2995, a3655), s2994, a3656);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((5)*(m1)))))), svex2_20);
            svex2_21.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2999, a3658), s2998, a3657);
            svex2_21.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2999, a3657), s2998, a3658);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((6)*(m1)))))), svex2_21);
            svex2_22.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2977, a3660), s2976, a3659);
            svex2_22.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2977, a3659), s2976, a3660);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((7)*(m1)))))), svex2_22);
            svex2_23.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2981, a3662), s2980, a3661);
            svex2_23.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2981, a3661), s2980, a3662);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((8)*(m1)))))), svex2_23);
            svex2_24.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2985, a3664), s2984, a3663);
            svex2_24.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2985, a3663), s2984, a3664);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((9)*(m1)))))), svex2_24);
            svex2_25.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2989, a3666), s2988, a3665);
            svex2_25.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2989, a3665), s2988, a3666);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((10)*(m1)))))), svex2_25);
            svex2_26.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2993, a3668), s2992, a3667);
            svex2_26.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2993, a3667), s2992, a3668);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((11)*(m1)))))), svex2_26);
            svex2_27.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s2997, a3670), s2996, a3669);
            svex2_27.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s2997, a3669), s2996, a3670);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((12)*(m1)))))), svex2_27);
            svex2_28.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s3001, a3672), s3000, a3671);
            svex2_28.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s3001, a3671), s3000, a3672);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((14)*(j1)))*(m1)) + ((13)*(m1)))))), svex2_28);
            k1 += svcntd();
            pg1 = svwhilelt_b64(k1, m1);
        } while(svptest_any(svptrue_b64(), pg1));
    }
}
