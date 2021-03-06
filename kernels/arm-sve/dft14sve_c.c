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

void dft14c_(float64_t  *Y, float64_t  *X, int  *lp1, int  *mp1) {
    int l1, m1;
    svfloat64x2_t s2792, s2795, s2798, s2801, s2804, s2807, s2810, s2813, 
            s2816, s2819, s2822, s2825, s2828, s2831, svex2_15, svex2_16, 
            svex2_17, svex2_18, svex2_19, svex2_20, svex2_21, svex2_22, svex2_23, svex2_24, 
            svex2_25, svex2_26, svex2_27, svex2_28;
    svfloat64_t s2793, s2794, s2796, s2797, s2799, s2800, s2802, s2803, 
            s2805, s2806, s2808, s2809, s2811, s2812, s2814, s2815, 
            s2817, s2818, s2820, s2821, s2823, s2824, s2826, s2827, 
            s2829, s2830, s2832, s2833, s2834, s2835, s2836, s2837, 
            s2838, s2839, s2840, s2841, s2842, s2843, s2844, s2845, 
            s2846, s2847, s2848, s2849, s2850, s2851, s2852, s2853, 
            s2854, s2855, s2856, s2857, s2858, s2859, s2860, s2861, 
            s2862, s2863, s2864, s2865, s2866, s2867, s2868, s2869, 
            s2870, s2871, s2872, s2873, s2874, s2875, s2876, s2877, 
            s2878, s2879, s2880, s2881, s2882, s2883, s2884, s2885, 
            s2886, s2887, s2888, s2889, s2890, s2891, s2892, s2893, 
            s2894, s2895, s2896, s2897, s2898, s2899, s2900, s2901, 
            s2902, s2903, s2904, s2905, s2906, s2907, s2908, s2909, 
            s2910, s2911, s2912, s2913, s2914, s2915, s2916, s2917, 
            s2918, s2919, s2920, s2921, s2922, s2923, s2924, s2925, 
            s2926, s2927, s2928, s2929, s2930, s2931, s2932, s2933, 
            s2934, s2935, s2936, s2937, s2938, s2939, s2940, s2941, 
            t6734, t6735, t6736, t6737, t6738, t6739, t6740, t6741, 
            t6742, t6743, t6744, t6745, t6746, t6747, t6748, t6749, 
            t6750, t6751, t6752, t6753, t6754, t6755, t6756, t6757, 
            t6758, t6759, t6760, t6761, t6762, t6763, t6764, t6765, 
            t6766, t6767, t6768, t6769, t6770, t6771, t6772, t6773, 
            t6774, t6775, t6776, t6777, t6778, t6779, t6780, t6781, 
            t6782, t6783, t6784, t6785, t6786, t6787, t6788, t6789, 
            t6790, t6791, t6792, t6793, t6794, t6795, t6796, t6797, 
            t6798, t6799, t6800, t6801, t6802, t6803, t6804, t6805, 
            t6806, t6807, t6808, t6809, t6810, t6811, t6812, t6813, 
            t6814, t6815, t6816, t6817, t6818, t6819, t6820, t6821, 
            t6822, t6823, t6824, t6825, t6826, t6827, t6828, t6829, 
            t6830, t6831, t6832, t6833, t6834, t6835, t6836, t6837, 
            t6838, t6839, t6840, t6841, t6842, t6843, t6844, t6845, 
            t6846, t6847, t6848, t6849, t6850, t6851, t6852, t6853, 
            t6854, t6855, t6856, t6857, t6858, t6859, t6860, t6861, 
            t6862, t6863, t6864, t6865;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    int k1 = 0;
    pg1 = svwhilelt_b64(k1, m1);
    do {
        s2792 = svld2_f64(pg1, (X + ((2)*(k1))));
        s2793 = s2792.v0;
        s2794 = s2792.v1;
        s2795 = svld2_f64(pg1, (X + ((2)*((k1 + ((l1)*(m1)))))));
        s2796 = s2795.v0;
        s2797 = s2795.v1;
        s2798 = svld2_f64(pg1, (X + ((2)*((k1 + ((((2)*(l1)))*(m1)))))));
        s2799 = s2798.v0;
        s2800 = s2798.v1;
        s2801 = svld2_f64(pg1, (X + ((2)*((k1 + ((((3)*(l1)))*(m1)))))));
        s2802 = s2801.v0;
        s2803 = s2801.v1;
        s2804 = svld2_f64(pg1, (X + ((2)*((k1 + ((((4)*(l1)))*(m1)))))));
        s2805 = s2804.v0;
        s2806 = s2804.v1;
        s2807 = svld2_f64(pg1, (X + ((2)*((k1 + ((((5)*(l1)))*(m1)))))));
        s2808 = s2807.v0;
        s2809 = s2807.v1;
        s2810 = svld2_f64(pg1, (X + ((2)*((k1 + ((((6)*(l1)))*(m1)))))));
        s2811 = s2810.v0;
        s2812 = s2810.v1;
        s2813 = svld2_f64(pg1, (X + ((2)*((k1 + ((((7)*(l1)))*(m1)))))));
        s2814 = s2813.v0;
        s2815 = s2813.v1;
        s2816 = svld2_f64(pg1, (X + ((2)*((k1 + ((((8)*(l1)))*(m1)))))));
        s2817 = s2816.v0;
        s2818 = s2816.v1;
        s2819 = svld2_f64(pg1, (X + ((2)*((k1 + ((((9)*(l1)))*(m1)))))));
        s2820 = s2819.v0;
        s2821 = s2819.v1;
        s2822 = svld2_f64(pg1, (X + ((2)*((k1 + ((((10)*(l1)))*(m1)))))));
        s2823 = s2822.v0;
        s2824 = s2822.v1;
        s2825 = svld2_f64(pg1, (X + ((2)*((k1 + ((((11)*(l1)))*(m1)))))));
        s2826 = s2825.v0;
        s2827 = s2825.v1;
        s2828 = svld2_f64(pg1, (X + ((2)*((k1 + ((((12)*(l1)))*(m1)))))));
        s2829 = s2828.v0;
        s2830 = s2828.v1;
        s2831 = svld2_f64(pg1, (X + ((2)*((k1 + ((((13)*(l1)))*(m1)))))));
        s2832 = s2831.v0;
        s2833 = s2831.v1;
        t6734 = svadd_f64_x(pg1, s2805, s2817);
        t6735 = svadd_f64_x(pg1, s2806, s2818);
        t6736 = svadd_f64_x(pg1, s2799, t6734);
        t6737 = svadd_f64_x(pg1, s2800, t6735);
        t6738 = svmls_n_f64_x(pg1, s2799, t6734, 0.5);
        t6739 = svmls_n_f64_x(pg1, s2800, t6735, 0.5);
        s2834 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2806, s2818), 0.8660254037844386);
        s2835 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2805, s2817), 0.8660254037844386);
        t6740 = svadd_f64_x(pg1, t6738, s2834);
        t6741 = svsub_f64_x(pg1, t6739, s2835);
        t6742 = svsub_f64_x(pg1, t6738, s2834);
        t6743 = svadd_f64_x(pg1, t6739, s2835);
        t6744 = svadd_f64_x(pg1, s2829, s2823);
        t6745 = svadd_f64_x(pg1, s2830, s2824);
        t6746 = svadd_f64_x(pg1, s2811, t6744);
        t6747 = svadd_f64_x(pg1, s2812, t6745);
        t6748 = svmls_n_f64_x(pg1, s2811, t6744, 0.5);
        t6749 = svmls_n_f64_x(pg1, s2812, t6745, 0.5);
        s2836 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2830, s2824), 0.8660254037844386);
        s2837 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2829, s2823), 0.8660254037844386);
        t6750 = svadd_f64_x(pg1, t6748, s2836);
        t6751 = svsub_f64_x(pg1, t6749, s2837);
        t6752 = svsub_f64_x(pg1, t6748, s2836);
        t6753 = svadd_f64_x(pg1, t6749, s2837);
        s2898 = svmla_n_f64_x(pg1, t6750, t6751, 1.7320508075688772);
        s2838 = svmul_n_f64_x(pg1, s2898, 0.5);
        s2899 = svmls_n_f64_x(pg1, t6751, t6750, 1.7320508075688772);
        s2839 = svmul_n_f64_x(pg1, s2899, 0.5);
        s2900 = svmls_n_f64_x(pg1, t6753, t6752, 0.57735026918962584);
        s2840 = svmul_n_f64_x(pg1, s2900, 0.8660254037844386);
        s2901 = svmla_n_f64_x(pg1, t6752, t6753, 0.57735026918962584);
        s2841 = svmul_n_f64_x(pg1, s2901, 0.8660254037844386);
        t6754 = svadd_f64_x(pg1, t6736, t6746);
        t6755 = svadd_f64_x(pg1, t6737, t6747);
        t6756 = svadd_f64_x(pg1, t6740, s2838);
        t6757 = svadd_f64_x(pg1, t6741, s2839);
        t6758 = svsub_f64_x(pg1, t6740, s2838);
        t6759 = svsub_f64_x(pg1, t6741, s2839);
        t6760 = svadd_f64_x(pg1, t6742, s2840);
        t6761 = svsub_f64_x(pg1, t6743, s2841);
        t6762 = svsub_f64_x(pg1, t6742, s2840);
        t6763 = svadd_f64_x(pg1, t6743, s2841);
        t6764 = svadd_f64_x(pg1, s2793, t6754);
        t6765 = svadd_f64_x(pg1, s2794, t6755);
        t6766 = svmls_n_f64_x(pg1, s2793, t6754, 0.16666666666666666);
        t6767 = svmls_n_f64_x(pg1, s2794, t6755, 0.16666666666666666);
        s2902 = svmla_n_f64_x(pg1, t6756, t6757, 0.41908315722758349);
        s2842 = svmul_n_f64_x(pg1, s2902, 0.4066888930575896);
        s2903 = svmls_n_f64_x(pg1, t6757, t6756, 0.41908315722758349);
        s2843 = svmul_n_f64_x(pg1, s2903, 0.4066888930575896);
        s2904 = svmla_n_f64_x(pg1, t6760, t6761, 0.49572725516748389);
        s2844 = svmul_n_f64_x(pg1, s2904, 0.39507823426270006);
        s2905 = svmls_n_f64_x(pg1, t6761, t6760, 0.49572725516748389);
        s2845 = svmul_n_f64_x(pg1, s2905, 0.39507823426270006);
        s2846 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t6737, t6747), 0.44095855184409843);
        s2847 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t6736, t6746), 0.44095855184409843);
        s2906 = svmls_n_f64_x(pg1, t6758, t6759, 0.49572725516748389);
        s2848 = svmul_n_f64_x(pg1, s2906, 0.39507823426270006);
        s2907 = svmla_n_f64_x(pg1, t6758, t6759, 2.0172382889501304);
        s2849 = svmul_n_f64_x(pg1, s2907, 0.1958510486474645);
        s2908 = svmls_n_f64_x(pg1, t6763, t6762, 2.386161273135941);
        s2850 = svmul_n_f64_x(pg1, s2908, 0.17043646531196566);
        s2909 = svmla_n_f64_x(pg1, t6762, t6763, 2.386161273135941);
        s2851 = svmul_n_f64_x(pg1, s2909, 0.17043646531196566);
        t6768 = svadd_f64_x(pg1, s2844, s2848);
        t6769 = svadd_f64_x(pg1, s2845, s2849);
        t6770 = svadd_f64_x(pg1, t6766, t6768);
        t6771 = svadd_f64_x(pg1, t6767, t6769);
        t6772 = svmls_n_f64_x(pg1, t6766, t6768, 0.5);
        t6773 = svmls_n_f64_x(pg1, t6767, t6769, 0.5);
        s2852 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2845, s2849), 0.8660254037844386);
        s2853 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2844, s2848), 0.8660254037844386);
        t6774 = svadd_f64_x(pg1, t6772, s2852);
        t6775 = svsub_f64_x(pg1, t6773, s2853);
        t6776 = svsub_f64_x(pg1, t6772, s2852);
        t6777 = svadd_f64_x(pg1, t6773, s2853);
        t6778 = svadd_f64_x(pg1, s2846, s2850);
        t6779 = svadd_f64_x(pg1, s2847, s2851);
        t6780 = svadd_f64_x(pg1, s2842, t6778);
        t6781 = svsub_f64_x(pg1, s2843, t6779);
        t6782 = svmls_n_f64_x(pg1, s2842, t6778, 0.5);
        t6783 = svmla_n_f64_x(pg1, s2843, t6779, 0.5);
        s2854 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2851, s2847), 0.8660254037844386);
        s2855 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2846, s2850), 0.8660254037844386);
        t6784 = svadd_f64_x(pg1, t6782, s2854);
        t6785 = svsub_f64_x(pg1, t6783, s2855);
        t6786 = svsub_f64_x(pg1, t6782, s2854);
        t6787 = svadd_f64_x(pg1, t6783, s2855);
        s2910 = svmla_n_f64_x(pg1, t6784, t6785, 1.7320508075688772);
        s2856 = svmul_n_f64_x(pg1, s2910, 0.5);
        s2911 = svmls_n_f64_x(pg1, t6785, t6784, 1.7320508075688772);
        s2857 = svmul_n_f64_x(pg1, s2911, 0.5);
        s2912 = svmls_n_f64_x(pg1, t6787, t6786, 0.57735026918962584);
        s2858 = svmul_n_f64_x(pg1, s2912, 0.8660254037844386);
        s2913 = svmla_n_f64_x(pg1, t6786, t6787, 0.57735026918962584);
        s2859 = svmul_n_f64_x(pg1, s2913, 0.8660254037844386);
        t6788 = svadd_f64_x(pg1, t6770, t6780);
        t6789 = svadd_f64_x(pg1, t6771, t6781);
        t6790 = svsub_f64_x(pg1, t6770, t6780);
        t6791 = svsub_f64_x(pg1, t6771, t6781);
        t6792 = svadd_f64_x(pg1, t6774, s2856);
        t6793 = svadd_f64_x(pg1, t6775, s2857);
        t6794 = svsub_f64_x(pg1, t6774, s2856);
        t6795 = svsub_f64_x(pg1, t6775, s2857);
        t6796 = svadd_f64_x(pg1, t6776, s2858);
        t6797 = svsub_f64_x(pg1, t6777, s2859);
        t6798 = svsub_f64_x(pg1, t6776, s2858);
        t6799 = svadd_f64_x(pg1, t6777, s2859);
        t6800 = svadd_f64_x(pg1, s2808, s2820);
        t6801 = svadd_f64_x(pg1, s2809, s2821);
        t6802 = svadd_f64_x(pg1, s2802, t6800);
        t6803 = svadd_f64_x(pg1, s2803, t6801);
        t6804 = svmls_n_f64_x(pg1, s2802, t6800, 0.5);
        t6805 = svmls_n_f64_x(pg1, s2803, t6801, 0.5);
        s2860 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2809, s2821), 0.8660254037844386);
        s2861 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2808, s2820), 0.8660254037844386);
        t6806 = svadd_f64_x(pg1, t6804, s2860);
        t6807 = svsub_f64_x(pg1, t6805, s2861);
        t6808 = svsub_f64_x(pg1, t6804, s2860);
        t6809 = svadd_f64_x(pg1, t6805, s2861);
        t6810 = svadd_f64_x(pg1, s2832, s2826);
        t6811 = svadd_f64_x(pg1, s2833, s2827);
        t6812 = svadd_f64_x(pg1, s2814, t6810);
        t6813 = svadd_f64_x(pg1, s2815, t6811);
        t6814 = svmls_n_f64_x(pg1, s2814, t6810, 0.5);
        t6815 = svmls_n_f64_x(pg1, s2815, t6811, 0.5);
        s2862 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2833, s2827), 0.8660254037844386);
        s2863 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2832, s2826), 0.8660254037844386);
        t6816 = svadd_f64_x(pg1, t6814, s2862);
        t6817 = svsub_f64_x(pg1, t6815, s2863);
        t6818 = svsub_f64_x(pg1, t6814, s2862);
        t6819 = svadd_f64_x(pg1, t6815, s2863);
        s2914 = svmla_n_f64_x(pg1, t6816, t6817, 1.7320508075688772);
        s2864 = svmul_n_f64_x(pg1, s2914, 0.5);
        s2915 = svmls_n_f64_x(pg1, t6817, t6816, 1.7320508075688772);
        s2865 = svmul_n_f64_x(pg1, s2915, 0.5);
        s2916 = svmls_n_f64_x(pg1, t6819, t6818, 0.57735026918962584);
        s2866 = svmul_n_f64_x(pg1, s2916, 0.8660254037844386);
        s2917 = svmla_n_f64_x(pg1, t6818, t6819, 0.57735026918962584);
        s2867 = svmul_n_f64_x(pg1, s2917, 0.8660254037844386);
        t6820 = svadd_f64_x(pg1, t6802, t6812);
        t6821 = svadd_f64_x(pg1, t6803, t6813);
        t6822 = svadd_f64_x(pg1, t6806, s2864);
        t6823 = svadd_f64_x(pg1, t6807, s2865);
        t6824 = svsub_f64_x(pg1, t6806, s2864);
        t6825 = svsub_f64_x(pg1, t6807, s2865);
        t6826 = svadd_f64_x(pg1, t6808, s2866);
        t6827 = svsub_f64_x(pg1, t6809, s2867);
        t6828 = svsub_f64_x(pg1, t6808, s2866);
        t6829 = svadd_f64_x(pg1, t6809, s2867);
        t6830 = svadd_f64_x(pg1, s2796, t6820);
        t6831 = svadd_f64_x(pg1, s2797, t6821);
        t6832 = svmls_n_f64_x(pg1, s2796, t6820, 0.16666666666666666);
        t6833 = svmls_n_f64_x(pg1, s2797, t6821, 0.16666666666666666);
        s2918 = svmla_n_f64_x(pg1, t6822, t6823, 0.41908315722758349);
        s2868 = svmul_n_f64_x(pg1, s2918, 0.4066888930575896);
        s2919 = svmls_n_f64_x(pg1, t6823, t6822, 0.41908315722758349);
        s2869 = svmul_n_f64_x(pg1, s2919, 0.4066888930575896);
        s2920 = svmla_n_f64_x(pg1, t6826, t6827, 0.49572725516748389);
        s2870 = svmul_n_f64_x(pg1, s2920, 0.39507823426270006);
        s2921 = svmls_n_f64_x(pg1, t6827, t6826, 0.49572725516748389);
        s2871 = svmul_n_f64_x(pg1, s2921, 0.39507823426270006);
        s2872 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t6803, t6813), 0.44095855184409843);
        s2873 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t6802, t6812), 0.44095855184409843);
        s2922 = svmls_n_f64_x(pg1, t6824, t6825, 0.49572725516748389);
        s2874 = svmul_n_f64_x(pg1, s2922, 0.39507823426270006);
        s2923 = svmla_n_f64_x(pg1, t6824, t6825, 2.0172382889501304);
        s2875 = svmul_n_f64_x(pg1, s2923, 0.1958510486474645);
        s2924 = svmls_n_f64_x(pg1, t6829, t6828, 2.386161273135941);
        s2876 = svmul_n_f64_x(pg1, s2924, 0.17043646531196566);
        s2925 = svmla_n_f64_x(pg1, t6828, t6829, 2.386161273135941);
        s2877 = svmul_n_f64_x(pg1, s2925, 0.17043646531196566);
        t6834 = svadd_f64_x(pg1, s2870, s2874);
        t6835 = svadd_f64_x(pg1, s2871, s2875);
        t6836 = svadd_f64_x(pg1, t6832, t6834);
        t6837 = svadd_f64_x(pg1, t6833, t6835);
        t6838 = svmls_n_f64_x(pg1, t6832, t6834, 0.5);
        t6839 = svmls_n_f64_x(pg1, t6833, t6835, 0.5);
        s2878 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2871, s2875), 0.8660254037844386);
        s2879 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2870, s2874), 0.8660254037844386);
        t6840 = svadd_f64_x(pg1, t6838, s2878);
        t6841 = svsub_f64_x(pg1, t6839, s2879);
        t6842 = svsub_f64_x(pg1, t6838, s2878);
        t6843 = svadd_f64_x(pg1, t6839, s2879);
        t6844 = svadd_f64_x(pg1, s2872, s2876);
        t6845 = svadd_f64_x(pg1, s2873, s2877);
        t6846 = svadd_f64_x(pg1, s2868, t6844);
        t6847 = svsub_f64_x(pg1, s2869, t6845);
        t6848 = svmls_n_f64_x(pg1, s2868, t6844, 0.5);
        t6849 = svmla_n_f64_x(pg1, s2869, t6845, 0.5);
        s2880 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2877, s2873), 0.8660254037844386);
        s2881 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s2872, s2876), 0.8660254037844386);
        t6850 = svadd_f64_x(pg1, t6848, s2880);
        t6851 = svsub_f64_x(pg1, t6849, s2881);
        t6852 = svsub_f64_x(pg1, t6848, s2880);
        t6853 = svadd_f64_x(pg1, t6849, s2881);
        s2926 = svmla_n_f64_x(pg1, t6850, t6851, 1.7320508075688772);
        s2882 = svmul_n_f64_x(pg1, s2926, 0.5);
        s2927 = svmls_n_f64_x(pg1, t6851, t6850, 1.7320508075688772);
        s2883 = svmul_n_f64_x(pg1, s2927, 0.5);
        s2928 = svmls_n_f64_x(pg1, t6853, t6852, 0.57735026918962584);
        s2884 = svmul_n_f64_x(pg1, s2928, 0.8660254037844386);
        s2929 = svmla_n_f64_x(pg1, t6852, t6853, 0.57735026918962584);
        s2885 = svmul_n_f64_x(pg1, s2929, 0.8660254037844386);
        t6854 = svadd_f64_x(pg1, t6836, t6846);
        t6855 = svadd_f64_x(pg1, t6837, t6847);
        t6856 = svsub_f64_x(pg1, t6836, t6846);
        t6857 = svsub_f64_x(pg1, t6837, t6847);
        s2930 = svmla_n_f64_x(pg1, t6854, t6855, 0.48157461880752861);
        s2886 = svmul_n_f64_x(pg1, s2930, 0.90096886790241915);
        s2931 = svmls_n_f64_x(pg1, t6855, t6854, 0.48157461880752861);
        s2887 = svmul_n_f64_x(pg1, s2931, 0.90096886790241915);
        s2932 = svmls_n_f64_x(pg1, t6857, t6856, 2.0765213965723368);
        s2888 = svmul_n_f64_x(pg1, s2932, 0.43388373911755812);
        s2933 = svmla_n_f64_x(pg1, t6856, t6857, 2.0765213965723368);
        s2889 = svmul_n_f64_x(pg1, s2933, 0.43388373911755812);
        t6858 = svadd_f64_x(pg1, t6840, s2882);
        t6859 = svadd_f64_x(pg1, t6841, s2883);
        t6860 = svsub_f64_x(pg1, t6840, s2882);
        t6861 = svsub_f64_x(pg1, t6841, s2883);
        s2934 = svmla_n_f64_x(pg1, t6858, t6859, 4.381286267534823);
        s2890 = svmul_n_f64_x(pg1, s2934, 0.22252093395631439);
        s2935 = svmls_n_f64_x(pg1, t6859, t6858, 4.381286267534823);
        s2891 = svmul_n_f64_x(pg1, s2935, 0.22252093395631439);
        s2936 = svmls_n_f64_x(pg1, t6861, t6860, 0.22824347439014991);
        s2892 = svmul_n_f64_x(pg1, s2936, 0.97492791218182362);
        s2937 = svmla_n_f64_x(pg1, t6860, t6861, 0.22824347439014991);
        s2893 = svmul_n_f64_x(pg1, s2937, 0.97492791218182362);
        t6862 = svadd_f64_x(pg1, t6842, s2884);
        t6863 = svsub_f64_x(pg1, t6843, s2885);
        t6864 = svsub_f64_x(pg1, t6842, s2884);
        t6865 = svadd_f64_x(pg1, t6843, s2885);
        s2938 = svmla_n_f64_x(pg1, t6862, t6863, 1.253960337662704);
        s2894 = svmul_n_f64_x(pg1, s2938, 0.62348980185873348);
        s2939 = svmls_n_f64_x(pg1, t6863, t6862, 1.253960337662704);
        s2895 = svmul_n_f64_x(pg1, s2939, 0.62348980185873348);
        s2940 = svmls_n_f64_x(pg1, t6865, t6864, 0.79747338888240393);
        s2896 = svmul_n_f64_x(pg1, s2940, 0.7818314824680298);
        s2941 = svmla_n_f64_x(pg1, t6864, t6865, 0.79747338888240393);
        s2897 = svmul_n_f64_x(pg1, s2941, 0.7818314824680298);
        svex2_15.v0 = svadd_f64_x(pg1, t6764, t6830);
        svex2_15.v1 = svadd_f64_x(pg1, t6765, t6831);
        svst2_f64(pg1, (Y + ((2)*(k1))), svex2_15);
        svex2_16.v0 = svadd_f64_x(pg1, t6788, s2886);
        svex2_16.v1 = svadd_f64_x(pg1, t6789, s2887);
        svst2_f64(pg1, (Y + ((2)*((k1 + m1)))), svex2_16);
        svex2_17.v0 = svadd_f64_x(pg1, t6796, s2894);
        svex2_17.v1 = svadd_f64_x(pg1, t6797, s2895);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((2)*(m1)))))), svex2_17);
        svex2_18.v0 = svadd_f64_x(pg1, t6792, s2890);
        svex2_18.v1 = svadd_f64_x(pg1, t6793, s2891);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((3)*(m1)))))), svex2_18);
        svex2_19.v0 = svadd_f64_x(pg1, t6794, s2892);
        svex2_19.v1 = svsub_f64_x(pg1, t6795, s2893);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((4)*(m1)))))), svex2_19);
        svex2_20.v0 = svadd_f64_x(pg1, t6798, s2896);
        svex2_20.v1 = svsub_f64_x(pg1, t6799, s2897);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((5)*(m1)))))), svex2_20);
        svex2_21.v0 = svadd_f64_x(pg1, t6790, s2888);
        svex2_21.v1 = svsub_f64_x(pg1, t6791, s2889);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((6)*(m1)))))), svex2_21);
        svex2_22.v0 = svsub_f64_x(pg1, t6764, t6830);
        svex2_22.v1 = svsub_f64_x(pg1, t6765, t6831);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((7)*(m1)))))), svex2_22);
        svex2_23.v0 = svsub_f64_x(pg1, t6788, s2886);
        svex2_23.v1 = svsub_f64_x(pg1, t6789, s2887);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((8)*(m1)))))), svex2_23);
        svex2_24.v0 = svsub_f64_x(pg1, t6796, s2894);
        svex2_24.v1 = svsub_f64_x(pg1, t6797, s2895);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((9)*(m1)))))), svex2_24);
        svex2_25.v0 = svsub_f64_x(pg1, t6792, s2890);
        svex2_25.v1 = svsub_f64_x(pg1, t6793, s2891);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((10)*(m1)))))), svex2_25);
        svex2_26.v0 = svsub_f64_x(pg1, t6794, s2892);
        svex2_26.v1 = svadd_f64_x(pg1, t6795, s2893);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((11)*(m1)))))), svex2_26);
        svex2_27.v0 = svsub_f64_x(pg1, t6798, s2896);
        svex2_27.v1 = svadd_f64_x(pg1, t6799, s2897);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((12)*(m1)))))), svex2_27);
        svex2_28.v0 = svsub_f64_x(pg1, t6790, s2888);
        svex2_28.v1 = svadd_f64_x(pg1, t6791, s2889);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((13)*(m1)))))), svex2_28);
        k1 += svcntd();
        pg1 = svwhilelt_b64(k1, m1);
    } while(svptest_any(svptrue_b64(), pg1));
}
