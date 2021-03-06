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

void dft16c_(float64_t  *Y, float64_t  *X, int  *lp1, int  *mp1) {
    int l1, m1;
    svfloat64x2_t s834, s837, s840, s843, s846, s849, s852, s855, 
            s858, s861, s864, s867, s870, s873, s876, s879, 
            svex2_17, svex2_18, svex2_19, svex2_20, svex2_21, svex2_22, svex2_23, svex2_24, 
            svex2_25, svex2_26, svex2_27, svex2_28, svex2_29, svex2_30, svex2_31, svex2_32;
    svfloat64_t a1274, a1275, a1276, a1277, a1278, a1279, a1280, a1281, 
            s835, s836, s838, s839, s841, s842, s844, s845, 
            s847, s848, s850, s851, s853, s854, s856, s857, 
            s859, s860, s862, s863, s865, s866, s868, s869, 
            s871, s872, s874, s875, s877, s878, s880, s881, 
            s882, s883, s884, s885, s886, s887, s888, s889, 
            s890, s891, s892, s893, s894, s895, s896, s897, 
            s898, s899, s900, s901, s902, s903, s904, s905, 
            t3018, t3019, t3020, t3021, t3022, t3023, t3024, t3025, 
            t3026, t3027, t3028, t3029, t3030, t3031, t3032, t3033, 
            t3034, t3035, t3036, t3037, t3038, t3039, t3040, t3041, 
            t3042, t3043, t3044, t3045, t3046, t3047, t3048, t3049, 
            t3050, t3051, t3052, t3053, t3054, t3055, t3056, t3057, 
            t3058, t3059, t3060, t3061, t3062, t3063, t3064, t3065, 
            t3066, t3067, t3068, t3069, t3070, t3071, t3072, t3073, 
            t3074, t3075, t3076, t3077, t3078, t3079, t3080, t3081, 
            t3082, t3083, t3084, t3085, t3086, t3087, t3088, t3089, 
            t3090, t3091, t3092, t3093, t3094, t3095, t3096, t3097, 
            t3098, t3099, t3100, t3101, t3102, t3103, t3104, t3105;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    int k1 = 0;
    pg1 = svwhilelt_b64(k1, m1);
    do {
        s834 = svld2_f64(pg1, (X + ((2)*(k1))));
        s835 = s834.v0;
        s836 = s834.v1;
        s837 = svld2_f64(pg1, (X + ((2)*((k1 + ((l1)*(m1)))))));
        s838 = s837.v0;
        s839 = s837.v1;
        s840 = svld2_f64(pg1, (X + ((2)*((k1 + ((((2)*(l1)))*(m1)))))));
        s841 = s840.v0;
        s842 = s840.v1;
        s843 = svld2_f64(pg1, (X + ((2)*((k1 + ((((3)*(l1)))*(m1)))))));
        s844 = s843.v0;
        s845 = s843.v1;
        s846 = svld2_f64(pg1, (X + ((2)*((k1 + ((((4)*(l1)))*(m1)))))));
        s847 = s846.v0;
        s848 = s846.v1;
        s849 = svld2_f64(pg1, (X + ((2)*((k1 + ((((5)*(l1)))*(m1)))))));
        s850 = s849.v0;
        s851 = s849.v1;
        s852 = svld2_f64(pg1, (X + ((2)*((k1 + ((((6)*(l1)))*(m1)))))));
        s853 = s852.v0;
        s854 = s852.v1;
        s855 = svld2_f64(pg1, (X + ((2)*((k1 + ((((7)*(l1)))*(m1)))))));
        s856 = s855.v0;
        s857 = s855.v1;
        s858 = svld2_f64(pg1, (X + ((2)*((k1 + ((((8)*(l1)))*(m1)))))));
        s859 = s858.v0;
        s860 = s858.v1;
        s861 = svld2_f64(pg1, (X + ((2)*((k1 + ((((9)*(l1)))*(m1)))))));
        s862 = s861.v0;
        s863 = s861.v1;
        s864 = svld2_f64(pg1, (X + ((2)*((k1 + ((((10)*(l1)))*(m1)))))));
        s865 = s864.v0;
        s866 = s864.v1;
        s867 = svld2_f64(pg1, (X + ((2)*((k1 + ((((11)*(l1)))*(m1)))))));
        s868 = s867.v0;
        s869 = s867.v1;
        s870 = svld2_f64(pg1, (X + ((2)*((k1 + ((((12)*(l1)))*(m1)))))));
        s871 = s870.v0;
        s872 = s870.v1;
        s873 = svld2_f64(pg1, (X + ((2)*((k1 + ((((13)*(l1)))*(m1)))))));
        s874 = s873.v0;
        s875 = s873.v1;
        s876 = svld2_f64(pg1, (X + ((2)*((k1 + ((((14)*(l1)))*(m1)))))));
        s877 = s876.v0;
        s878 = s876.v1;
        s879 = svld2_f64(pg1, (X + ((2)*((k1 + ((((15)*(l1)))*(m1)))))));
        s880 = s879.v0;
        s881 = s879.v1;
        t3018 = svadd_f64_x(pg1, s835, s859);
        t3019 = svadd_f64_x(pg1, s836, s860);
        t3020 = svsub_f64_x(pg1, s835, s859);
        t3021 = svsub_f64_x(pg1, s836, s860);
        t3022 = svadd_f64_x(pg1, s847, s871);
        t3023 = svadd_f64_x(pg1, s848, s872);
        t3024 = svsub_f64_x(pg1, s847, s871);
        t3025 = svsub_f64_x(pg1, s848, s872);
        t3026 = svadd_f64_x(pg1, t3018, t3022);
        t3027 = svadd_f64_x(pg1, t3019, t3023);
        t3028 = svsub_f64_x(pg1, t3018, t3022);
        t3029 = svsub_f64_x(pg1, t3019, t3023);
        t3030 = svadd_f64_x(pg1, t3020, t3025);
        t3031 = svsub_f64_x(pg1, t3021, t3024);
        t3032 = svsub_f64_x(pg1, t3020, t3025);
        t3033 = svadd_f64_x(pg1, t3021, t3024);
        t3034 = svadd_f64_x(pg1, s838, s862);
        t3035 = svadd_f64_x(pg1, s839, s863);
        t3036 = svsub_f64_x(pg1, s838, s862);
        t3037 = svsub_f64_x(pg1, s839, s863);
        t3038 = svadd_f64_x(pg1, s850, s874);
        t3039 = svadd_f64_x(pg1, s851, s875);
        t3040 = svsub_f64_x(pg1, s850, s874);
        t3041 = svsub_f64_x(pg1, s851, s875);
        t3042 = svadd_f64_x(pg1, t3034, t3038);
        t3043 = svadd_f64_x(pg1, t3035, t3039);
        a1274 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t3034, t3038), 0.70710678118654757);
        a1275 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t3035, t3039), 0.70710678118654757);
        s882 = svadd_f64_x(pg1, a1274, a1275);
        s883 = svsub_f64_x(pg1, a1275, a1274);
        t3044 = svadd_f64_x(pg1, t3036, t3041);
        t3045 = svsub_f64_x(pg1, t3037, t3040);
        t3046 = svsub_f64_x(pg1, t3036, t3041);
        t3047 = svadd_f64_x(pg1, t3037, t3040);
        s898 = svmla_n_f64_x(pg1, t3044, t3045, 0.41421356237309509);
        s884 = svmul_n_f64_x(pg1, s898, 0.92387953251128674);
        s899 = svmls_n_f64_x(pg1, t3045, t3044, 0.41421356237309509);
        s885 = svmul_n_f64_x(pg1, s899, 0.92387953251128674);
        s900 = svmla_n_f64_x(pg1, t3046, t3047, 2.4142135623730949);
        s886 = svmul_n_f64_x(pg1, s900, 0.38268343236508978);
        s901 = svmls_n_f64_x(pg1, t3047, t3046, 2.4142135623730949);
        s887 = svmul_n_f64_x(pg1, s901, 0.38268343236508978);
        t3048 = svadd_f64_x(pg1, s841, s865);
        t3049 = svadd_f64_x(pg1, s842, s866);
        t3050 = svsub_f64_x(pg1, s841, s865);
        t3051 = svsub_f64_x(pg1, s842, s866);
        t3052 = svadd_f64_x(pg1, s853, s877);
        t3053 = svadd_f64_x(pg1, s854, s878);
        t3054 = svsub_f64_x(pg1, s853, s877);
        t3055 = svsub_f64_x(pg1, s854, s878);
        t3056 = svadd_f64_x(pg1, t3048, t3052);
        t3057 = svadd_f64_x(pg1, t3049, t3053);
        t3058 = svsub_f64_x(pg1, t3048, t3052);
        t3059 = svsub_f64_x(pg1, t3049, t3053);
        a1276 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, t3050, t3055), 0.70710678118654757);
        a1277 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t3051, t3054), 0.70710678118654757);
        s888 = svadd_f64_x(pg1, a1276, a1277);
        s889 = svsub_f64_x(pg1, a1277, a1276);
        a1278 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, t3051, t3054), 0.70710678118654757);
        a1279 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t3050, t3055), 0.70710678118654757);
        s890 = svsub_f64_x(pg1, a1278, a1279);
        s891 = svadd_f64_x(pg1, a1279, a1278);
        t3060 = svadd_f64_x(pg1, s844, s868);
        t3061 = svadd_f64_x(pg1, s845, s869);
        t3062 = svsub_f64_x(pg1, s844, s868);
        t3063 = svsub_f64_x(pg1, s845, s869);
        t3064 = svadd_f64_x(pg1, s856, s880);
        t3065 = svadd_f64_x(pg1, s857, s881);
        t3066 = svsub_f64_x(pg1, s856, s880);
        t3067 = svsub_f64_x(pg1, s857, s881);
        t3068 = svadd_f64_x(pg1, t3060, t3064);
        t3069 = svadd_f64_x(pg1, t3061, t3065);
        a1280 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t3061, t3065), 0.70710678118654757);
        a1281 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t3060, t3064), 0.70710678118654757);
        s892 = svsub_f64_x(pg1, a1280, a1281);
        s893 = svadd_f64_x(pg1, a1281, a1280);
        t3070 = svadd_f64_x(pg1, t3062, t3067);
        t3071 = svsub_f64_x(pg1, t3063, t3066);
        t3072 = svsub_f64_x(pg1, t3062, t3067);
        t3073 = svadd_f64_x(pg1, t3063, t3066);
        s902 = svmla_n_f64_x(pg1, t3070, t3071, 2.4142135623730949);
        s894 = svmul_n_f64_x(pg1, s902, 0.38268343236508978);
        s903 = svmls_n_f64_x(pg1, t3071, t3070, 2.4142135623730949);
        s895 = svmul_n_f64_x(pg1, s903, 0.38268343236508978);
        s904 = svmla_n_f64_x(pg1, t3072, t3073, 0.41421356237309509);
        s896 = svmul_n_f64_x(pg1, s904, 0.92387953251128674);
        s905 = svmls_n_f64_x(pg1, t3072, t3073, 2.4142135623730949);
        s897 = svmul_n_f64_x(pg1, s905, 0.38268343236508978);
        t3074 = svadd_f64_x(pg1, t3026, t3056);
        t3075 = svadd_f64_x(pg1, t3027, t3057);
        t3076 = svsub_f64_x(pg1, t3026, t3056);
        t3077 = svsub_f64_x(pg1, t3027, t3057);
        t3078 = svadd_f64_x(pg1, t3042, t3068);
        t3079 = svadd_f64_x(pg1, t3043, t3069);
        t3080 = svsub_f64_x(pg1, t3042, t3068);
        t3081 = svsub_f64_x(pg1, t3043, t3069);
        t3082 = svadd_f64_x(pg1, t3030, s888);
        t3083 = svadd_f64_x(pg1, t3031, s889);
        t3084 = svsub_f64_x(pg1, t3030, s888);
        t3085 = svsub_f64_x(pg1, t3031, s889);
        t3086 = svadd_f64_x(pg1, s884, s894);
        t3087 = svadd_f64_x(pg1, s885, s895);
        t3088 = svsub_f64_x(pg1, s884, s894);
        t3089 = svsub_f64_x(pg1, s885, s895);
        t3090 = svadd_f64_x(pg1, t3028, t3059);
        t3091 = svsub_f64_x(pg1, t3029, t3058);
        t3092 = svsub_f64_x(pg1, t3028, t3059);
        t3093 = svadd_f64_x(pg1, t3029, t3058);
        t3094 = svadd_f64_x(pg1, s882, s892);
        t3095 = svsub_f64_x(pg1, s883, s893);
        t3096 = svsub_f64_x(pg1, s882, s892);
        t3097 = svadd_f64_x(pg1, s883, s893);
        t3098 = svadd_f64_x(pg1, t3032, s890);
        t3099 = svsub_f64_x(pg1, t3033, s891);
        t3100 = svsub_f64_x(pg1, t3032, s890);
        t3101 = svadd_f64_x(pg1, t3033, s891);
        t3102 = svsub_f64_x(pg1, s886, s896);
        t3103 = svadd_f64_x(pg1, s887, s897);
        t3104 = svadd_f64_x(pg1, s886, s896);
        t3105 = svsub_f64_x(pg1, s887, s897);
        svex2_17.v0 = svadd_f64_x(pg1, t3074, t3078);
        svex2_17.v1 = svadd_f64_x(pg1, t3075, t3079);
        svst2_f64(pg1, (Y + ((2)*(k1))), svex2_17);
        svex2_18.v0 = svadd_f64_x(pg1, t3082, t3086);
        svex2_18.v1 = svadd_f64_x(pg1, t3083, t3087);
        svst2_f64(pg1, (Y + ((2)*((k1 + m1)))), svex2_18);
        svex2_19.v0 = svadd_f64_x(pg1, t3090, t3094);
        svex2_19.v1 = svadd_f64_x(pg1, t3091, t3095);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((2)*(m1)))))), svex2_19);
        svex2_20.v0 = svadd_f64_x(pg1, t3098, t3102);
        svex2_20.v1 = svadd_f64_x(pg1, t3099, t3103);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((3)*(m1)))))), svex2_20);
        svex2_21.v0 = svadd_f64_x(pg1, t3076, t3081);
        svex2_21.v1 = svsub_f64_x(pg1, t3077, t3080);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((4)*(m1)))))), svex2_21);
        svex2_22.v0 = svadd_f64_x(pg1, t3084, t3089);
        svex2_22.v1 = svsub_f64_x(pg1, t3085, t3088);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((5)*(m1)))))), svex2_22);
        svex2_23.v0 = svadd_f64_x(pg1, t3092, t3097);
        svex2_23.v1 = svsub_f64_x(pg1, t3093, t3096);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((6)*(m1)))))), svex2_23);
        svex2_24.v0 = svadd_f64_x(pg1, t3100, t3105);
        svex2_24.v1 = svsub_f64_x(pg1, t3101, t3104);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((7)*(m1)))))), svex2_24);
        svex2_25.v0 = svsub_f64_x(pg1, t3074, t3078);
        svex2_25.v1 = svsub_f64_x(pg1, t3075, t3079);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((8)*(m1)))))), svex2_25);
        svex2_26.v0 = svsub_f64_x(pg1, t3082, t3086);
        svex2_26.v1 = svsub_f64_x(pg1, t3083, t3087);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((9)*(m1)))))), svex2_26);
        svex2_27.v0 = svsub_f64_x(pg1, t3090, t3094);
        svex2_27.v1 = svsub_f64_x(pg1, t3091, t3095);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((10)*(m1)))))), svex2_27);
        svex2_28.v0 = svsub_f64_x(pg1, t3098, t3102);
        svex2_28.v1 = svsub_f64_x(pg1, t3099, t3103);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((11)*(m1)))))), svex2_28);
        svex2_29.v0 = svsub_f64_x(pg1, t3076, t3081);
        svex2_29.v1 = svadd_f64_x(pg1, t3077, t3080);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((12)*(m1)))))), svex2_29);
        svex2_30.v0 = svsub_f64_x(pg1, t3084, t3089);
        svex2_30.v1 = svadd_f64_x(pg1, t3085, t3088);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((13)*(m1)))))), svex2_30);
        svex2_31.v0 = svsub_f64_x(pg1, t3092, t3097);
        svex2_31.v1 = svadd_f64_x(pg1, t3093, t3096);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((14)*(m1)))))), svex2_31);
        svex2_32.v0 = svsub_f64_x(pg1, t3100, t3105);
        svex2_32.v1 = svadd_f64_x(pg1, t3101, t3104);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((15)*(m1)))))), svex2_32);
        k1 += svcntd();
        pg1 = svwhilelt_b64(k1, m1);
    } while(svptest_any(svptrue_b64(), pg1));
}
