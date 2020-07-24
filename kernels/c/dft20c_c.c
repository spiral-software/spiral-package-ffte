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


void dft20c_(double  *Y, double  *X, int  *lp1, int  *mp1) {
    static double D23[40];
    double s1534, s1535, s1536, s1537, s1538, s1539, s1540, s1541, 
            s1542, s1543, s1544, s1545, s1546, s1547, s1548, s1549, 
            s1550, s1551, s1552, s1553, s1554, s1555, s1556, s1557, 
            s1558, s1559, s1560, s1561, s1562, s1563, s1564, s1565, 
            s1566, s1567, s1568, s1569, s1570, s1571, s1572, s1573, 
            s1574, s1575, s1576, s1577, s1578, s1579, s1580, s1581, 
            s1582, s1583, s1584, s1585, s1586, s1587, s1588, s1589, 
            s1590, s1591, s1592, s1593, s1594, s1595, s1596, s1597, 
            s1598, s1599, s1600, s1601, s1602, s1603, s1604, s1605, 
            s1606, s1607, s1608, s1609, s1610, s1611, s1612, s1613, 
            s1614, s1615, s1616, s1617, s1618, s1619, s1620, s1621, 
            s1622, s1623, s1624, s1625, s1626, s1627, s1628, s1629, 
            s1630, s1631, s1632, s1633, s1634, s1635, s1636, s1637, 
            t2886, t2887, t2888, t2889, t2890, t2891, t2892, t2893, 
            t2894, t2895, t2896, t2897, t2898, t2899, t2900, t2901, 
            t2902, t2903, t2904, t2905, t2906, t2907, t2908, t2909, 
            t2910, t2911, t2912, t2913, t2914, t2915, t2916, t2917, 
            t2918, t2919, t2920, t2921, t2922, t2923, t2924, t2925, 
            t2926, t2927, t2928, t2929, t2930, t2931, t2932, t2933, 
            t2934, t2935, t2936, t2937, t2938, t2939, t2940, t2941, 
            t2942, t2943, t2944, t2945, t2946, t2947, t2948, t2949, 
            t2950, t2951, t2952, t2953, t2954, t2955, t2956, t2957, 
            t2958, t2959, t2960, t2961, t2962, t2963, t2964, t2965, 
            t2966, t2967, t2968, t2969, t2970, t2971, t2972, t2973, 
            t2974, t2975, t2976, t2977, t2978, t2979, t2980, t2981, 
            t2982, t2983, t2984, t2985, t2986, t2987, t2988, t2989, 
            t2990, t2991, t2992, t2993, t2994, t2995, t2996, t2997, 
            t2998, t2999, t3000, t3001, t3002, t3003, t3004, t3005, 
            t3006, t3007, t3008, t3009, t3010, t3011, t3012, t3013, 
            t3014, t3015, t3016, t3017, t3018, t3019, t3020, t3021, 
            t3022, t3023, t3024, t3025, t3026, t3027, t3028, t3029, 
            t3030, t3031, t3032, t3033, t3034, t3035, t3036, t3037, 
            t3038, t3039, t3040, t3041, t3042, t3043, t3044, t3045, 
            t3046, t3047, t3048, t3049, t3050, t3051, t3052, t3053, 
            t3054, t3055, t3056, t3057, t3058, t3059, t3060, t3061;
    int a3425, a3426, a3427, a3428, a3429, a3430, a3431, a3432, 
            a3433, a3434, a3435, a3436, a3437, a3438, a3439, a3440, 
            a3441, a3442, a3443, a3444, a3445, a3446, a3447, a3448, 
            a3449, a3450, a3451, a3452, a3453, a3454, a3455, a3456, 
            a3457, a3458, a3459, a3460, a3461, a3462, a3463, a3464, 
            b83, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int k1 = 0; k1 < m1; k1++) {
        a3425 = (2*k1);
        s1534 = X[a3425];
        a3426 = (a3425 + 1);
        s1535 = X[a3426];
        b83 = (l1*m1);
        a3427 = (a3425 + (2*b83));
        s1536 = X[a3427];
        s1537 = X[(a3427 + 1)];
        a3428 = (a3425 + (4*b83));
        s1538 = X[a3428];
        s1539 = X[(a3428 + 1)];
        a3429 = (a3425 + (6*b83));
        s1540 = X[a3429];
        s1541 = X[(a3429 + 1)];
        a3430 = (a3425 + (8*b83));
        s1542 = X[a3430];
        s1543 = X[(a3430 + 1)];
        a3431 = (a3425 + (10*b83));
        s1544 = X[a3431];
        s1545 = X[(a3431 + 1)];
        a3432 = (a3425 + (12*b83));
        s1546 = X[a3432];
        s1547 = X[(a3432 + 1)];
        a3433 = (a3425 + (14*b83));
        s1548 = X[a3433];
        s1549 = X[(a3433 + 1)];
        a3434 = (a3425 + (16*b83));
        s1550 = X[a3434];
        s1551 = X[(a3434 + 1)];
        a3435 = (a3425 + (18*b83));
        s1552 = X[a3435];
        s1553 = X[(a3435 + 1)];
        a3436 = (a3425 + (20*b83));
        s1554 = X[a3436];
        s1555 = X[(a3436 + 1)];
        a3437 = (a3425 + (22*b83));
        s1556 = X[a3437];
        s1557 = X[(a3437 + 1)];
        a3438 = (a3425 + (24*b83));
        s1558 = X[a3438];
        s1559 = X[(a3438 + 1)];
        a3439 = (a3425 + (26*b83));
        s1560 = X[a3439];
        s1561 = X[(a3439 + 1)];
        a3440 = (a3425 + (28*b83));
        s1562 = X[a3440];
        s1563 = X[(a3440 + 1)];
        a3441 = (a3425 + (30*b83));
        s1564 = X[a3441];
        s1565 = X[(a3441 + 1)];
        a3442 = (a3425 + (32*b83));
        s1566 = X[a3442];
        s1567 = X[(a3442 + 1)];
        a3443 = (a3425 + (34*b83));
        s1568 = X[a3443];
        s1569 = X[(a3443 + 1)];
        a3444 = (a3425 + (36*b83));
        s1570 = X[a3444];
        s1571 = X[(a3444 + 1)];
        a3445 = (a3425 + (38*b83));
        s1572 = X[a3445];
        s1573 = X[(a3445 + 1)];
        t2886 = (s1534 + s1554);
        t2887 = (s1535 + s1555);
        t2888 = (s1534 - s1554);
        t2889 = (s1535 - s1555);
        t2890 = (s1544 + s1564);
        t2891 = (s1545 + s1565);
        t2892 = (s1544 - s1564);
        t2893 = (s1545 - s1565);
        t2894 = (t2886 + t2890);
        t2895 = (t2887 + t2891);
        t2896 = (t2886 - t2890);
        t2897 = (t2887 - t2891);
        s1574 = ((D23[0]*t2894) - (D23[1]*t2895));
        s1575 = ((D23[1]*t2894) + (D23[0]*t2895));
        s1576 = ((D23[2]*t2896) - (D23[3]*t2897));
        s1577 = ((D23[3]*t2896) + (D23[2]*t2897));
        t2898 = (t2888 + t2893);
        t2899 = (t2889 - t2892);
        t2900 = (t2888 - t2893);
        t2901 = (t2889 + t2892);
        s1578 = ((D23[4]*t2898) - (D23[5]*t2899));
        s1579 = ((D23[5]*t2898) + (D23[4]*t2899));
        s1580 = ((D23[6]*t2900) - (D23[7]*t2901));
        s1581 = ((D23[7]*t2900) + (D23[6]*t2901));
        t2902 = (s1536 + s1556);
        t2903 = (s1537 + s1557);
        t2904 = (s1536 - s1556);
        t2905 = (s1537 - s1557);
        t2906 = (s1546 + s1566);
        t2907 = (s1547 + s1567);
        t2908 = (s1546 - s1566);
        t2909 = (s1547 - s1567);
        t2910 = (t2902 + t2906);
        t2911 = (t2903 + t2907);
        t2912 = (t2902 - t2906);
        t2913 = (t2903 - t2907);
        s1582 = ((D23[8]*t2910) - (D23[9]*t2911));
        s1583 = ((D23[9]*t2910) + (D23[8]*t2911));
        s1584 = ((D23[10]*t2912) - (D23[11]*t2913));
        s1585 = ((D23[11]*t2912) + (D23[10]*t2913));
        t2914 = (t2904 + t2909);
        t2915 = (t2905 - t2908);
        t2916 = (t2904 - t2909);
        t2917 = (t2905 + t2908);
        s1586 = ((D23[12]*t2914) - (D23[13]*t2915));
        s1587 = ((D23[13]*t2914) + (D23[12]*t2915));
        s1588 = ((D23[14]*t2916) - (D23[15]*t2917));
        s1589 = ((D23[15]*t2916) + (D23[14]*t2917));
        t2918 = (s1538 + s1558);
        t2919 = (s1539 + s1559);
        t2920 = (s1538 - s1558);
        t2921 = (s1539 - s1559);
        t2922 = (s1548 + s1568);
        t2923 = (s1549 + s1569);
        t2924 = (s1548 - s1568);
        t2925 = (s1549 - s1569);
        t2926 = (t2918 + t2922);
        t2927 = (t2919 + t2923);
        t2928 = (t2918 - t2922);
        t2929 = (t2919 - t2923);
        s1590 = ((D23[16]*t2926) - (D23[17]*t2927));
        s1591 = ((D23[17]*t2926) + (D23[16]*t2927));
        s1592 = ((D23[18]*t2928) - (D23[19]*t2929));
        s1593 = ((D23[19]*t2928) + (D23[18]*t2929));
        t2930 = (t2920 + t2925);
        t2931 = (t2921 - t2924);
        t2932 = (t2920 - t2925);
        t2933 = (t2921 + t2924);
        s1594 = ((D23[20]*t2930) - (D23[21]*t2931));
        s1595 = ((D23[21]*t2930) + (D23[20]*t2931));
        s1596 = ((D23[22]*t2932) - (D23[23]*t2933));
        s1597 = ((D23[23]*t2932) + (D23[22]*t2933));
        t2934 = (s1540 + s1560);
        t2935 = (s1541 + s1561);
        t2936 = (s1540 - s1560);
        t2937 = (s1541 - s1561);
        t2938 = (s1550 + s1570);
        t2939 = (s1551 + s1571);
        t2940 = (s1550 - s1570);
        t2941 = (s1551 - s1571);
        t2942 = (t2934 + t2938);
        t2943 = (t2935 + t2939);
        t2944 = (t2934 - t2938);
        t2945 = (t2935 - t2939);
        s1598 = ((D23[24]*t2942) - (D23[25]*t2943));
        s1599 = ((D23[25]*t2942) + (D23[24]*t2943));
        s1600 = ((D23[26]*t2944) - (D23[27]*t2945));
        s1601 = ((D23[27]*t2944) + (D23[26]*t2945));
        t2946 = (t2936 + t2941);
        t2947 = (t2937 - t2940);
        t2948 = (t2936 - t2941);
        t2949 = (t2937 + t2940);
        s1602 = ((D23[28]*t2946) - (D23[29]*t2947));
        s1603 = ((D23[29]*t2946) + (D23[28]*t2947));
        s1604 = ((D23[30]*t2948) - (D23[31]*t2949));
        s1605 = ((D23[31]*t2948) + (D23[30]*t2949));
        t2950 = (s1542 + s1562);
        t2951 = (s1543 + s1563);
        t2952 = (s1542 - s1562);
        t2953 = (s1543 - s1563);
        t2954 = (s1552 + s1572);
        t2955 = (s1553 + s1573);
        t2956 = (s1552 - s1572);
        t2957 = (s1553 - s1573);
        t2958 = (t2950 + t2954);
        t2959 = (t2951 + t2955);
        t2960 = (t2950 - t2954);
        t2961 = (t2951 - t2955);
        s1606 = ((D23[32]*t2958) - (D23[33]*t2959));
        s1607 = ((D23[33]*t2958) + (D23[32]*t2959));
        s1608 = ((D23[34]*t2960) - (D23[35]*t2961));
        s1609 = ((D23[35]*t2960) + (D23[34]*t2961));
        t2962 = (t2952 + t2957);
        t2963 = (t2953 - t2956);
        t2964 = (t2952 - t2957);
        t2965 = (t2953 + t2956);
        s1610 = ((D23[36]*t2962) - (D23[37]*t2963));
        s1611 = ((D23[37]*t2962) + (D23[36]*t2963));
        s1612 = ((D23[38]*t2964) - (D23[39]*t2965));
        s1613 = ((D23[39]*t2964) + (D23[38]*t2965));
        t2966 = (s1582 + s1606);
        t2967 = (s1583 + s1607);
        t2968 = (s1582 - s1606);
        t2969 = (s1583 - s1607);
        t2970 = (s1590 + s1598);
        t2971 = (s1591 + s1599);
        t2972 = (s1590 - s1598);
        t2973 = (s1591 - s1599);
        t2974 = (t2966 + t2970);
        t2975 = (t2967 + t2971);
        t2976 = (t2968 + t2973);
        t2977 = (t2969 - t2972);
        t2978 = (t2968 - t2973);
        t2979 = (t2969 + t2972);
        t2980 = (s1574 - (0.25*t2974));
        t2981 = (s1575 - (0.25*t2975));
        s1614 = ((0.29389262614623657*t2976) + (0.47552825814757677*t2977));
        s1615 = ((0.29389262614623657*t2977) - (0.47552825814757677*t2976));
        s1616 = (0.55901699437494745*(t2966 - t2970));
        s1617 = (0.55901699437494745*(t2967 - t2971));
        s1618 = ((0.47552825814757682*t2979) - (0.29389262614623657*t2978));
        s1619 = ((0.47552825814757682*t2978) + (0.29389262614623657*t2979));
        t2982 = (t2980 + s1616);
        t2983 = (t2981 + s1617);
        t2984 = (t2980 - s1616);
        t2985 = (t2981 - s1617);
        t2986 = (s1614 + s1618);
        t2987 = (s1615 - s1619);
        t2988 = (s1614 - s1618);
        t2989 = (s1615 + s1619);
        t2990 = (s1586 + s1610);
        t2991 = (s1587 + s1611);
        t2992 = (s1586 - s1610);
        t2993 = (s1587 - s1611);
        t2994 = (s1594 + s1602);
        t2995 = (s1595 + s1603);
        t2996 = (s1594 - s1602);
        t2997 = (s1595 - s1603);
        t2998 = (t2990 + t2994);
        t2999 = (t2991 + t2995);
        t3000 = (t2992 + t2997);
        t3001 = (t2993 - t2996);
        t3002 = (t2992 - t2997);
        t3003 = (t2993 + t2996);
        t3004 = (s1578 - (0.25*t2998));
        t3005 = (s1579 - (0.25*t2999));
        s1620 = ((0.29389262614623657*t3000) + (0.47552825814757677*t3001));
        s1621 = ((0.29389262614623657*t3001) - (0.47552825814757677*t3000));
        s1622 = (0.55901699437494745*(t2990 - t2994));
        s1623 = (0.55901699437494745*(t2991 - t2995));
        s1624 = ((0.47552825814757682*t3003) - (0.29389262614623657*t3002));
        s1625 = ((0.47552825814757682*t3002) + (0.29389262614623657*t3003));
        t3006 = (t3004 + s1622);
        t3007 = (t3005 + s1623);
        t3008 = (t3004 - s1622);
        t3009 = (t3005 - s1623);
        t3010 = (s1620 + s1624);
        t3011 = (s1621 - s1625);
        t3012 = (s1620 - s1624);
        t3013 = (s1621 + s1625);
        t3014 = (s1584 + s1608);
        t3015 = (s1585 + s1609);
        t3016 = (s1584 - s1608);
        t3017 = (s1585 - s1609);
        t3018 = (s1592 + s1600);
        t3019 = (s1593 + s1601);
        t3020 = (s1592 - s1600);
        t3021 = (s1593 - s1601);
        t3022 = (t3014 + t3018);
        t3023 = (t3015 + t3019);
        t3024 = (t3016 + t3021);
        t3025 = (t3017 - t3020);
        t3026 = (t3016 - t3021);
        t3027 = (t3017 + t3020);
        t3028 = (s1576 - (0.25*t3022));
        t3029 = (s1577 - (0.25*t3023));
        s1626 = ((0.29389262614623657*t3024) + (0.47552825814757677*t3025));
        s1627 = ((0.29389262614623657*t3025) - (0.47552825814757677*t3024));
        s1628 = (0.55901699437494745*(t3014 - t3018));
        s1629 = (0.55901699437494745*(t3015 - t3019));
        s1630 = ((0.47552825814757682*t3027) - (0.29389262614623657*t3026));
        s1631 = ((0.47552825814757682*t3026) + (0.29389262614623657*t3027));
        t3030 = (t3028 + s1628);
        t3031 = (t3029 + s1629);
        t3032 = (t3028 - s1628);
        t3033 = (t3029 - s1629);
        t3034 = (s1626 + s1630);
        t3035 = (s1627 - s1631);
        t3036 = (s1626 - s1630);
        t3037 = (s1627 + s1631);
        t3038 = (s1588 + s1612);
        t3039 = (s1589 + s1613);
        t3040 = (s1588 - s1612);
        t3041 = (s1589 - s1613);
        t3042 = (s1596 + s1604);
        t3043 = (s1597 + s1605);
        t3044 = (s1596 - s1604);
        t3045 = (s1597 - s1605);
        t3046 = (t3038 + t3042);
        t3047 = (t3039 + t3043);
        t3048 = (t3040 + t3045);
        t3049 = (t3041 - t3044);
        t3050 = (t3040 - t3045);
        t3051 = (t3041 + t3044);
        t3052 = (s1580 - (0.25*t3046));
        t3053 = (s1581 - (0.25*t3047));
        s1632 = ((0.29389262614623657*t3048) + (0.47552825814757677*t3049));
        s1633 = ((0.29389262614623657*t3049) - (0.47552825814757677*t3048));
        s1634 = (0.55901699437494745*(t3038 - t3042));
        s1635 = (0.55901699437494745*(t3039 - t3043));
        s1636 = ((0.47552825814757682*t3051) - (0.29389262614623657*t3050));
        s1637 = ((0.47552825814757682*t3050) + (0.29389262614623657*t3051));
        t3054 = (t3052 + s1634);
        t3055 = (t3053 + s1635);
        t3056 = (t3052 - s1634);
        t3057 = (t3053 - s1635);
        t3058 = (s1632 + s1636);
        t3059 = (s1633 - s1637);
        t3060 = (s1632 - s1636);
        t3061 = (s1633 + s1637);
        Y[a3425] = (s1574 + t2974);
        Y[a3426] = (s1575 + t2975);
        a3446 = (a3425 + (2*m1));
        Y[a3446] = (s1578 + t2998);
        Y[(a3446 + 1)] = (s1579 + t2999);
        a3447 = (a3425 + (4*m1));
        Y[a3447] = (s1576 + t3022);
        Y[(a3447 + 1)] = (s1577 + t3023);
        a3448 = (a3425 + (6*m1));
        Y[a3448] = (s1580 + t3046);
        Y[(a3448 + 1)] = (s1581 + t3047);
        a3449 = (a3425 + (8*m1));
        Y[a3449] = (t2982 + t2986);
        Y[(a3449 + 1)] = (t2983 + t2987);
        a3450 = (a3425 + (10*m1));
        Y[a3450] = (t3006 + t3010);
        Y[(a3450 + 1)] = (t3007 + t3011);
        a3451 = (a3425 + (12*m1));
        Y[a3451] = (t3030 + t3034);
        Y[(a3451 + 1)] = (t3031 + t3035);
        a3452 = (a3425 + (14*m1));
        Y[a3452] = (t3054 + t3058);
        Y[(a3452 + 1)] = (t3055 + t3059);
        a3453 = (a3425 + (16*m1));
        Y[a3453] = (t2984 + t2989);
        Y[(a3453 + 1)] = (t2985 - t2988);
        a3454 = (a3425 + (18*m1));
        Y[a3454] = (t3008 + t3013);
        Y[(a3454 + 1)] = (t3009 - t3012);
        a3455 = (a3425 + (20*m1));
        Y[a3455] = (t3032 + t3037);
        Y[(a3455 + 1)] = (t3033 - t3036);
        a3456 = (a3425 + (22*m1));
        Y[a3456] = (t3056 + t3061);
        Y[(a3456 + 1)] = (t3057 - t3060);
        a3457 = (a3425 + (24*m1));
        Y[a3457] = (t2984 - t2989);
        Y[(a3457 + 1)] = (t2985 + t2988);
        a3458 = (a3425 + (26*m1));
        Y[a3458] = (t3008 - t3013);
        Y[(a3458 + 1)] = (t3009 + t3012);
        a3459 = (a3425 + (28*m1));
        Y[a3459] = (t3032 - t3037);
        Y[(a3459 + 1)] = (t3033 + t3036);
        a3460 = (a3425 + (30*m1));
        Y[a3460] = (t3056 - t3061);
        Y[(a3460 + 1)] = (t3057 + t3060);
        a3461 = (a3425 + (32*m1));
        Y[a3461] = (t2982 - t2986);
        Y[(a3461 + 1)] = (t2983 - t2987);
        a3462 = (a3425 + (34*m1));
        Y[a3462] = (t3006 - t3010);
        Y[(a3462 + 1)] = (t3007 - t3011);
        a3463 = (a3425 + (36*m1));
        Y[a3463] = (t3030 - t3034);
        Y[(a3463 + 1)] = (t3031 - t3035);
        a3464 = (a3425 + (38*m1));
        Y[a3464] = (t3054 - t3058);
        Y[(a3464 + 1)] = (t3055 - t3059);
    }
}
