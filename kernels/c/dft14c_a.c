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


void dft14a_(double  *Y, double  *X, double  *TW1, int  *lp1) {
    double a3760, a3761, a3762, a3763, a3764, a3765, a3766, a3767, 
            a3768, a3769, a3770, a3771, a3772, a3773, a3774, a3775, 
            a3776, a3777, a3778, a3779, a3780, a3781, a3782, a3783, 
            a3784, a3785, s2716, s2717, s2718, s2719, s2720, s2721, 
            s2722, s2723, s2724, s2725, s2726, s2727, s2728, s2729, 
            s2730, s2731, s2732, s2733, s2734, s2735, s2736, s2737, 
            s2738, s2739, s2740, s2741, s2742, s2743, s2744, s2745, 
            s2746, s2747, s2748, s2749, s2750, s2751, s2752, s2753, 
            s2754, s2755, s2756, s2757, s2758, s2759, s2760, s2761, 
            s2762, s2763, s2764, s2765, s2766, s2767, s2768, s2769, 
            s2770, s2771, s2772, s2773, s2774, s2775, s2776, s2777, 
            s2778, s2779, s2780, s2781, s2782, s2783, s2784, s2785, 
            s2786, s2787, s2788, s2789, s2790, s2791, s2792, s2793, 
            s2794, s2795, s2796, s2797, s2798, s2799, s2800, s2801, 
            s2802, s2803, s2804, s2805, s2806, s2807, s2808, s2809, 
            s2810, s2811, s2812, s2813, s2814, s2815, s2816, s2817, 
            s2818, s2819, s2820, s2821, s2822, s2823, s2824, s2825, 
            s2826, s2827, s2828, s2829, s2830, s2831, s2832, s2833, 
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
    int a3745, a3746, a3747, a3748, a3749, a3750, a3751, a3752, 
            a3753, a3754, a3755, a3756, a3757, a3758, a3759, a3786, 
            l1;
    l1 = *(lp1);
    for(int j1 = 0; j1 < l1; j1++) {
        l1 = *(lp1);
        a3745 = (2*j1);
        s2716 = X[a3745];
        s2717 = X[(a3745 + 1)];
        a3746 = (a3745 + (2*l1));
        s2718 = X[a3746];
        s2719 = X[(a3746 + 1)];
        a3747 = (a3745 + (4*l1));
        s2720 = X[a3747];
        s2721 = X[(a3747 + 1)];
        a3748 = (a3745 + (6*l1));
        s2722 = X[a3748];
        s2723 = X[(a3748 + 1)];
        a3749 = (a3745 + (8*l1));
        s2724 = X[a3749];
        s2725 = X[(a3749 + 1)];
        a3750 = (a3745 + (10*l1));
        s2726 = X[a3750];
        s2727 = X[(a3750 + 1)];
        a3751 = (a3745 + (12*l1));
        s2728 = X[a3751];
        s2729 = X[(a3751 + 1)];
        a3752 = (a3745 + (14*l1));
        s2730 = X[a3752];
        s2731 = X[(a3752 + 1)];
        a3753 = (a3745 + (16*l1));
        s2732 = X[a3753];
        s2733 = X[(a3753 + 1)];
        a3754 = (a3745 + (18*l1));
        s2734 = X[a3754];
        s2735 = X[(a3754 + 1)];
        a3755 = (a3745 + (20*l1));
        s2736 = X[a3755];
        s2737 = X[(a3755 + 1)];
        a3756 = (a3745 + (22*l1));
        s2738 = X[a3756];
        s2739 = X[(a3756 + 1)];
        a3757 = (a3745 + (24*l1));
        s2740 = X[a3757];
        s2741 = X[(a3757 + 1)];
        a3758 = (a3745 + (26*l1));
        s2742 = X[a3758];
        s2743 = X[(a3758 + 1)];
        t6734 = (s2724 + s2732);
        t6735 = (s2725 + s2733);
        t6736 = (s2720 + t6734);
        t6737 = (s2721 + t6735);
        t6738 = (s2720 - (0.5*t6734));
        t6739 = (s2721 - (0.5*t6735));
        s2744 = (0.8660254037844386*(s2725 - s2733));
        s2745 = (0.8660254037844386*(s2724 - s2732));
        t6740 = (t6738 + s2744);
        t6741 = (t6739 - s2745);
        t6742 = (t6738 - s2744);
        t6743 = (t6739 + s2745);
        t6744 = (s2740 + s2736);
        t6745 = (s2741 + s2737);
        t6746 = (s2728 + t6744);
        t6747 = (s2729 + t6745);
        t6748 = (s2728 - (0.5*t6744));
        t6749 = (s2729 - (0.5*t6745));
        s2746 = (0.8660254037844386*(s2741 - s2737));
        s2747 = (0.8660254037844386*(s2740 - s2736));
        t6750 = (t6748 + s2746);
        t6751 = (t6749 - s2747);
        t6752 = (t6748 - s2746);
        t6753 = (t6749 + s2747);
        s2748 = ((0.5*t6750) + (0.8660254037844386*t6751));
        s2749 = ((0.5*t6751) - (0.8660254037844386*t6750));
        s2750 = ((0.8660254037844386*t6753) - (0.5*t6752));
        s2751 = ((0.8660254037844386*t6752) + (0.5*t6753));
        t6754 = (t6736 + t6746);
        t6755 = (t6737 + t6747);
        t6756 = (t6740 + s2748);
        t6757 = (t6741 + s2749);
        t6758 = (t6740 - s2748);
        t6759 = (t6741 - s2749);
        t6760 = (t6742 + s2750);
        t6761 = (t6743 - s2751);
        t6762 = (t6742 - s2750);
        t6763 = (t6743 + s2751);
        t6764 = (s2716 + t6754);
        t6765 = (s2717 + t6755);
        t6766 = (s2716 - (0.16666666666666666*t6754));
        t6767 = (s2717 - (0.16666666666666666*t6755));
        s2752 = ((0.4066888930575896*t6756) + (0.17043646531196571*t6757));
        s2753 = ((0.4066888930575896*t6757) - (0.17043646531196571*t6756));
        s2754 = ((0.39507823426270006*t6760) + (0.1958510486474645*t6761));
        s2755 = ((0.39507823426270006*t6761) - (0.1958510486474645*t6760));
        s2756 = (0.44095855184409843*(t6737 - t6747));
        s2757 = (0.44095855184409843*(t6736 - t6746));
        s2758 = ((0.39507823426270006*t6758) - (0.1958510486474645*t6759));
        s2759 = ((0.1958510486474645*t6758) + (0.39507823426270006*t6759));
        s2760 = ((0.17043646531196566*t6763) - (0.4066888930575896*t6762));
        s2761 = ((0.17043646531196566*t6762) + (0.4066888930575896*t6763));
        t6768 = (s2754 + s2758);
        t6769 = (s2755 + s2759);
        t6770 = (t6766 + t6768);
        t6771 = (t6767 + t6769);
        t6772 = (t6766 - (0.5*t6768));
        t6773 = (t6767 - (0.5*t6769));
        s2762 = (0.8660254037844386*(s2755 - s2759));
        s2763 = (0.8660254037844386*(s2754 - s2758));
        t6774 = (t6772 + s2762);
        t6775 = (t6773 - s2763);
        t6776 = (t6772 - s2762);
        t6777 = (t6773 + s2763);
        t6778 = (s2756 + s2760);
        t6779 = (s2757 + s2761);
        t6780 = (s2752 + t6778);
        t6781 = (s2753 - t6779);
        t6782 = (s2752 - (0.5*t6778));
        t6783 = (s2753 + (0.5*t6779));
        s2764 = (0.8660254037844386*(s2761 - s2757));
        s2765 = (0.8660254037844386*(s2756 - s2760));
        t6784 = (t6782 + s2764);
        t6785 = (t6783 - s2765);
        t6786 = (t6782 - s2764);
        t6787 = (t6783 + s2765);
        s2766 = ((0.5*t6784) + (0.8660254037844386*t6785));
        s2767 = ((0.5*t6785) - (0.8660254037844386*t6784));
        s2768 = ((0.8660254037844386*t6787) - (0.5*t6786));
        s2769 = ((0.8660254037844386*t6786) + (0.5*t6787));
        t6788 = (t6770 + t6780);
        t6789 = (t6771 + t6781);
        t6790 = (t6770 - t6780);
        t6791 = (t6771 - t6781);
        t6792 = (t6774 + s2766);
        t6793 = (t6775 + s2767);
        t6794 = (t6774 - s2766);
        t6795 = (t6775 - s2767);
        t6796 = (t6776 + s2768);
        t6797 = (t6777 - s2769);
        t6798 = (t6776 - s2768);
        t6799 = (t6777 + s2769);
        t6800 = (s2726 + s2734);
        t6801 = (s2727 + s2735);
        t6802 = (s2722 + t6800);
        t6803 = (s2723 + t6801);
        t6804 = (s2722 - (0.5*t6800));
        t6805 = (s2723 - (0.5*t6801));
        s2770 = (0.8660254037844386*(s2727 - s2735));
        s2771 = (0.8660254037844386*(s2726 - s2734));
        t6806 = (t6804 + s2770);
        t6807 = (t6805 - s2771);
        t6808 = (t6804 - s2770);
        t6809 = (t6805 + s2771);
        t6810 = (s2742 + s2738);
        t6811 = (s2743 + s2739);
        t6812 = (s2730 + t6810);
        t6813 = (s2731 + t6811);
        t6814 = (s2730 - (0.5*t6810));
        t6815 = (s2731 - (0.5*t6811));
        s2772 = (0.8660254037844386*(s2743 - s2739));
        s2773 = (0.8660254037844386*(s2742 - s2738));
        t6816 = (t6814 + s2772);
        t6817 = (t6815 - s2773);
        t6818 = (t6814 - s2772);
        t6819 = (t6815 + s2773);
        s2774 = ((0.5*t6816) + (0.8660254037844386*t6817));
        s2775 = ((0.5*t6817) - (0.8660254037844386*t6816));
        s2776 = ((0.8660254037844386*t6819) - (0.5*t6818));
        s2777 = ((0.8660254037844386*t6818) + (0.5*t6819));
        t6820 = (t6802 + t6812);
        t6821 = (t6803 + t6813);
        t6822 = (t6806 + s2774);
        t6823 = (t6807 + s2775);
        t6824 = (t6806 - s2774);
        t6825 = (t6807 - s2775);
        t6826 = (t6808 + s2776);
        t6827 = (t6809 - s2777);
        t6828 = (t6808 - s2776);
        t6829 = (t6809 + s2777);
        t6830 = (s2718 + t6820);
        t6831 = (s2719 + t6821);
        t6832 = (s2718 - (0.16666666666666666*t6820));
        t6833 = (s2719 - (0.16666666666666666*t6821));
        s2778 = ((0.4066888930575896*t6822) + (0.17043646531196571*t6823));
        s2779 = ((0.4066888930575896*t6823) - (0.17043646531196571*t6822));
        s2780 = ((0.39507823426270006*t6826) + (0.1958510486474645*t6827));
        s2781 = ((0.39507823426270006*t6827) - (0.1958510486474645*t6826));
        s2782 = (0.44095855184409843*(t6803 - t6813));
        s2783 = (0.44095855184409843*(t6802 - t6812));
        s2784 = ((0.39507823426270006*t6824) - (0.1958510486474645*t6825));
        s2785 = ((0.1958510486474645*t6824) + (0.39507823426270006*t6825));
        s2786 = ((0.17043646531196566*t6829) - (0.4066888930575896*t6828));
        s2787 = ((0.17043646531196566*t6828) + (0.4066888930575896*t6829));
        t6834 = (s2780 + s2784);
        t6835 = (s2781 + s2785);
        t6836 = (t6832 + t6834);
        t6837 = (t6833 + t6835);
        t6838 = (t6832 - (0.5*t6834));
        t6839 = (t6833 - (0.5*t6835));
        s2788 = (0.8660254037844386*(s2781 - s2785));
        s2789 = (0.8660254037844386*(s2780 - s2784));
        t6840 = (t6838 + s2788);
        t6841 = (t6839 - s2789);
        t6842 = (t6838 - s2788);
        t6843 = (t6839 + s2789);
        t6844 = (s2782 + s2786);
        t6845 = (s2783 + s2787);
        t6846 = (s2778 + t6844);
        t6847 = (s2779 - t6845);
        t6848 = (s2778 - (0.5*t6844));
        t6849 = (s2779 + (0.5*t6845));
        s2790 = (0.8660254037844386*(s2787 - s2783));
        s2791 = (0.8660254037844386*(s2782 - s2786));
        t6850 = (t6848 + s2790);
        t6851 = (t6849 - s2791);
        t6852 = (t6848 - s2790);
        t6853 = (t6849 + s2791);
        s2792 = ((0.5*t6850) + (0.8660254037844386*t6851));
        s2793 = ((0.5*t6851) - (0.8660254037844386*t6850));
        s2794 = ((0.8660254037844386*t6853) - (0.5*t6852));
        s2795 = ((0.8660254037844386*t6852) + (0.5*t6853));
        t6854 = (t6836 + t6846);
        t6855 = (t6837 + t6847);
        t6856 = (t6836 - t6846);
        t6857 = (t6837 - t6847);
        s2796 = ((0.90096886790241915*t6854) + (0.43388373911755812*t6855));
        s2797 = ((0.90096886790241915*t6855) - (0.43388373911755812*t6854));
        s2798 = ((0.43388373911755812*t6857) - (0.90096886790241915*t6856));
        s2799 = ((0.43388373911755812*t6856) + (0.90096886790241915*t6857));
        t6858 = (t6840 + s2792);
        t6859 = (t6841 + s2793);
        t6860 = (t6840 - s2792);
        t6861 = (t6841 - s2793);
        s2800 = ((0.22252093395631439*t6858) + (0.97492791218182362*t6859));
        s2801 = ((0.22252093395631439*t6859) - (0.97492791218182362*t6858));
        s2802 = ((0.97492791218182362*t6861) - (0.22252093395631439*t6860));
        s2803 = ((0.97492791218182362*t6860) + (0.22252093395631439*t6861));
        t6862 = (t6842 + s2794);
        t6863 = (t6843 - s2795);
        t6864 = (t6842 - s2794);
        t6865 = (t6843 + s2795);
        s2804 = ((0.62348980185873348*t6862) + (0.7818314824680298*t6863));
        s2805 = ((0.62348980185873348*t6863) - (0.7818314824680298*t6862));
        s2806 = ((0.7818314824680298*t6865) - (0.62348980185873348*t6864));
        s2807 = ((0.7818314824680298*t6864) + (0.62348980185873348*t6865));
        s2808 = (t6764 - t6830);
        s2809 = (t6765 - t6831);
        s2810 = (t6788 + s2796);
        s2811 = (t6789 + s2797);
        s2812 = (t6788 - s2796);
        s2813 = (t6789 - s2797);
        s2814 = (t6796 + s2804);
        s2815 = (t6797 + s2805);
        s2816 = (t6796 - s2804);
        s2817 = (t6797 - s2805);
        s2818 = (t6792 + s2800);
        s2819 = (t6793 + s2801);
        s2820 = (t6792 - s2800);
        s2821 = (t6793 - s2801);
        s2822 = (t6794 + s2802);
        s2823 = (t6795 - s2803);
        s2824 = (t6794 - s2802);
        s2825 = (t6795 + s2803);
        s2826 = (t6798 + s2806);
        s2827 = (t6799 - s2807);
        s2828 = (t6798 - s2806);
        s2829 = (t6799 + s2807);
        s2830 = (t6790 + s2798);
        s2831 = (t6791 - s2799);
        s2832 = (t6790 - s2798);
        s2833 = (t6791 + s2799);
        a3759 = (26*j1);
        a3760 = TW1[a3759];
        a3761 = TW1[(a3759 + 1)];
        a3762 = TW1[(a3759 + 2)];
        a3763 = TW1[(a3759 + 3)];
        a3764 = TW1[(a3759 + 4)];
        a3765 = TW1[(a3759 + 5)];
        a3766 = TW1[(a3759 + 6)];
        a3767 = TW1[(a3759 + 7)];
        a3768 = TW1[(a3759 + 8)];
        a3769 = TW1[(a3759 + 9)];
        a3770 = TW1[(a3759 + 10)];
        a3771 = TW1[(a3759 + 11)];
        a3772 = TW1[(a3759 + 12)];
        a3773 = TW1[(a3759 + 13)];
        a3774 = TW1[(a3759 + 14)];
        a3775 = TW1[(a3759 + 15)];
        a3776 = TW1[(a3759 + 16)];
        a3777 = TW1[(a3759 + 17)];
        a3778 = TW1[(a3759 + 18)];
        a3779 = TW1[(a3759 + 19)];
        a3780 = TW1[(a3759 + 20)];
        a3781 = TW1[(a3759 + 21)];
        a3782 = TW1[(a3759 + 22)];
        a3783 = TW1[(a3759 + 23)];
        a3784 = TW1[(a3759 + 24)];
        a3785 = TW1[(a3759 + 25)];
        a3786 = (28*j1);
        Y[a3786] = (t6764 + t6830);
        Y[(a3786 + 1)] = (t6765 + t6831);
        Y[(a3786 + 2)] = ((a3760*s2810) - (a3761*s2811));
        Y[(a3786 + 3)] = ((a3761*s2810) + (a3760*s2811));
        Y[(a3786 + 4)] = ((a3762*s2814) - (a3763*s2815));
        Y[(a3786 + 5)] = ((a3763*s2814) + (a3762*s2815));
        Y[(a3786 + 6)] = ((a3764*s2818) - (a3765*s2819));
        Y[(a3786 + 7)] = ((a3765*s2818) + (a3764*s2819));
        Y[(a3786 + 8)] = ((a3766*s2822) - (a3767*s2823));
        Y[(a3786 + 9)] = ((a3767*s2822) + (a3766*s2823));
        Y[(a3786 + 10)] = ((a3768*s2826) - (a3769*s2827));
        Y[(a3786 + 11)] = ((a3769*s2826) + (a3768*s2827));
        Y[(a3786 + 12)] = ((a3770*s2830) - (a3771*s2831));
        Y[(a3786 + 13)] = ((a3771*s2830) + (a3770*s2831));
        Y[(a3786 + 14)] = ((a3772*s2808) - (a3773*s2809));
        Y[(a3786 + 15)] = ((a3773*s2808) + (a3772*s2809));
        Y[(a3786 + 16)] = ((a3774*s2812) - (a3775*s2813));
        Y[(a3786 + 17)] = ((a3775*s2812) + (a3774*s2813));
        Y[(a3786 + 18)] = ((a3776*s2816) - (a3777*s2817));
        Y[(a3786 + 19)] = ((a3777*s2816) + (a3776*s2817));
        Y[(a3786 + 20)] = ((a3778*s2820) - (a3779*s2821));
        Y[(a3786 + 21)] = ((a3779*s2820) + (a3778*s2821));
        Y[(a3786 + 22)] = ((a3780*s2824) - (a3781*s2825));
        Y[(a3786 + 23)] = ((a3781*s2824) + (a3780*s2825));
        Y[(a3786 + 24)] = ((a3782*s2828) - (a3783*s2829));
        Y[(a3786 + 25)] = ((a3783*s2828) + (a3782*s2829));
        Y[(a3786 + 26)] = ((a3784*s2832) - (a3785*s2833));
        Y[(a3786 + 27)] = ((a3785*s2832) + (a3784*s2833));
    }
}