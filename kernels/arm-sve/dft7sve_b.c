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

void dft7b_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1, int  *mp1) {
    float64_t a1110, a1111, a1112, a1113, a1114, a1115, a1116, a1117, 
            a1118, a1119, a1120, a1121;
    int a1108, a1109, j1, l1, m1;
    svfloat64x2_t s783, s786, s789, s792, s795, s798, s801, svex2_10, 
            svex2_11, svex2_12, svex2_13, svex2_14, svex2_8, svex2_9;
    svfloat64_t s784, s785, s787, s788, s790, s791, s793, s794, 
            s796, s797, s799, s800, s802, s803, s804, s805, 
            s806, s807, s808, s809, s810, s811, s812, s813, 
            s814, s815, s816, s817, s818, s819, s820, s821, 
            s822, s823, s824, s825, s826, s827, s828, s829, 
            s830, s831, s832, s833, s834, s835, s836, s837, 
            s838, s839, s840, s841, s842, s843, s844, s845, 
            s846, s847, s848, s849, s850, s851, s852, s853, 
            s854, s855, s856, s857, t1799, t1800, t1801, t1802, 
            t1803, t1804, t1805, t1806, t1807, t1808, t1809, t1810, 
            t1811, t1812, t1813, t1814, t1815, t1816, t1817, t1818, 
            t1819, t1820, t1821, t1822, t1823, t1824, t1825, t1826, 
            t1827, t1828, t1829, t1830, t1831, t1832, t1833, t1834, 
            t1835, t1836, t1837, t1838, t1839, t1840, t1841, t1842, 
            t1843, t1844, t1845, t1846, t1847, t1848, t1849, t1850;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        int k1 = 0;
        pg1 = svwhilelt_b64(k1, m1);
        do {
            a1108 = (k1 + ((j1)*(m1)));
            s783 = svld2_f64(pg1, (X + ((2)*(a1108))));
            s784 = s783.v0;
            s785 = s783.v1;
            s786 = svld2_f64(pg1, (X + ((2)*((a1108 + ((l1)*(m1)))))));
            s787 = s786.v0;
            s788 = s786.v1;
            s789 = svld2_f64(pg1, (X + ((2)*((a1108 + ((((2)*(l1)))*(m1)))))));
            s790 = s789.v0;
            s791 = s789.v1;
            s792 = svld2_f64(pg1, (X + ((2)*((a1108 + ((((3)*(l1)))*(m1)))))));
            s793 = s792.v0;
            s794 = s792.v1;
            s795 = svld2_f64(pg1, (X + ((2)*((a1108 + ((((4)*(l1)))*(m1)))))));
            s796 = s795.v0;
            s797 = s795.v1;
            s798 = svld2_f64(pg1, (X + ((2)*((a1108 + ((((5)*(l1)))*(m1)))))));
            s799 = s798.v0;
            s800 = s798.v1;
            s801 = svld2_f64(pg1, (X + ((2)*((a1108 + ((((6)*(l1)))*(m1)))))));
            s802 = s801.v0;
            s803 = s801.v1;
            t1799 = svadd_f64_x(pg1, s790, s796);
            t1800 = svadd_f64_x(pg1, s791, s797);
            t1801 = svadd_f64_x(pg1, s787, t1799);
            t1802 = svadd_f64_x(pg1, s788, t1800);
            t1803 = svmls_n_f64_x(pg1, s787, t1799, 0.5);
            t1804 = svmls_n_f64_x(pg1, s788, t1800, 0.5);
            s804 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s791, s797), 0.8660254037844386);
            s805 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s790, s796), 0.8660254037844386);
            t1805 = svadd_f64_x(pg1, t1803, s804);
            t1806 = svsub_f64_x(pg1, t1804, s805);
            t1807 = svsub_f64_x(pg1, t1803, s804);
            t1808 = svadd_f64_x(pg1, t1804, s805);
            t1809 = svadd_f64_x(pg1, s802, s799);
            t1810 = svadd_f64_x(pg1, s803, s800);
            t1811 = svadd_f64_x(pg1, s793, t1809);
            t1812 = svadd_f64_x(pg1, s794, t1810);
            t1813 = svmls_n_f64_x(pg1, s793, t1809, 0.5);
            t1814 = svmls_n_f64_x(pg1, s794, t1810, 0.5);
            s806 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s803, s800), 0.8660254037844386);
            s807 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s802, s799), 0.8660254037844386);
            t1815 = svadd_f64_x(pg1, t1813, s806);
            t1816 = svsub_f64_x(pg1, t1814, s807);
            t1817 = svsub_f64_x(pg1, t1813, s806);
            t1818 = svadd_f64_x(pg1, t1814, s807);
            s842 = svmla_n_f64_x(pg1, t1815, t1816, 1.7320508075688772);
            s808 = svmul_n_f64_x(pg1, s842, 0.5);
            s843 = svmls_n_f64_x(pg1, t1816, t1815, 1.7320508075688772);
            s809 = svmul_n_f64_x(pg1, s843, 0.5);
            s844 = svmls_n_f64_x(pg1, t1818, t1817, 0.57735026918962584);
            s810 = svmul_n_f64_x(pg1, s844, 0.8660254037844386);
            s845 = svmla_n_f64_x(pg1, t1817, t1818, 0.57735026918962584);
            s811 = svmul_n_f64_x(pg1, s845, 0.8660254037844386);
            t1819 = svadd_f64_x(pg1, t1801, t1811);
            t1820 = svadd_f64_x(pg1, t1802, t1812);
            t1821 = svadd_f64_x(pg1, t1805, s808);
            t1822 = svadd_f64_x(pg1, t1806, s809);
            t1823 = svsub_f64_x(pg1, t1805, s808);
            t1824 = svsub_f64_x(pg1, t1806, s809);
            t1825 = svadd_f64_x(pg1, t1807, s810);
            t1826 = svsub_f64_x(pg1, t1808, s811);
            t1827 = svsub_f64_x(pg1, t1807, s810);
            t1828 = svadd_f64_x(pg1, t1808, s811);
            t1829 = svmls_n_f64_x(pg1, s784, t1819, 0.16666666666666666);
            t1830 = svmls_n_f64_x(pg1, s785, t1820, 0.16666666666666666);
            s846 = svmla_n_f64_x(pg1, t1821, t1822, 0.41908315722758349);
            s812 = svmul_n_f64_x(pg1, s846, 0.4066888930575896);
            s847 = svmls_n_f64_x(pg1, t1822, t1821, 0.41908315722758349);
            s813 = svmul_n_f64_x(pg1, s847, 0.4066888930575896);
            s848 = svmla_n_f64_x(pg1, t1825, t1826, 0.49572725516748389);
            s814 = svmul_n_f64_x(pg1, s848, 0.39507823426270006);
            s849 = svmls_n_f64_x(pg1, t1826, t1825, 0.49572725516748389);
            s815 = svmul_n_f64_x(pg1, s849, 0.39507823426270006);
            s816 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1802, t1812), 0.44095855184409843);
            s817 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1801, t1811), 0.44095855184409843);
            s850 = svmls_n_f64_x(pg1, t1823, t1824, 0.49572725516748389);
            s818 = svmul_n_f64_x(pg1, s850, 0.39507823426270006);
            s851 = svmla_n_f64_x(pg1, t1823, t1824, 2.0172382889501304);
            s819 = svmul_n_f64_x(pg1, s851, 0.1958510486474645);
            s852 = svmls_n_f64_x(pg1, t1828, t1827, 2.386161273135941);
            s820 = svmul_n_f64_x(pg1, s852, 0.17043646531196566);
            s853 = svmla_n_f64_x(pg1, t1827, t1828, 2.386161273135941);
            s821 = svmul_n_f64_x(pg1, s853, 0.17043646531196566);
            t1831 = svadd_f64_x(pg1, s814, s818);
            t1832 = svadd_f64_x(pg1, s815, s819);
            t1833 = svadd_f64_x(pg1, t1829, t1831);
            t1834 = svadd_f64_x(pg1, t1830, t1832);
            t1835 = svmls_n_f64_x(pg1, t1829, t1831, 0.5);
            t1836 = svmls_n_f64_x(pg1, t1830, t1832, 0.5);
            s822 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s815, s819), 0.8660254037844386);
            s823 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s814, s818), 0.8660254037844386);
            t1837 = svadd_f64_x(pg1, t1835, s822);
            t1838 = svsub_f64_x(pg1, t1836, s823);
            t1839 = svsub_f64_x(pg1, t1835, s822);
            t1840 = svadd_f64_x(pg1, t1836, s823);
            t1841 = svadd_f64_x(pg1, s816, s820);
            t1842 = svadd_f64_x(pg1, s817, s821);
            t1843 = svadd_f64_x(pg1, s812, t1841);
            t1844 = svsub_f64_x(pg1, s813, t1842);
            t1845 = svmls_n_f64_x(pg1, s812, t1841, 0.5);
            t1846 = svmla_n_f64_x(pg1, s813, t1842, 0.5);
            s824 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s821, s817), 0.8660254037844386);
            s825 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s816, s820), 0.8660254037844386);
            t1847 = svadd_f64_x(pg1, t1845, s824);
            t1848 = svsub_f64_x(pg1, t1846, s825);
            t1849 = svsub_f64_x(pg1, t1845, s824);
            t1850 = svadd_f64_x(pg1, t1846, s825);
            s854 = svmla_n_f64_x(pg1, t1847, t1848, 1.7320508075688772);
            s826 = svmul_n_f64_x(pg1, s854, 0.5);
            s855 = svmls_n_f64_x(pg1, t1848, t1847, 1.7320508075688772);
            s827 = svmul_n_f64_x(pg1, s855, 0.5);
            s856 = svmls_n_f64_x(pg1, t1850, t1849, 0.57735026918962584);
            s828 = svmul_n_f64_x(pg1, s856, 0.8660254037844386);
            s857 = svmla_n_f64_x(pg1, t1849, t1850, 0.57735026918962584);
            s829 = svmul_n_f64_x(pg1, s857, 0.8660254037844386);
            s830 = svadd_f64_x(pg1, t1833, t1843);
            s831 = svadd_f64_x(pg1, t1834, t1844);
            s832 = svsub_f64_x(pg1, t1833, t1843);
            s833 = svsub_f64_x(pg1, t1834, t1844);
            s834 = svadd_f64_x(pg1, t1837, s826);
            s835 = svadd_f64_x(pg1, t1838, s827);
            s836 = svsub_f64_x(pg1, t1837, s826);
            s837 = svsub_f64_x(pg1, t1838, s827);
            s838 = svadd_f64_x(pg1, t1839, s828);
            s839 = svsub_f64_x(pg1, t1840, s829);
            s840 = svsub_f64_x(pg1, t1839, s828);
            s841 = svadd_f64_x(pg1, t1840, s829);
            a1109 = ((12)*(j1));
            a1110 = TW1[a1109];
            a1111 = TW1[(a1109 + 1)];
            a1112 = TW1[(a1109 + 2)];
            a1113 = TW1[(a1109 + 3)];
            a1114 = TW1[(a1109 + 4)];
            a1115 = TW1[(a1109 + 5)];
            a1116 = TW1[(a1109 + 6)];
            a1117 = TW1[(a1109 + 7)];
            a1118 = TW1[(a1109 + 8)];
            a1119 = TW1[(a1109 + 9)];
            a1120 = TW1[(a1109 + 10)];
            a1121 = TW1[(a1109 + 11)];
            svex2_8.v0 = svadd_f64_x(pg1, s784, t1819);
            svex2_8.v1 = svadd_f64_x(pg1, s785, t1820);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((7)*(j1)))*(m1)))))), svex2_8);
            svex2_9.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s831, a1111), s830, a1110);
            svex2_9.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s831, a1110), s830, a1111);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((7)*(j1)))*(m1)) + m1)))), svex2_9);
            svex2_10.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s839, a1113), s838, a1112);
            svex2_10.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s839, a1112), s838, a1113);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((7)*(j1)))*(m1)) + ((2)*(m1)))))), svex2_10);
            svex2_11.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s835, a1115), s834, a1114);
            svex2_11.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s835, a1114), s834, a1115);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((7)*(j1)))*(m1)) + ((3)*(m1)))))), svex2_11);
            svex2_12.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s837, a1117), s836, a1116);
            svex2_12.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s837, a1116), s836, a1117);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((7)*(j1)))*(m1)) + ((4)*(m1)))))), svex2_12);
            svex2_13.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s841, a1119), s840, a1118);
            svex2_13.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s841, a1118), s840, a1119);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((7)*(j1)))*(m1)) + ((5)*(m1)))))), svex2_13);
            svex2_14.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s833, a1121), s832, a1120);
            svex2_14.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s833, a1120), s832, a1121);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((7)*(j1)))*(m1)) + ((6)*(m1)))))), svex2_14);
            k1 += svcntd();
            pg1 = svwhilelt_b64(k1, m1);
        } while(svptest_any(svptrue_b64(), pg1));
    }
}
