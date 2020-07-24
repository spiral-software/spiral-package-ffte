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


void dft7c_(double  *Y, double  *X, int  *lp1, int  *mp1) {
    double s670, s671, s672, s673, s674, s675, s676, s677, 
            s678, s679, s680, s681, s682, s683, s684, s685, 
            s686, s687, s688, s689, s690, s691, s692, s693, 
            s694, s695, s696, s697, s698, s699, s700, s701, 
            s702, s703, s704, s705, s706, s707, s708, s709, 
            t1798, t1799, t1800, t1801, t1802, t1803, t1804, t1805, 
            t1806, t1807, t1808, t1809, t1810, t1811, t1812, t1813, 
            t1814, t1815, t1816, t1817, t1818, t1819, t1820, t1821, 
            t1822, t1823, t1824, t1825, t1826, t1827, t1828, t1829, 
            t1830, t1831, t1832, t1833, t1834, t1835, t1836, t1837, 
            t1838, t1839, t1840, t1841, t1842, t1843, t1844, t1845, 
            t1846, t1847, t1848, t1849;
    int a903, a904, a905, a906, a907, a908, a909, a910, 
            a911, a912, a913, a914, a915, a916, b31, l1, 
            m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int k1 = 0; k1 < m1; k1++) {
        a903 = (2*k1);
        s670 = X[a903];
        a904 = (a903 + 1);
        s671 = X[a904];
        b31 = (l1*m1);
        a905 = (a903 + (2*b31));
        s672 = X[a905];
        s673 = X[(a905 + 1)];
        a906 = (a903 + (4*b31));
        s674 = X[a906];
        s675 = X[(a906 + 1)];
        a907 = (a903 + (6*b31));
        s676 = X[a907];
        s677 = X[(a907 + 1)];
        a908 = (a903 + (8*b31));
        s678 = X[a908];
        s679 = X[(a908 + 1)];
        a909 = (a903 + (10*b31));
        s680 = X[a909];
        s681 = X[(a909 + 1)];
        a910 = (a903 + (12*b31));
        s682 = X[a910];
        s683 = X[(a910 + 1)];
        t1798 = (s674 + s678);
        t1799 = (s675 + s679);
        t1800 = (s672 + t1798);
        t1801 = (s673 + t1799);
        t1802 = (s672 - (0.5*t1798));
        t1803 = (s673 - (0.5*t1799));
        s684 = (0.8660254037844386*(s675 - s679));
        s685 = (0.8660254037844386*(s674 - s678));
        t1804 = (t1802 + s684);
        t1805 = (t1803 - s685);
        t1806 = (t1802 - s684);
        t1807 = (t1803 + s685);
        t1808 = (s682 + s680);
        t1809 = (s683 + s681);
        t1810 = (s676 + t1808);
        t1811 = (s677 + t1809);
        t1812 = (s676 - (0.5*t1808));
        t1813 = (s677 - (0.5*t1809));
        s686 = (0.8660254037844386*(s683 - s681));
        s687 = (0.8660254037844386*(s682 - s680));
        t1814 = (t1812 + s686);
        t1815 = (t1813 - s687);
        t1816 = (t1812 - s686);
        t1817 = (t1813 + s687);
        s688 = ((0.5*t1814) + (0.8660254037844386*t1815));
        s689 = ((0.5*t1815) - (0.8660254037844386*t1814));
        s690 = ((0.8660254037844386*t1817) - (0.5*t1816));
        s691 = ((0.8660254037844386*t1816) + (0.5*t1817));
        t1818 = (t1800 + t1810);
        t1819 = (t1801 + t1811);
        t1820 = (t1804 + s688);
        t1821 = (t1805 + s689);
        t1822 = (t1804 - s688);
        t1823 = (t1805 - s689);
        t1824 = (t1806 + s690);
        t1825 = (t1807 - s691);
        t1826 = (t1806 - s690);
        t1827 = (t1807 + s691);
        t1828 = (s670 - (0.16666666666666666*t1818));
        t1829 = (s671 - (0.16666666666666666*t1819));
        s692 = ((0.4066888930575896*t1820) + (0.17043646531196571*t1821));
        s693 = ((0.4066888930575896*t1821) - (0.17043646531196571*t1820));
        s694 = ((0.39507823426270006*t1824) + (0.1958510486474645*t1825));
        s695 = ((0.39507823426270006*t1825) - (0.1958510486474645*t1824));
        s696 = (0.44095855184409843*(t1801 - t1811));
        s697 = (0.44095855184409843*(t1800 - t1810));
        s698 = ((0.39507823426270006*t1822) - (0.1958510486474645*t1823));
        s699 = ((0.1958510486474645*t1822) + (0.39507823426270006*t1823));
        s700 = ((0.17043646531196566*t1827) - (0.4066888930575896*t1826));
        s701 = ((0.17043646531196566*t1826) + (0.4066888930575896*t1827));
        t1830 = (s694 + s698);
        t1831 = (s695 + s699);
        t1832 = (t1828 + t1830);
        t1833 = (t1829 + t1831);
        t1834 = (t1828 - (0.5*t1830));
        t1835 = (t1829 - (0.5*t1831));
        s702 = (0.8660254037844386*(s695 - s699));
        s703 = (0.8660254037844386*(s694 - s698));
        t1836 = (t1834 + s702);
        t1837 = (t1835 - s703);
        t1838 = (t1834 - s702);
        t1839 = (t1835 + s703);
        t1840 = (s696 + s700);
        t1841 = (s697 + s701);
        t1842 = (s692 + t1840);
        t1843 = (s693 - t1841);
        t1844 = (s692 - (0.5*t1840));
        t1845 = (s693 + (0.5*t1841));
        s704 = (0.8660254037844386*(s701 - s697));
        s705 = (0.8660254037844386*(s696 - s700));
        t1846 = (t1844 + s704);
        t1847 = (t1845 - s705);
        t1848 = (t1844 - s704);
        t1849 = (t1845 + s705);
        s706 = ((0.5*t1846) + (0.8660254037844386*t1847));
        s707 = ((0.5*t1847) - (0.8660254037844386*t1846));
        s708 = ((0.8660254037844386*t1849) - (0.5*t1848));
        s709 = ((0.8660254037844386*t1848) + (0.5*t1849));
        Y[a903] = (s670 + t1818);
        Y[a904] = (s671 + t1819);
        a911 = (a903 + (2*m1));
        Y[a911] = (t1832 + t1842);
        Y[(a911 + 1)] = (t1833 + t1843);
        a912 = (a903 + (4*m1));
        Y[a912] = (t1838 + s708);
        Y[(a912 + 1)] = (t1839 - s709);
        a913 = (a903 + (6*m1));
        Y[a913] = (t1836 + s706);
        Y[(a913 + 1)] = (t1837 + s707);
        a914 = (a903 + (8*m1));
        Y[a914] = (t1836 - s706);
        Y[(a914 + 1)] = (t1837 - s707);
        a915 = (a903 + (10*m1));
        Y[a915] = (t1838 - s708);
        Y[(a915 + 1)] = (t1839 + s709);
        a916 = (a903 + (12*m1));
        Y[a916] = (t1832 - t1842);
        Y[(a916 + 1)] = (t1833 - t1843);
    }
}