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


void dft30b_(double  *Y, double  *X, double  *TW1, int  *lp1, int  *mp1) {
    static double D73[12];
    static double D74[48];
    double a9187, a9188, a9189, a9190, a9191, a9192, a9193, a9194, 
            a9195, a9196, a9197, a9198, a9199, a9200, a9201, a9202, 
            a9203, a9204, a9205, a9206, a9207, a9208, a9209, a9210, 
            a9211, a9212, a9213, a9214, a9215, a9216, a9217, a9218, 
            a9219, a9220, a9221, a9222, a9223, a9224, a9225, a9226, 
            a9227, a9228, a9229, a9230, a9231, a9232, a9233, a9234, 
            a9235, a9236, a9237, a9238, a9239, a9240, a9241, a9242, 
            a9243, a9244, s3968, s3969, s3970, s3971, s3972, s3973, 
            s3974, s3975, s3976, s3977, s3978, s3979, s3980, s3981, 
            s3982, s3983, s3984, s3985, s3986, s3987, s3988, s3989, 
            s3990, s3991, s3992, s3993, s3994, s3995, s3996, s3997, 
            s3998, s3999, s4000, s4001, s4002, s4003, s4004, s4005, 
            s4006, s4007, s4008, s4009, s4010, s4011, s4012, s4013, 
            s4014, s4015, s4016, s4017, s4018, s4019, s4020, s4021, 
            s4022, s4023, s4024, s4025, s4026, s4027, s4028, s4029, 
            s4030, s4031, s4032, s4033, s4034, s4035, s4036, s4037, 
            s4038, s4039, s4040, s4041, s4042, s4043, s4044, s4045, 
            s4046, s4047, s4048, s4049, s4050, s4051, s4052, s4053, 
            s4054, s4055, s4056, s4057, s4058, s4059, s4060, s4061, 
            s4062, s4063, s4064, s4065, s4066, s4067, s4068, s4069, 
            s4070, s4071, s4072, s4073, s4074, s4075, s4076, s4077, 
            s4078, s4079, s4080, s4081, s4082, s4083, s4084, s4085, 
            s4086, s4087, s4088, s4089, s4090, s4091, s4092, s4093, 
            s4094, s4095, s4096, s4097, s4098, s4099, s4100, s4101, 
            s4102, s4103, s4104, s4105, s4106, s4107, s4108, s4109, 
            s4110, s4111, s4112, s4113, s4114, s4115, s4116, s4117, 
            s4118, s4119, s4120, s4121, s4122, s4123, s4124, s4125, 
            s4126, s4127, s4128, s4129, s4130, s4131, s4132, s4133, 
            s4134, s4135, s4136, s4137, s4138, s4139, s4140, s4141, 
            s4142, s4143, s4144, s4145, s4146, s4147, s4148, s4149, 
            s4150, s4151, s4152, s4153, s4154, s4155, s4156, s4157, 
            s4158, s4159, s4160, s4161, s4162, s4163, s4164, s4165, 
            s4166, s4167, s4168, s4169, s4170, s4171, s4172, s4173, 
            s4174, s4175, s4176, s4177, s4178, s4179, s4180, s4181, 
            s4182, s4183, s4184, s4185, s4186, s4187, s4188, s4189, 
            s4190, s4191, s4192, s4193, s4194, s4195, s4196, s4197, 
            s4198, s4199, s4200, s4201, s4202, s4203, s4204, s4205, 
            s4206, s4207, s4208, s4209, s4210, s4211, s4212, s4213, 
            s4214, s4215, s4216, s4217, s4218, s4219, s4220, s4221, 
            t7118, t7119, t7120, t7121, t7122, t7123, t7124, t7125, 
            t7126, t7127, t7128, t7129, t7130, t7131, t7132, t7133, 
            t7134, t7135, t7136, t7137, t7138, t7139, t7140, t7141, 
            t7142, t7143, t7144, t7145, t7146, t7147, t7148, t7149, 
            t7150, t7151, t7152, t7153, t7154, t7155, t7156, t7157, 
            t7158, t7159, t7160, t7161, t7162, t7163, t7164, t7165, 
            t7166, t7167, t7168, t7169, t7170, t7171, t7172, t7173, 
            t7174, t7175, t7176, t7177, t7178, t7179, t7180, t7181, 
            t7182, t7183, t7184, t7185, t7186, t7187, t7188, t7189, 
            t7190, t7191, t7192, t7193, t7194, t7195, t7196, t7197, 
            t7198, t7199, t7200, t7201, t7202, t7203, t7204, t7205, 
            t7206, t7207, t7208, t7209, t7210, t7211, t7212, t7213, 
            t7214, t7215, t7216, t7217, t7218, t7219, t7220, t7221, 
            t7222, t7223, t7224, t7225, t7226, t7227, t7228, t7229, 
            t7230, t7231, t7232, t7233, t7234, t7235, t7236, t7237, 
            t7238, t7239, t7240, t7241, t7242, t7243, t7244, t7245, 
            t7246, t7247, t7248, t7249, t7250, t7251, t7252, t7253, 
            t7254, t7255, t7256, t7257, t7258, t7259, t7260, t7261, 
            t7262, t7263, t7264, t7265, t7266, t7267, t7268, t7269, 
            t7270, t7271, t7272, t7273, t7274, t7275, t7276, t7277, 
            t7278, t7279, t7280, t7281, t7282, t7283, t7284, t7285, 
            t7286, t7287, t7288, t7289, t7290, t7291, t7292, t7293, 
            t7294, t7295, t7296, t7297, t7298, t7299, t7300, t7301, 
            t7302, t7303, t7304, t7305, t7306, t7307, t7308, t7309, 
            t7310, t7311, t7312, t7313, t7314, t7315, t7316, t7317, 
            t7318, t7319, t7320, t7321, t7322, t7323, t7324, t7325, 
            t7326, t7327, t7328, t7329, t7330, t7331, t7332, t7333, 
            t7334, t7335, t7336, t7337, t7338, t7339, t7340, t7341, 
            t7342, t7343, t7344, t7345, t7346, t7347, t7348, t7349, 
            t7350, t7351, t7352, t7353, t7354, t7355, t7356, t7357, 
            t7358, t7359, t7360, t7361, t7362, t7363, t7364, t7365, 
            t7366, t7367, t7368, t7369, t7370, t7371, t7372, t7373, 
            t7374, t7375, t7376, t7377, t7378, t7379, t7380, t7381, 
            t7382, t7383, t7384, t7385, t7386, t7387, t7388, t7389, 
            t7390, t7391, t7392, t7393, t7394, t7395, t7396, t7397, 
            t7398, t7399, t7400, t7401, t7402, t7403, t7404, t7405, 
            t7406, t7407, t7408, t7409, t7410, t7411, t7412, t7413, 
            t7414, t7415, t7416, t7417, t7418, t7419, t7420, t7421;
    int a9154, a9155, a9156, a9157, a9158, a9159, a9160, a9161, 
            a9162, a9163, a9164, a9165, a9166, a9167, a9168, a9169, 
            a9170, a9171, a9172, a9173, a9174, a9175, a9176, a9177, 
            a9178, a9179, a9180, a9181, a9182, a9183, a9184, a9185, 
            a9186, a9245, a9246, a9247, a9248, a9249, a9250, a9251, 
            a9252, a9253, a9254, a9255, a9256, a9257, a9258, a9259, 
            a9260, a9261, a9262, a9263, a9264, a9265, a9266, a9267, 
            a9268, a9269, a9270, a9271, a9272, a9273, a9274, a9275, 
            b451, j1, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        for(int k1 = 0; k1 < m1; k1++) {
            a9154 = (2*k1);
            a9155 = (j1*m1);
            a9156 = (a9154 + (2*a9155));
            s3968 = X[a9156];
            s3969 = X[(a9156 + 1)];
            b451 = (l1*m1);
            a9157 = (a9156 + (2*b451));
            s3970 = X[a9157];
            s3971 = X[(a9157 + 1)];
            a9158 = (a9156 + (4*b451));
            s3972 = X[a9158];
            s3973 = X[(a9158 + 1)];
            a9159 = (a9156 + (6*b451));
            s3974 = X[a9159];
            s3975 = X[(a9159 + 1)];
            a9160 = (a9156 + (8*b451));
            s3976 = X[a9160];
            s3977 = X[(a9160 + 1)];
            a9161 = (a9156 + (10*b451));
            s3978 = X[a9161];
            s3979 = X[(a9161 + 1)];
            a9162 = (a9156 + (12*b451));
            s3980 = X[a9162];
            s3981 = X[(a9162 + 1)];
            a9163 = (a9156 + (14*b451));
            s3982 = X[a9163];
            s3983 = X[(a9163 + 1)];
            a9164 = (a9156 + (16*b451));
            s3984 = X[a9164];
            s3985 = X[(a9164 + 1)];
            a9165 = (a9156 + (18*b451));
            s3986 = X[a9165];
            s3987 = X[(a9165 + 1)];
            a9166 = (a9156 + (20*b451));
            s3988 = X[a9166];
            s3989 = X[(a9166 + 1)];
            a9167 = (a9156 + (22*b451));
            s3990 = X[a9167];
            s3991 = X[(a9167 + 1)];
            a9168 = (a9156 + (24*b451));
            s3992 = X[a9168];
            s3993 = X[(a9168 + 1)];
            a9169 = (a9156 + (26*b451));
            s3994 = X[a9169];
            s3995 = X[(a9169 + 1)];
            a9170 = (a9156 + (28*b451));
            s3996 = X[a9170];
            s3997 = X[(a9170 + 1)];
            a9171 = (a9156 + (30*b451));
            s3998 = X[a9171];
            s3999 = X[(a9171 + 1)];
            a9172 = (a9156 + (32*b451));
            s4000 = X[a9172];
            s4001 = X[(a9172 + 1)];
            a9173 = (a9156 + (34*b451));
            s4002 = X[a9173];
            s4003 = X[(a9173 + 1)];
            a9174 = (a9156 + (36*b451));
            s4004 = X[a9174];
            s4005 = X[(a9174 + 1)];
            a9175 = (a9156 + (38*b451));
            s4006 = X[a9175];
            s4007 = X[(a9175 + 1)];
            a9176 = (a9156 + (40*b451));
            s4008 = X[a9176];
            s4009 = X[(a9176 + 1)];
            a9177 = (a9156 + (42*b451));
            s4010 = X[a9177];
            s4011 = X[(a9177 + 1)];
            a9178 = (a9156 + (44*b451));
            s4012 = X[a9178];
            s4013 = X[(a9178 + 1)];
            a9179 = (a9156 + (46*b451));
            s4014 = X[a9179];
            s4015 = X[(a9179 + 1)];
            a9180 = (a9156 + (48*b451));
            s4016 = X[a9180];
            s4017 = X[(a9180 + 1)];
            a9181 = (a9156 + (50*b451));
            s4018 = X[a9181];
            s4019 = X[(a9181 + 1)];
            a9182 = (a9156 + (52*b451));
            s4020 = X[a9182];
            s4021 = X[(a9182 + 1)];
            a9183 = (a9156 + (54*b451));
            s4022 = X[a9183];
            s4023 = X[(a9183 + 1)];
            a9184 = (a9156 + (56*b451));
            s4024 = X[a9184];
            s4025 = X[(a9184 + 1)];
            a9185 = (a9156 + (58*b451));
            s4026 = X[a9185];
            s4027 = X[(a9185 + 1)];
            t7118 = (s3980 + s4016);
            t7119 = (s3981 + s4017);
            t7120 = (s3980 - s4016);
            t7121 = (s3981 - s4017);
            t7122 = (s3992 + s4004);
            t7123 = (s3993 + s4005);
            t7124 = (s3992 - s4004);
            t7125 = (s3993 - s4005);
            t7126 = (t7118 + t7122);
            t7127 = (t7119 + t7123);
            t7128 = (t7120 + t7125);
            t7129 = (t7121 - t7124);
            t7130 = (t7120 - t7125);
            t7131 = (t7121 + t7124);
            t7132 = (s3968 + t7126);
            t7133 = (s3969 + t7127);
            t7134 = (s3968 - (0.25*t7126));
            t7135 = (s3969 - (0.25*t7127));
            s4028 = ((0.29389262614623657*t7128) + (0.47552825814757677*t7129));
            s4029 = ((0.29389262614623657*t7129) - (0.47552825814757677*t7128));
            s4030 = (0.55901699437494745*(t7118 - t7122));
            s4031 = (0.55901699437494745*(t7119 - t7123));
            s4032 = ((0.47552825814757682*t7131) - (0.29389262614623657*t7130));
            s4033 = ((0.47552825814757682*t7130) + (0.29389262614623657*t7131));
            s4034 = ((D73[0]*t7132) - (D73[1]*t7133));
            s4035 = ((D73[1]*t7132) + (D73[0]*t7133));
            t7136 = (t7134 + s4030);
            t7137 = (t7135 + s4031);
            t7138 = (t7134 - s4030);
            t7139 = (t7135 - s4031);
            t7140 = (s4028 + s4032);
            t7141 = (s4029 - s4033);
            t7142 = (s4028 - s4032);
            t7143 = (s4029 + s4033);
            t7144 = (t7136 + t7140);
            t7145 = (t7137 + t7141);
            t7146 = (t7136 - t7140);
            t7147 = (t7137 - t7141);
            s4036 = ((D74[0]*t7144) - (D74[1]*t7145));
            s4037 = ((D74[1]*t7144) + (D74[0]*t7145));
            s4038 = ((D74[2]*t7146) - (D74[3]*t7147));
            s4039 = ((D74[3]*t7146) + (D74[2]*t7147));
            t7148 = (t7138 + t7143);
            t7149 = (t7139 - t7142);
            t7150 = (t7138 - t7143);
            t7151 = (t7139 + t7142);
            s4040 = ((D74[4]*t7148) - (D74[5]*t7149));
            s4041 = ((D74[5]*t7148) + (D74[4]*t7149));
            s4042 = ((D74[6]*t7150) - (D74[7]*t7151));
            s4043 = ((D74[7]*t7150) + (D74[6]*t7151));
            t7152 = (s3982 + s4018);
            t7153 = (s3983 + s4019);
            t7154 = (s3982 - s4018);
            t7155 = (s3983 - s4019);
            t7156 = (s3994 + s4006);
            t7157 = (s3995 + s4007);
            t7158 = (s3994 - s4006);
            t7159 = (s3995 - s4007);
            t7160 = (t7152 + t7156);
            t7161 = (t7153 + t7157);
            t7162 = (t7154 + t7159);
            t7163 = (t7155 - t7158);
            t7164 = (t7154 - t7159);
            t7165 = (t7155 + t7158);
            t7166 = (s3970 + t7160);
            t7167 = (s3971 + t7161);
            t7168 = (s3970 - (0.25*t7160));
            t7169 = (s3971 - (0.25*t7161));
            s4044 = ((0.29389262614623657*t7162) + (0.47552825814757677*t7163));
            s4045 = ((0.29389262614623657*t7163) - (0.47552825814757677*t7162));
            s4046 = (0.55901699437494745*(t7152 - t7156));
            s4047 = (0.55901699437494745*(t7153 - t7157));
            s4048 = ((0.47552825814757682*t7165) - (0.29389262614623657*t7164));
            s4049 = ((0.47552825814757682*t7164) + (0.29389262614623657*t7165));
            s4050 = ((D73[2]*t7166) - (D73[3]*t7167));
            s4051 = ((D73[3]*t7166) + (D73[2]*t7167));
            t7170 = (t7168 + s4046);
            t7171 = (t7169 + s4047);
            t7172 = (t7168 - s4046);
            t7173 = (t7169 - s4047);
            t7174 = (s4044 + s4048);
            t7175 = (s4045 - s4049);
            t7176 = (s4044 - s4048);
            t7177 = (s4045 + s4049);
            t7178 = (t7170 + t7174);
            t7179 = (t7171 + t7175);
            t7180 = (t7170 - t7174);
            t7181 = (t7171 - t7175);
            s4052 = ((D74[8]*t7178) - (D74[9]*t7179));
            s4053 = ((D74[9]*t7178) + (D74[8]*t7179));
            s4054 = ((D74[10]*t7180) - (D74[11]*t7181));
            s4055 = ((D74[11]*t7180) + (D74[10]*t7181));
            t7182 = (t7172 + t7177);
            t7183 = (t7173 - t7176);
            t7184 = (t7172 - t7177);
            t7185 = (t7173 + t7176);
            s4056 = ((D74[12]*t7182) - (D74[13]*t7183));
            s4057 = ((D74[13]*t7182) + (D74[12]*t7183));
            s4058 = ((D74[14]*t7184) - (D74[15]*t7185));
            s4059 = ((D74[15]*t7184) + (D74[14]*t7185));
            t7186 = (s3984 + s4020);
            t7187 = (s3985 + s4021);
            t7188 = (s3984 - s4020);
            t7189 = (s3985 - s4021);
            t7190 = (s3996 + s4008);
            t7191 = (s3997 + s4009);
            t7192 = (s3996 - s4008);
            t7193 = (s3997 - s4009);
            t7194 = (t7186 + t7190);
            t7195 = (t7187 + t7191);
            t7196 = (t7188 + t7193);
            t7197 = (t7189 - t7192);
            t7198 = (t7188 - t7193);
            t7199 = (t7189 + t7192);
            t7200 = (s3972 + t7194);
            t7201 = (s3973 + t7195);
            t7202 = (s3972 - (0.25*t7194));
            t7203 = (s3973 - (0.25*t7195));
            s4060 = ((0.29389262614623657*t7196) + (0.47552825814757677*t7197));
            s4061 = ((0.29389262614623657*t7197) - (0.47552825814757677*t7196));
            s4062 = (0.55901699437494745*(t7186 - t7190));
            s4063 = (0.55901699437494745*(t7187 - t7191));
            s4064 = ((0.47552825814757682*t7199) - (0.29389262614623657*t7198));
            s4065 = ((0.47552825814757682*t7198) + (0.29389262614623657*t7199));
            s4066 = ((D73[4]*t7200) - (D73[5]*t7201));
            s4067 = ((D73[5]*t7200) + (D73[4]*t7201));
            t7204 = (t7202 + s4062);
            t7205 = (t7203 + s4063);
            t7206 = (t7202 - s4062);
            t7207 = (t7203 - s4063);
            t7208 = (s4060 + s4064);
            t7209 = (s4061 - s4065);
            t7210 = (s4060 - s4064);
            t7211 = (s4061 + s4065);
            t7212 = (t7204 + t7208);
            t7213 = (t7205 + t7209);
            t7214 = (t7204 - t7208);
            t7215 = (t7205 - t7209);
            s4068 = ((D74[16]*t7212) - (D74[17]*t7213));
            s4069 = ((D74[17]*t7212) + (D74[16]*t7213));
            s4070 = ((D74[18]*t7214) - (D74[19]*t7215));
            s4071 = ((D74[19]*t7214) + (D74[18]*t7215));
            t7216 = (t7206 + t7211);
            t7217 = (t7207 - t7210);
            t7218 = (t7206 - t7211);
            t7219 = (t7207 + t7210);
            s4072 = ((D74[20]*t7216) - (D74[21]*t7217));
            s4073 = ((D74[21]*t7216) + (D74[20]*t7217));
            s4074 = ((D74[22]*t7218) - (D74[23]*t7219));
            s4075 = ((D74[23]*t7218) + (D74[22]*t7219));
            t7220 = (s3986 + s4022);
            t7221 = (s3987 + s4023);
            t7222 = (s3986 - s4022);
            t7223 = (s3987 - s4023);
            t7224 = (s3998 + s4010);
            t7225 = (s3999 + s4011);
            t7226 = (s3998 - s4010);
            t7227 = (s3999 - s4011);
            t7228 = (t7220 + t7224);
            t7229 = (t7221 + t7225);
            t7230 = (t7222 + t7227);
            t7231 = (t7223 - t7226);
            t7232 = (t7222 - t7227);
            t7233 = (t7223 + t7226);
            t7234 = (s3974 + t7228);
            t7235 = (s3975 + t7229);
            t7236 = (s3974 - (0.25*t7228));
            t7237 = (s3975 - (0.25*t7229));
            s4076 = ((0.29389262614623657*t7230) + (0.47552825814757677*t7231));
            s4077 = ((0.29389262614623657*t7231) - (0.47552825814757677*t7230));
            s4078 = (0.55901699437494745*(t7220 - t7224));
            s4079 = (0.55901699437494745*(t7221 - t7225));
            s4080 = ((0.47552825814757682*t7233) - (0.29389262614623657*t7232));
            s4081 = ((0.47552825814757682*t7232) + (0.29389262614623657*t7233));
            s4082 = ((D73[6]*t7234) - (D73[7]*t7235));
            s4083 = ((D73[7]*t7234) + (D73[6]*t7235));
            t7238 = (t7236 + s4078);
            t7239 = (t7237 + s4079);
            t7240 = (t7236 - s4078);
            t7241 = (t7237 - s4079);
            t7242 = (s4076 + s4080);
            t7243 = (s4077 - s4081);
            t7244 = (s4076 - s4080);
            t7245 = (s4077 + s4081);
            t7246 = (t7238 + t7242);
            t7247 = (t7239 + t7243);
            t7248 = (t7238 - t7242);
            t7249 = (t7239 - t7243);
            s4084 = ((D74[24]*t7246) - (D74[25]*t7247));
            s4085 = ((D74[25]*t7246) + (D74[24]*t7247));
            s4086 = ((D74[26]*t7248) - (D74[27]*t7249));
            s4087 = ((D74[27]*t7248) + (D74[26]*t7249));
            t7250 = (t7240 + t7245);
            t7251 = (t7241 - t7244);
            t7252 = (t7240 - t7245);
            t7253 = (t7241 + t7244);
            s4088 = ((D74[28]*t7250) - (D74[29]*t7251));
            s4089 = ((D74[29]*t7250) + (D74[28]*t7251));
            s4090 = ((D74[30]*t7252) - (D74[31]*t7253));
            s4091 = ((D74[31]*t7252) + (D74[30]*t7253));
            t7254 = (s3988 + s4024);
            t7255 = (s3989 + s4025);
            t7256 = (s3988 - s4024);
            t7257 = (s3989 - s4025);
            t7258 = (s4000 + s4012);
            t7259 = (s4001 + s4013);
            t7260 = (s4000 - s4012);
            t7261 = (s4001 - s4013);
            t7262 = (t7254 + t7258);
            t7263 = (t7255 + t7259);
            t7264 = (t7256 + t7261);
            t7265 = (t7257 - t7260);
            t7266 = (t7256 - t7261);
            t7267 = (t7257 + t7260);
            t7268 = (s3976 + t7262);
            t7269 = (s3977 + t7263);
            t7270 = (s3976 - (0.25*t7262));
            t7271 = (s3977 - (0.25*t7263));
            s4092 = ((0.29389262614623657*t7264) + (0.47552825814757677*t7265));
            s4093 = ((0.29389262614623657*t7265) - (0.47552825814757677*t7264));
            s4094 = (0.55901699437494745*(t7254 - t7258));
            s4095 = (0.55901699437494745*(t7255 - t7259));
            s4096 = ((0.47552825814757682*t7267) - (0.29389262614623657*t7266));
            s4097 = ((0.47552825814757682*t7266) + (0.29389262614623657*t7267));
            s4098 = ((D73[8]*t7268) - (D73[9]*t7269));
            s4099 = ((D73[9]*t7268) + (D73[8]*t7269));
            t7272 = (t7270 + s4094);
            t7273 = (t7271 + s4095);
            t7274 = (t7270 - s4094);
            t7275 = (t7271 - s4095);
            t7276 = (s4092 + s4096);
            t7277 = (s4093 - s4097);
            t7278 = (s4092 - s4096);
            t7279 = (s4093 + s4097);
            t7280 = (t7272 + t7276);
            t7281 = (t7273 + t7277);
            t7282 = (t7272 - t7276);
            t7283 = (t7273 - t7277);
            s4100 = ((D74[32]*t7280) - (D74[33]*t7281));
            s4101 = ((D74[33]*t7280) + (D74[32]*t7281));
            s4102 = ((D74[34]*t7282) - (D74[35]*t7283));
            s4103 = ((D74[35]*t7282) + (D74[34]*t7283));
            t7284 = (t7274 + t7279);
            t7285 = (t7275 - t7278);
            t7286 = (t7274 - t7279);
            t7287 = (t7275 + t7278);
            s4104 = ((D74[36]*t7284) - (D74[37]*t7285));
            s4105 = ((D74[37]*t7284) + (D74[36]*t7285));
            s4106 = ((D74[38]*t7286) - (D74[39]*t7287));
            s4107 = ((D74[39]*t7286) + (D74[38]*t7287));
            t7288 = (s3990 + s4026);
            t7289 = (s3991 + s4027);
            t7290 = (s3990 - s4026);
            t7291 = (s3991 - s4027);
            t7292 = (s4002 + s4014);
            t7293 = (s4003 + s4015);
            t7294 = (s4002 - s4014);
            t7295 = (s4003 - s4015);
            t7296 = (t7288 + t7292);
            t7297 = (t7289 + t7293);
            t7298 = (t7290 + t7295);
            t7299 = (t7291 - t7294);
            t7300 = (t7290 - t7295);
            t7301 = (t7291 + t7294);
            t7302 = (s3978 + t7296);
            t7303 = (s3979 + t7297);
            t7304 = (s3978 - (0.25*t7296));
            t7305 = (s3979 - (0.25*t7297));
            s4108 = ((0.29389262614623657*t7298) + (0.47552825814757677*t7299));
            s4109 = ((0.29389262614623657*t7299) - (0.47552825814757677*t7298));
            s4110 = (0.55901699437494745*(t7288 - t7292));
            s4111 = (0.55901699437494745*(t7289 - t7293));
            s4112 = ((0.47552825814757682*t7301) - (0.29389262614623657*t7300));
            s4113 = ((0.47552825814757682*t7300) + (0.29389262614623657*t7301));
            s4114 = ((D73[10]*t7302) - (D73[11]*t7303));
            s4115 = ((D73[11]*t7302) + (D73[10]*t7303));
            t7306 = (t7304 + s4110);
            t7307 = (t7305 + s4111);
            t7308 = (t7304 - s4110);
            t7309 = (t7305 - s4111);
            t7310 = (s4108 + s4112);
            t7311 = (s4109 - s4113);
            t7312 = (s4108 - s4112);
            t7313 = (s4109 + s4113);
            t7314 = (t7306 + t7310);
            t7315 = (t7307 + t7311);
            t7316 = (t7306 - t7310);
            t7317 = (t7307 - t7311);
            s4116 = ((D74[40]*t7314) - (D74[41]*t7315));
            s4117 = ((D74[41]*t7314) + (D74[40]*t7315));
            s4118 = ((D74[42]*t7316) - (D74[43]*t7317));
            s4119 = ((D74[43]*t7316) + (D74[42]*t7317));
            t7318 = (t7308 + t7313);
            t7319 = (t7309 - t7312);
            t7320 = (t7308 - t7313);
            t7321 = (t7309 + t7312);
            s4120 = ((D74[44]*t7318) - (D74[45]*t7319));
            s4121 = ((D74[45]*t7318) + (D74[44]*t7319));
            s4122 = ((D74[46]*t7320) - (D74[47]*t7321));
            s4123 = ((D74[47]*t7320) + (D74[46]*t7321));
            t7322 = (s4066 + s4098);
            t7323 = (s4067 + s4099);
            t7324 = (s4034 + t7322);
            t7325 = (s4035 + t7323);
            t7326 = (s4034 - (0.5*t7322));
            t7327 = (s4035 - (0.5*t7323));
            s4124 = (0.8660254037844386*(s4067 - s4099));
            s4125 = (0.8660254037844386*(s4066 - s4098));
            t7328 = (t7326 + s4124);
            t7329 = (t7327 - s4125);
            t7330 = (t7326 - s4124);
            t7331 = (t7327 + s4125);
            t7332 = (s4082 + s4114);
            t7333 = (s4083 + s4115);
            t7334 = (s4050 + t7332);
            t7335 = (s4051 + t7333);
            t7336 = (s4050 - (0.5*t7332));
            t7337 = (s4051 - (0.5*t7333));
            s4126 = (0.8660254037844386*(s4083 - s4115));
            s4127 = (0.8660254037844386*(s4082 - s4114));
            t7338 = (t7336 + s4126);
            t7339 = (t7337 - s4127);
            t7340 = (t7336 - s4126);
            t7341 = (t7337 + s4127);
            s4128 = ((0.5*t7338) + (0.8660254037844386*t7339));
            s4129 = ((0.5*t7339) - (0.8660254037844386*t7338));
            s4130 = ((0.8660254037844386*t7341) - (0.5*t7340));
            s4131 = ((0.8660254037844386*t7340) + (0.5*t7341));
            s4132 = (t7324 - t7334);
            s4133 = (t7325 - t7335);
            s4134 = (t7328 + s4128);
            s4135 = (t7329 + s4129);
            s4136 = (t7328 - s4128);
            s4137 = (t7329 - s4129);
            s4138 = (t7330 + s4130);
            s4139 = (t7331 - s4131);
            s4140 = (t7330 - s4130);
            s4141 = (t7331 + s4131);
            t7342 = (s4068 + s4100);
            t7343 = (s4069 + s4101);
            t7344 = (s4036 + t7342);
            t7345 = (s4037 + t7343);
            t7346 = (s4036 - (0.5*t7342));
            t7347 = (s4037 - (0.5*t7343));
            s4142 = (0.8660254037844386*(s4069 - s4101));
            s4143 = (0.8660254037844386*(s4068 - s4100));
            t7348 = (t7346 + s4142);
            t7349 = (t7347 - s4143);
            t7350 = (t7346 - s4142);
            t7351 = (t7347 + s4143);
            t7352 = (s4084 + s4116);
            t7353 = (s4085 + s4117);
            t7354 = (s4052 + t7352);
            t7355 = (s4053 + t7353);
            t7356 = (s4052 - (0.5*t7352));
            t7357 = (s4053 - (0.5*t7353));
            s4144 = (0.8660254037844386*(s4085 - s4117));
            s4145 = (0.8660254037844386*(s4084 - s4116));
            t7358 = (t7356 + s4144);
            t7359 = (t7357 - s4145);
            t7360 = (t7356 - s4144);
            t7361 = (t7357 + s4145);
            s4146 = ((0.5*t7358) + (0.8660254037844386*t7359));
            s4147 = ((0.5*t7359) - (0.8660254037844386*t7358));
            s4148 = ((0.8660254037844386*t7361) - (0.5*t7360));
            s4149 = ((0.8660254037844386*t7360) + (0.5*t7361));
            s4150 = (t7344 + t7354);
            s4151 = (t7345 + t7355);
            s4152 = (t7344 - t7354);
            s4153 = (t7345 - t7355);
            s4154 = (t7348 + s4146);
            s4155 = (t7349 + s4147);
            s4156 = (t7348 - s4146);
            s4157 = (t7349 - s4147);
            s4158 = (t7350 + s4148);
            s4159 = (t7351 - s4149);
            s4160 = (t7350 - s4148);
            s4161 = (t7351 + s4149);
            t7362 = (s4072 + s4104);
            t7363 = (s4073 + s4105);
            t7364 = (s4040 + t7362);
            t7365 = (s4041 + t7363);
            t7366 = (s4040 - (0.5*t7362));
            t7367 = (s4041 - (0.5*t7363));
            s4162 = (0.8660254037844386*(s4073 - s4105));
            s4163 = (0.8660254037844386*(s4072 - s4104));
            t7368 = (t7366 + s4162);
            t7369 = (t7367 - s4163);
            t7370 = (t7366 - s4162);
            t7371 = (t7367 + s4163);
            t7372 = (s4088 + s4120);
            t7373 = (s4089 + s4121);
            t7374 = (s4056 + t7372);
            t7375 = (s4057 + t7373);
            t7376 = (s4056 - (0.5*t7372));
            t7377 = (s4057 - (0.5*t7373));
            s4164 = (0.8660254037844386*(s4089 - s4121));
            s4165 = (0.8660254037844386*(s4088 - s4120));
            t7378 = (t7376 + s4164);
            t7379 = (t7377 - s4165);
            t7380 = (t7376 - s4164);
            t7381 = (t7377 + s4165);
            s4166 = ((0.5*t7378) + (0.8660254037844386*t7379));
            s4167 = ((0.5*t7379) - (0.8660254037844386*t7378));
            s4168 = ((0.8660254037844386*t7381) - (0.5*t7380));
            s4169 = ((0.8660254037844386*t7380) + (0.5*t7381));
            s4170 = (t7364 + t7374);
            s4171 = (t7365 + t7375);
            s4172 = (t7364 - t7374);
            s4173 = (t7365 - t7375);
            s4174 = (t7368 + s4166);
            s4175 = (t7369 + s4167);
            s4176 = (t7368 - s4166);
            s4177 = (t7369 - s4167);
            s4178 = (t7370 + s4168);
            s4179 = (t7371 - s4169);
            s4180 = (t7370 - s4168);
            s4181 = (t7371 + s4169);
            t7382 = (s4074 + s4106);
            t7383 = (s4075 + s4107);
            t7384 = (s4042 + t7382);
            t7385 = (s4043 + t7383);
            t7386 = (s4042 - (0.5*t7382));
            t7387 = (s4043 - (0.5*t7383));
            s4182 = (0.8660254037844386*(s4075 - s4107));
            s4183 = (0.8660254037844386*(s4074 - s4106));
            t7388 = (t7386 + s4182);
            t7389 = (t7387 - s4183);
            t7390 = (t7386 - s4182);
            t7391 = (t7387 + s4183);
            t7392 = (s4090 + s4122);
            t7393 = (s4091 + s4123);
            t7394 = (s4058 + t7392);
            t7395 = (s4059 + t7393);
            t7396 = (s4058 - (0.5*t7392));
            t7397 = (s4059 - (0.5*t7393));
            s4184 = (0.8660254037844386*(s4091 - s4123));
            s4185 = (0.8660254037844386*(s4090 - s4122));
            t7398 = (t7396 + s4184);
            t7399 = (t7397 - s4185);
            t7400 = (t7396 - s4184);
            t7401 = (t7397 + s4185);
            s4186 = ((0.5*t7398) + (0.8660254037844386*t7399));
            s4187 = ((0.5*t7399) - (0.8660254037844386*t7398));
            s4188 = ((0.8660254037844386*t7401) - (0.5*t7400));
            s4189 = ((0.8660254037844386*t7400) + (0.5*t7401));
            s4190 = (t7384 + t7394);
            s4191 = (t7385 + t7395);
            s4192 = (t7384 - t7394);
            s4193 = (t7385 - t7395);
            s4194 = (t7388 + s4186);
            s4195 = (t7389 + s4187);
            s4196 = (t7388 - s4186);
            s4197 = (t7389 - s4187);
            s4198 = (t7390 + s4188);
            s4199 = (t7391 - s4189);
            s4200 = (t7390 - s4188);
            s4201 = (t7391 + s4189);
            t7402 = (s4070 + s4102);
            t7403 = (s4071 + s4103);
            t7404 = (s4038 + t7402);
            t7405 = (s4039 + t7403);
            t7406 = (s4038 - (0.5*t7402));
            t7407 = (s4039 - (0.5*t7403));
            s4202 = (0.8660254037844386*(s4071 - s4103));
            s4203 = (0.8660254037844386*(s4070 - s4102));
            t7408 = (t7406 + s4202);
            t7409 = (t7407 - s4203);
            t7410 = (t7406 - s4202);
            t7411 = (t7407 + s4203);
            t7412 = (s4086 + s4118);
            t7413 = (s4087 + s4119);
            t7414 = (s4054 + t7412);
            t7415 = (s4055 + t7413);
            t7416 = (s4054 - (0.5*t7412));
            t7417 = (s4055 - (0.5*t7413));
            s4204 = (0.8660254037844386*(s4087 - s4119));
            s4205 = (0.8660254037844386*(s4086 - s4118));
            t7418 = (t7416 + s4204);
            t7419 = (t7417 - s4205);
            t7420 = (t7416 - s4204);
            t7421 = (t7417 + s4205);
            s4206 = ((0.5*t7418) + (0.8660254037844386*t7419));
            s4207 = ((0.5*t7419) - (0.8660254037844386*t7418));
            s4208 = ((0.8660254037844386*t7421) - (0.5*t7420));
            s4209 = ((0.8660254037844386*t7420) + (0.5*t7421));
            s4210 = (t7404 + t7414);
            s4211 = (t7405 + t7415);
            s4212 = (t7404 - t7414);
            s4213 = (t7405 - t7415);
            s4214 = (t7408 + s4206);
            s4215 = (t7409 + s4207);
            s4216 = (t7408 - s4206);
            s4217 = (t7409 - s4207);
            s4218 = (t7410 + s4208);
            s4219 = (t7411 - s4209);
            s4220 = (t7410 - s4208);
            s4221 = (t7411 + s4209);
            a9186 = (58*j1);
            a9187 = TW1[a9186];
            a9188 = TW1[(a9186 + 1)];
            a9189 = TW1[(a9186 + 2)];
            a9190 = TW1[(a9186 + 3)];
            a9191 = TW1[(a9186 + 4)];
            a9192 = TW1[(a9186 + 5)];
            a9193 = TW1[(a9186 + 6)];
            a9194 = TW1[(a9186 + 7)];
            a9195 = TW1[(a9186 + 8)];
            a9196 = TW1[(a9186 + 9)];
            a9197 = TW1[(a9186 + 10)];
            a9198 = TW1[(a9186 + 11)];
            a9199 = TW1[(a9186 + 12)];
            a9200 = TW1[(a9186 + 13)];
            a9201 = TW1[(a9186 + 14)];
            a9202 = TW1[(a9186 + 15)];
            a9203 = TW1[(a9186 + 16)];
            a9204 = TW1[(a9186 + 17)];
            a9205 = TW1[(a9186 + 18)];
            a9206 = TW1[(a9186 + 19)];
            a9207 = TW1[(a9186 + 20)];
            a9208 = TW1[(a9186 + 21)];
            a9209 = TW1[(a9186 + 22)];
            a9210 = TW1[(a9186 + 23)];
            a9211 = TW1[(a9186 + 24)];
            a9212 = TW1[(a9186 + 25)];
            a9213 = TW1[(a9186 + 26)];
            a9214 = TW1[(a9186 + 27)];
            a9215 = TW1[(a9186 + 28)];
            a9216 = TW1[(a9186 + 29)];
            a9217 = TW1[(a9186 + 30)];
            a9218 = TW1[(a9186 + 31)];
            a9219 = TW1[(a9186 + 32)];
            a9220 = TW1[(a9186 + 33)];
            a9221 = TW1[(a9186 + 34)];
            a9222 = TW1[(a9186 + 35)];
            a9223 = TW1[(a9186 + 36)];
            a9224 = TW1[(a9186 + 37)];
            a9225 = TW1[(a9186 + 38)];
            a9226 = TW1[(a9186 + 39)];
            a9227 = TW1[(a9186 + 40)];
            a9228 = TW1[(a9186 + 41)];
            a9229 = TW1[(a9186 + 42)];
            a9230 = TW1[(a9186 + 43)];
            a9231 = TW1[(a9186 + 44)];
            a9232 = TW1[(a9186 + 45)];
            a9233 = TW1[(a9186 + 46)];
            a9234 = TW1[(a9186 + 47)];
            a9235 = TW1[(a9186 + 48)];
            a9236 = TW1[(a9186 + 49)];
            a9237 = TW1[(a9186 + 50)];
            a9238 = TW1[(a9186 + 51)];
            a9239 = TW1[(a9186 + 52)];
            a9240 = TW1[(a9186 + 53)];
            a9241 = TW1[(a9186 + 54)];
            a9242 = TW1[(a9186 + 55)];
            a9243 = TW1[(a9186 + 56)];
            a9244 = TW1[(a9186 + 57)];
            a9245 = (60*a9155);
            a9246 = (a9154 + a9245);
            Y[a9246] = (t7324 + t7334);
            Y[(a9246 + 1)] = (t7325 + t7335);
            a9247 = (a9154 + (2*m1) + a9245);
            Y[a9247] = ((a9187*s4150) - (a9188*s4151));
            Y[(a9247 + 1)] = ((a9188*s4150) + (a9187*s4151));
            a9248 = (a9154 + (4*m1) + a9245);
            Y[a9248] = ((a9189*s4170) - (a9190*s4171));
            Y[(a9248 + 1)] = ((a9190*s4170) + (a9189*s4171));
            a9249 = (a9154 + (6*m1) + a9245);
            Y[a9249] = ((a9191*s4190) - (a9192*s4191));
            Y[(a9249 + 1)] = ((a9192*s4190) + (a9191*s4191));
            a9250 = (a9154 + (8*m1) + a9245);
            Y[a9250] = ((a9193*s4210) - (a9194*s4211));
            Y[(a9250 + 1)] = ((a9194*s4210) + (a9193*s4211));
            a9251 = (a9154 + (10*m1) + a9245);
            Y[a9251] = ((a9195*s4134) - (a9196*s4135));
            Y[(a9251 + 1)] = ((a9196*s4134) + (a9195*s4135));
            a9252 = (a9154 + (12*m1) + a9245);
            Y[a9252] = ((a9197*s4154) - (a9198*s4155));
            Y[(a9252 + 1)] = ((a9198*s4154) + (a9197*s4155));
            a9253 = (a9154 + (14*m1) + a9245);
            Y[a9253] = ((a9199*s4174) - (a9200*s4175));
            Y[(a9253 + 1)] = ((a9200*s4174) + (a9199*s4175));
            a9254 = (a9154 + (16*m1) + a9245);
            Y[a9254] = ((a9201*s4194) - (a9202*s4195));
            Y[(a9254 + 1)] = ((a9202*s4194) + (a9201*s4195));
            a9255 = (a9154 + (18*m1) + a9245);
            Y[a9255] = ((a9203*s4214) - (a9204*s4215));
            Y[(a9255 + 1)] = ((a9204*s4214) + (a9203*s4215));
            a9256 = (a9154 + (20*m1) + a9245);
            Y[a9256] = ((a9205*s4138) - (a9206*s4139));
            Y[(a9256 + 1)] = ((a9206*s4138) + (a9205*s4139));
            a9257 = (a9154 + (22*m1) + a9245);
            Y[a9257] = ((a9207*s4158) - (a9208*s4159));
            Y[(a9257 + 1)] = ((a9208*s4158) + (a9207*s4159));
            a9258 = (a9154 + (24*m1) + a9245);
            Y[a9258] = ((a9209*s4178) - (a9210*s4179));
            Y[(a9258 + 1)] = ((a9210*s4178) + (a9209*s4179));
            a9259 = (a9154 + (26*m1) + a9245);
            Y[a9259] = ((a9211*s4198) - (a9212*s4199));
            Y[(a9259 + 1)] = ((a9212*s4198) + (a9211*s4199));
            a9260 = (a9154 + (28*m1) + a9245);
            Y[a9260] = ((a9213*s4218) - (a9214*s4219));
            Y[(a9260 + 1)] = ((a9214*s4218) + (a9213*s4219));
            a9261 = (a9154 + (30*m1) + a9245);
            Y[a9261] = ((a9215*s4132) - (a9216*s4133));
            Y[(a9261 + 1)] = ((a9216*s4132) + (a9215*s4133));
            a9262 = (a9154 + (32*m1) + a9245);
            Y[a9262] = ((a9217*s4152) - (a9218*s4153));
            Y[(a9262 + 1)] = ((a9218*s4152) + (a9217*s4153));
            a9263 = (a9154 + (34*m1) + a9245);
            Y[a9263] = ((a9219*s4172) - (a9220*s4173));
            Y[(a9263 + 1)] = ((a9220*s4172) + (a9219*s4173));
            a9264 = (a9154 + (36*m1) + a9245);
            Y[a9264] = ((a9221*s4192) - (a9222*s4193));
            Y[(a9264 + 1)] = ((a9222*s4192) + (a9221*s4193));
            a9265 = (a9154 + (38*m1) + a9245);
            Y[a9265] = ((a9223*s4212) - (a9224*s4213));
            Y[(a9265 + 1)] = ((a9224*s4212) + (a9223*s4213));
            a9266 = (a9154 + (40*m1) + a9245);
            Y[a9266] = ((a9225*s4136) - (a9226*s4137));
            Y[(a9266 + 1)] = ((a9226*s4136) + (a9225*s4137));
            a9267 = (a9154 + (42*m1) + a9245);
            Y[a9267] = ((a9227*s4156) - (a9228*s4157));
            Y[(a9267 + 1)] = ((a9228*s4156) + (a9227*s4157));
            a9268 = (a9154 + (44*m1) + a9245);
            Y[a9268] = ((a9229*s4176) - (a9230*s4177));
            Y[(a9268 + 1)] = ((a9230*s4176) + (a9229*s4177));
            a9269 = (a9154 + (46*m1) + a9245);
            Y[a9269] = ((a9231*s4196) - (a9232*s4197));
            Y[(a9269 + 1)] = ((a9232*s4196) + (a9231*s4197));
            a9270 = (a9154 + (48*m1) + a9245);
            Y[a9270] = ((a9233*s4216) - (a9234*s4217));
            Y[(a9270 + 1)] = ((a9234*s4216) + (a9233*s4217));
            a9271 = (a9154 + (50*m1) + a9245);
            Y[a9271] = ((a9235*s4140) - (a9236*s4141));
            Y[(a9271 + 1)] = ((a9236*s4140) + (a9235*s4141));
            a9272 = (a9154 + (52*m1) + a9245);
            Y[a9272] = ((a9237*s4160) - (a9238*s4161));
            Y[(a9272 + 1)] = ((a9238*s4160) + (a9237*s4161));
            a9273 = (a9154 + (54*m1) + a9245);
            Y[a9273] = ((a9239*s4180) - (a9240*s4181));
            Y[(a9273 + 1)] = ((a9240*s4180) + (a9239*s4181));
            a9274 = (a9154 + (56*m1) + a9245);
            Y[a9274] = ((a9241*s4200) - (a9242*s4201));
            Y[(a9274 + 1)] = ((a9242*s4200) + (a9241*s4201));
            a9275 = (a9154 + (58*m1) + a9245);
            Y[a9275] = ((a9243*s4220) - (a9244*s4221));
            Y[(a9275 + 1)] = ((a9244*s4220) + (a9243*s4221));
        }
    }
}
