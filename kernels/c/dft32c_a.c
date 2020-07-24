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


void dft32a_(double  *Y, double  *X, double  *TW1, int  *lp1) {
    static double D15[64];
    double a10525, a10526, a10527, a10528, a10529, a10530, a10531, a10532, 
            a10533, a10534, a10535, a10536, a10537, a10538, a10539, a10540, 
            a10542, a10543, a10544, a10545, a10546, a10547, a10548, a10549, 
            a10550, a10551, a10552, a10553, a10554, a10555, a10556, a10557, 
            a10558, a10559, a10560, a10561, a10562, a10563, a10564, a10565, 
            a10566, a10567, a10568, a10569, a10570, a10571, a10572, a10573, 
            a10574, a10575, a10576, a10577, a10578, a10579, a10580, a10581, 
            a10582, a10583, a10584, a10585, a10586, a10587, a10588, a10589, 
            a10590, a10591, a10592, a10593, a10594, a10595, a10596, a10597, 
            a10598, a10599, a10600, a10601, a10602, a10603, s3988, s3989, 
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
            s4190, s4191, s4192, s4193, t7130, t7131, t7132, t7133, 
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
            t7366, t7367, t7368, t7369;
    int a10493, a10494, a10495, a10496, a10497, a10498, a10499, a10500, 
            a10501, a10502, a10503, a10504, a10505, a10506, a10507, a10508, 
            a10509, a10510, a10511, a10512, a10513, a10514, a10515, a10516, 
            a10517, a10518, a10519, a10520, a10521, a10522, a10523, a10524, 
            a10541, a10604, l1;
    l1 = *(lp1);
    for(int j1 = 0; j1 < l1; j1++) {
        l1 = *(lp1);
        a10493 = (2*j1);
        s3988 = X[a10493];
        s3989 = X[(a10493 + 1)];
        a10494 = (a10493 + (2*l1));
        s3990 = X[a10494];
        s3991 = X[(a10494 + 1)];
        a10495 = (a10493 + (4*l1));
        s3992 = X[a10495];
        s3993 = X[(a10495 + 1)];
        a10496 = (a10493 + (6*l1));
        s3994 = X[a10496];
        s3995 = X[(a10496 + 1)];
        a10497 = (a10493 + (8*l1));
        s3996 = X[a10497];
        s3997 = X[(a10497 + 1)];
        a10498 = (a10493 + (10*l1));
        s3998 = X[a10498];
        s3999 = X[(a10498 + 1)];
        a10499 = (a10493 + (12*l1));
        s4000 = X[a10499];
        s4001 = X[(a10499 + 1)];
        a10500 = (a10493 + (14*l1));
        s4002 = X[a10500];
        s4003 = X[(a10500 + 1)];
        a10501 = (a10493 + (16*l1));
        s4004 = X[a10501];
        s4005 = X[(a10501 + 1)];
        a10502 = (a10493 + (18*l1));
        s4006 = X[a10502];
        s4007 = X[(a10502 + 1)];
        a10503 = (a10493 + (20*l1));
        s4008 = X[a10503];
        s4009 = X[(a10503 + 1)];
        a10504 = (a10493 + (22*l1));
        s4010 = X[a10504];
        s4011 = X[(a10504 + 1)];
        a10505 = (a10493 + (24*l1));
        s4012 = X[a10505];
        s4013 = X[(a10505 + 1)];
        a10506 = (a10493 + (26*l1));
        s4014 = X[a10506];
        s4015 = X[(a10506 + 1)];
        a10507 = (a10493 + (28*l1));
        s4016 = X[a10507];
        s4017 = X[(a10507 + 1)];
        a10508 = (a10493 + (30*l1));
        s4018 = X[a10508];
        s4019 = X[(a10508 + 1)];
        a10509 = (a10493 + (32*l1));
        s4020 = X[a10509];
        s4021 = X[(a10509 + 1)];
        a10510 = (a10493 + (34*l1));
        s4022 = X[a10510];
        s4023 = X[(a10510 + 1)];
        a10511 = (a10493 + (36*l1));
        s4024 = X[a10511];
        s4025 = X[(a10511 + 1)];
        a10512 = (a10493 + (38*l1));
        s4026 = X[a10512];
        s4027 = X[(a10512 + 1)];
        a10513 = (a10493 + (40*l1));
        s4028 = X[a10513];
        s4029 = X[(a10513 + 1)];
        a10514 = (a10493 + (42*l1));
        s4030 = X[a10514];
        s4031 = X[(a10514 + 1)];
        a10515 = (a10493 + (44*l1));
        s4032 = X[a10515];
        s4033 = X[(a10515 + 1)];
        a10516 = (a10493 + (46*l1));
        s4034 = X[a10516];
        s4035 = X[(a10516 + 1)];
        a10517 = (a10493 + (48*l1));
        s4036 = X[a10517];
        s4037 = X[(a10517 + 1)];
        a10518 = (a10493 + (50*l1));
        s4038 = X[a10518];
        s4039 = X[(a10518 + 1)];
        a10519 = (a10493 + (52*l1));
        s4040 = X[a10519];
        s4041 = X[(a10519 + 1)];
        a10520 = (a10493 + (54*l1));
        s4042 = X[a10520];
        s4043 = X[(a10520 + 1)];
        a10521 = (a10493 + (56*l1));
        s4044 = X[a10521];
        s4045 = X[(a10521 + 1)];
        a10522 = (a10493 + (58*l1));
        s4046 = X[a10522];
        s4047 = X[(a10522 + 1)];
        a10523 = (a10493 + (60*l1));
        s4048 = X[a10523];
        s4049 = X[(a10523 + 1)];
        a10524 = (a10493 + (62*l1));
        s4050 = X[a10524];
        s4051 = X[(a10524 + 1)];
        t7130 = (s3988 + s4020);
        t7131 = (s3989 + s4021);
        t7132 = (s3988 - s4020);
        t7133 = (s3989 - s4021);
        t7134 = (s4004 + s4036);
        t7135 = (s4005 + s4037);
        t7136 = (s4004 - s4036);
        t7137 = (s4005 - s4037);
        t7138 = (t7130 + t7134);
        t7139 = (t7131 + t7135);
        t7140 = (t7130 - t7134);
        t7141 = (t7131 - t7135);
        s4052 = ((D15[0]*t7138) - (D15[1]*t7139));
        s4053 = ((D15[1]*t7138) + (D15[0]*t7139));
        s4054 = ((D15[2]*t7140) - (D15[3]*t7141));
        s4055 = ((D15[3]*t7140) + (D15[2]*t7141));
        t7142 = (t7132 + t7137);
        t7143 = (t7133 - t7136);
        t7144 = (t7132 - t7137);
        t7145 = (t7133 + t7136);
        s4056 = ((D15[4]*t7142) - (D15[5]*t7143));
        s4057 = ((D15[5]*t7142) + (D15[4]*t7143));
        s4058 = ((D15[6]*t7144) - (D15[7]*t7145));
        s4059 = ((D15[7]*t7144) + (D15[6]*t7145));
        t7146 = (s3990 + s4022);
        t7147 = (s3991 + s4023);
        t7148 = (s3990 - s4022);
        t7149 = (s3991 - s4023);
        t7150 = (s4006 + s4038);
        t7151 = (s4007 + s4039);
        t7152 = (s4006 - s4038);
        t7153 = (s4007 - s4039);
        t7154 = (t7146 + t7150);
        t7155 = (t7147 + t7151);
        t7156 = (t7146 - t7150);
        t7157 = (t7147 - t7151);
        s4060 = ((D15[8]*t7154) - (D15[9]*t7155));
        s4061 = ((D15[9]*t7154) + (D15[8]*t7155));
        s4062 = ((D15[10]*t7156) - (D15[11]*t7157));
        s4063 = ((D15[11]*t7156) + (D15[10]*t7157));
        t7158 = (t7148 + t7153);
        t7159 = (t7149 - t7152);
        t7160 = (t7148 - t7153);
        t7161 = (t7149 + t7152);
        s4064 = ((D15[12]*t7158) - (D15[13]*t7159));
        s4065 = ((D15[13]*t7158) + (D15[12]*t7159));
        s4066 = ((D15[14]*t7160) - (D15[15]*t7161));
        s4067 = ((D15[15]*t7160) + (D15[14]*t7161));
        t7162 = (s3992 + s4024);
        t7163 = (s3993 + s4025);
        t7164 = (s3992 - s4024);
        t7165 = (s3993 - s4025);
        t7166 = (s4008 + s4040);
        t7167 = (s4009 + s4041);
        t7168 = (s4008 - s4040);
        t7169 = (s4009 - s4041);
        t7170 = (t7162 + t7166);
        t7171 = (t7163 + t7167);
        t7172 = (t7162 - t7166);
        t7173 = (t7163 - t7167);
        s4068 = ((D15[16]*t7170) - (D15[17]*t7171));
        s4069 = ((D15[17]*t7170) + (D15[16]*t7171));
        s4070 = ((D15[18]*t7172) - (D15[19]*t7173));
        s4071 = ((D15[19]*t7172) + (D15[18]*t7173));
        t7174 = (t7164 + t7169);
        t7175 = (t7165 - t7168);
        t7176 = (t7164 - t7169);
        t7177 = (t7165 + t7168);
        s4072 = ((D15[20]*t7174) - (D15[21]*t7175));
        s4073 = ((D15[21]*t7174) + (D15[20]*t7175));
        s4074 = ((D15[22]*t7176) - (D15[23]*t7177));
        s4075 = ((D15[23]*t7176) + (D15[22]*t7177));
        t7178 = (s3994 + s4026);
        t7179 = (s3995 + s4027);
        t7180 = (s3994 - s4026);
        t7181 = (s3995 - s4027);
        t7182 = (s4010 + s4042);
        t7183 = (s4011 + s4043);
        t7184 = (s4010 - s4042);
        t7185 = (s4011 - s4043);
        t7186 = (t7178 + t7182);
        t7187 = (t7179 + t7183);
        t7188 = (t7178 - t7182);
        t7189 = (t7179 - t7183);
        s4076 = ((D15[24]*t7186) - (D15[25]*t7187));
        s4077 = ((D15[25]*t7186) + (D15[24]*t7187));
        s4078 = ((D15[26]*t7188) - (D15[27]*t7189));
        s4079 = ((D15[27]*t7188) + (D15[26]*t7189));
        t7190 = (t7180 + t7185);
        t7191 = (t7181 - t7184);
        t7192 = (t7180 - t7185);
        t7193 = (t7181 + t7184);
        s4080 = ((D15[28]*t7190) - (D15[29]*t7191));
        s4081 = ((D15[29]*t7190) + (D15[28]*t7191));
        s4082 = ((D15[30]*t7192) - (D15[31]*t7193));
        s4083 = ((D15[31]*t7192) + (D15[30]*t7193));
        t7194 = (s3996 + s4028);
        t7195 = (s3997 + s4029);
        t7196 = (s3996 - s4028);
        t7197 = (s3997 - s4029);
        t7198 = (s4012 + s4044);
        t7199 = (s4013 + s4045);
        t7200 = (s4012 - s4044);
        t7201 = (s4013 - s4045);
        t7202 = (t7194 + t7198);
        t7203 = (t7195 + t7199);
        t7204 = (t7194 - t7198);
        t7205 = (t7195 - t7199);
        s4084 = ((D15[32]*t7202) - (D15[33]*t7203));
        s4085 = ((D15[33]*t7202) + (D15[32]*t7203));
        s4086 = ((D15[34]*t7204) - (D15[35]*t7205));
        s4087 = ((D15[35]*t7204) + (D15[34]*t7205));
        t7206 = (t7196 + t7201);
        t7207 = (t7197 - t7200);
        t7208 = (t7196 - t7201);
        t7209 = (t7197 + t7200);
        s4088 = ((D15[36]*t7206) - (D15[37]*t7207));
        s4089 = ((D15[37]*t7206) + (D15[36]*t7207));
        s4090 = ((D15[38]*t7208) - (D15[39]*t7209));
        s4091 = ((D15[39]*t7208) + (D15[38]*t7209));
        t7210 = (s3998 + s4030);
        t7211 = (s3999 + s4031);
        t7212 = (s3998 - s4030);
        t7213 = (s3999 - s4031);
        t7214 = (s4014 + s4046);
        t7215 = (s4015 + s4047);
        t7216 = (s4014 - s4046);
        t7217 = (s4015 - s4047);
        t7218 = (t7210 + t7214);
        t7219 = (t7211 + t7215);
        t7220 = (t7210 - t7214);
        t7221 = (t7211 - t7215);
        s4092 = ((D15[40]*t7218) - (D15[41]*t7219));
        s4093 = ((D15[41]*t7218) + (D15[40]*t7219));
        s4094 = ((D15[42]*t7220) - (D15[43]*t7221));
        s4095 = ((D15[43]*t7220) + (D15[42]*t7221));
        t7222 = (t7212 + t7217);
        t7223 = (t7213 - t7216);
        t7224 = (t7212 - t7217);
        t7225 = (t7213 + t7216);
        s4096 = ((D15[44]*t7222) - (D15[45]*t7223));
        s4097 = ((D15[45]*t7222) + (D15[44]*t7223));
        s4098 = ((D15[46]*t7224) - (D15[47]*t7225));
        s4099 = ((D15[47]*t7224) + (D15[46]*t7225));
        t7226 = (s4000 + s4032);
        t7227 = (s4001 + s4033);
        t7228 = (s4000 - s4032);
        t7229 = (s4001 - s4033);
        t7230 = (s4016 + s4048);
        t7231 = (s4017 + s4049);
        t7232 = (s4016 - s4048);
        t7233 = (s4017 - s4049);
        t7234 = (t7226 + t7230);
        t7235 = (t7227 + t7231);
        t7236 = (t7226 - t7230);
        t7237 = (t7227 - t7231);
        s4100 = ((D15[48]*t7234) - (D15[49]*t7235));
        s4101 = ((D15[49]*t7234) + (D15[48]*t7235));
        s4102 = ((D15[50]*t7236) - (D15[51]*t7237));
        s4103 = ((D15[51]*t7236) + (D15[50]*t7237));
        t7238 = (t7228 + t7233);
        t7239 = (t7229 - t7232);
        t7240 = (t7228 - t7233);
        t7241 = (t7229 + t7232);
        s4104 = ((D15[52]*t7238) - (D15[53]*t7239));
        s4105 = ((D15[53]*t7238) + (D15[52]*t7239));
        s4106 = ((D15[54]*t7240) - (D15[55]*t7241));
        s4107 = ((D15[55]*t7240) + (D15[54]*t7241));
        t7242 = (s4002 + s4034);
        t7243 = (s4003 + s4035);
        t7244 = (s4002 - s4034);
        t7245 = (s4003 - s4035);
        t7246 = (s4018 + s4050);
        t7247 = (s4019 + s4051);
        t7248 = (s4018 - s4050);
        t7249 = (s4019 - s4051);
        t7250 = (t7242 + t7246);
        t7251 = (t7243 + t7247);
        t7252 = (t7242 - t7246);
        t7253 = (t7243 - t7247);
        s4108 = ((D15[56]*t7250) - (D15[57]*t7251));
        s4109 = ((D15[57]*t7250) + (D15[56]*t7251));
        s4110 = ((D15[58]*t7252) - (D15[59]*t7253));
        s4111 = ((D15[59]*t7252) + (D15[58]*t7253));
        t7254 = (t7244 + t7249);
        t7255 = (t7245 - t7248);
        t7256 = (t7244 - t7249);
        t7257 = (t7245 + t7248);
        s4112 = ((D15[60]*t7254) - (D15[61]*t7255));
        s4113 = ((D15[61]*t7254) + (D15[60]*t7255));
        s4114 = ((D15[62]*t7256) - (D15[63]*t7257));
        s4115 = ((D15[63]*t7256) + (D15[62]*t7257));
        t7258 = (s4052 + s4084);
        t7259 = (s4053 + s4085);
        t7260 = (s4052 - s4084);
        t7261 = (s4053 - s4085);
        t7262 = (s4068 + s4100);
        t7263 = (s4069 + s4101);
        t7264 = (s4068 - s4100);
        t7265 = (s4069 - s4101);
        t7266 = (t7258 + t7262);
        t7267 = (t7259 + t7263);
        t7268 = (t7258 - t7262);
        t7269 = (t7259 - t7263);
        t7270 = (t7260 + t7265);
        t7271 = (t7261 - t7264);
        t7272 = (t7260 - t7265);
        t7273 = (t7261 + t7264);
        t7274 = (s4060 + s4092);
        t7275 = (s4061 + s4093);
        t7276 = (s4060 - s4092);
        t7277 = (s4061 - s4093);
        t7278 = (s4076 + s4108);
        t7279 = (s4077 + s4109);
        t7280 = (s4076 - s4108);
        t7281 = (s4077 - s4109);
        t7282 = (t7274 + t7278);
        t7283 = (t7275 + t7279);
        t7284 = (t7274 - t7278);
        t7285 = (t7275 - t7279);
        a10525 = (0.70710678118654757*(t7276 + t7281));
        a10526 = (0.70710678118654757*(t7277 - t7280));
        s4116 = (a10525 + a10526);
        s4117 = (a10526 - a10525);
        a10527 = (0.70710678118654757*(t7277 + t7280));
        a10528 = (0.70710678118654757*(t7276 - t7281));
        s4118 = (a10527 - a10528);
        s4119 = (a10528 + a10527);
        s4120 = (t7266 - t7282);
        s4121 = (t7267 - t7283);
        s4122 = (t7270 + s4116);
        s4123 = (t7271 + s4117);
        s4124 = (t7270 - s4116);
        s4125 = (t7271 - s4117);
        s4126 = (t7268 + t7285);
        s4127 = (t7269 - t7284);
        s4128 = (t7268 - t7285);
        s4129 = (t7269 + t7284);
        s4130 = (t7272 + s4118);
        s4131 = (t7273 - s4119);
        s4132 = (t7272 - s4118);
        s4133 = (t7273 + s4119);
        t7286 = (s4056 + s4088);
        t7287 = (s4057 + s4089);
        t7288 = (s4056 - s4088);
        t7289 = (s4057 - s4089);
        t7290 = (s4072 + s4104);
        t7291 = (s4073 + s4105);
        t7292 = (s4072 - s4104);
        t7293 = (s4073 - s4105);
        t7294 = (t7286 + t7290);
        t7295 = (t7287 + t7291);
        t7296 = (t7286 - t7290);
        t7297 = (t7287 - t7291);
        t7298 = (t7288 + t7293);
        t7299 = (t7289 - t7292);
        t7300 = (t7288 - t7293);
        t7301 = (t7289 + t7292);
        t7302 = (s4064 + s4096);
        t7303 = (s4065 + s4097);
        t7304 = (s4064 - s4096);
        t7305 = (s4065 - s4097);
        t7306 = (s4080 + s4112);
        t7307 = (s4081 + s4113);
        t7308 = (s4080 - s4112);
        t7309 = (s4081 - s4113);
        t7310 = (t7302 + t7306);
        t7311 = (t7303 + t7307);
        t7312 = (t7302 - t7306);
        t7313 = (t7303 - t7307);
        a10529 = (0.70710678118654757*(t7304 + t7309));
        a10530 = (0.70710678118654757*(t7305 - t7308));
        s4134 = (a10529 + a10530);
        s4135 = (a10530 - a10529);
        a10531 = (0.70710678118654757*(t7305 + t7308));
        a10532 = (0.70710678118654757*(t7304 - t7309));
        s4136 = (a10531 - a10532);
        s4137 = (a10532 + a10531);
        s4138 = (t7294 + t7310);
        s4139 = (t7295 + t7311);
        s4140 = (t7294 - t7310);
        s4141 = (t7295 - t7311);
        s4142 = (t7298 + s4134);
        s4143 = (t7299 + s4135);
        s4144 = (t7298 - s4134);
        s4145 = (t7299 - s4135);
        s4146 = (t7296 + t7313);
        s4147 = (t7297 - t7312);
        s4148 = (t7296 - t7313);
        s4149 = (t7297 + t7312);
        s4150 = (t7300 + s4136);
        s4151 = (t7301 - s4137);
        s4152 = (t7300 - s4136);
        s4153 = (t7301 + s4137);
        t7314 = (s4054 + s4086);
        t7315 = (s4055 + s4087);
        t7316 = (s4054 - s4086);
        t7317 = (s4055 - s4087);
        t7318 = (s4070 + s4102);
        t7319 = (s4071 + s4103);
        t7320 = (s4070 - s4102);
        t7321 = (s4071 - s4103);
        t7322 = (t7314 + t7318);
        t7323 = (t7315 + t7319);
        t7324 = (t7314 - t7318);
        t7325 = (t7315 - t7319);
        t7326 = (t7316 + t7321);
        t7327 = (t7317 - t7320);
        t7328 = (t7316 - t7321);
        t7329 = (t7317 + t7320);
        t7330 = (s4062 + s4094);
        t7331 = (s4063 + s4095);
        t7332 = (s4062 - s4094);
        t7333 = (s4063 - s4095);
        t7334 = (s4078 + s4110);
        t7335 = (s4079 + s4111);
        t7336 = (s4078 - s4110);
        t7337 = (s4079 - s4111);
        t7338 = (t7330 + t7334);
        t7339 = (t7331 + t7335);
        t7340 = (t7330 - t7334);
        t7341 = (t7331 - t7335);
        a10533 = (0.70710678118654757*(t7332 + t7337));
        a10534 = (0.70710678118654757*(t7333 - t7336));
        s4154 = (a10533 + a10534);
        s4155 = (a10534 - a10533);
        a10535 = (0.70710678118654757*(t7333 + t7336));
        a10536 = (0.70710678118654757*(t7332 - t7337));
        s4156 = (a10535 - a10536);
        s4157 = (a10536 + a10535);
        s4158 = (t7322 + t7338);
        s4159 = (t7323 + t7339);
        s4160 = (t7322 - t7338);
        s4161 = (t7323 - t7339);
        s4162 = (t7326 + s4154);
        s4163 = (t7327 + s4155);
        s4164 = (t7326 - s4154);
        s4165 = (t7327 - s4155);
        s4166 = (t7324 + t7341);
        s4167 = (t7325 - t7340);
        s4168 = (t7324 - t7341);
        s4169 = (t7325 + t7340);
        s4170 = (t7328 + s4156);
        s4171 = (t7329 - s4157);
        s4172 = (t7328 - s4156);
        s4173 = (t7329 + s4157);
        t7342 = (s4058 + s4090);
        t7343 = (s4059 + s4091);
        t7344 = (s4058 - s4090);
        t7345 = (s4059 - s4091);
        t7346 = (s4074 + s4106);
        t7347 = (s4075 + s4107);
        t7348 = (s4074 - s4106);
        t7349 = (s4075 - s4107);
        t7350 = (t7342 + t7346);
        t7351 = (t7343 + t7347);
        t7352 = (t7342 - t7346);
        t7353 = (t7343 - t7347);
        t7354 = (t7344 + t7349);
        t7355 = (t7345 - t7348);
        t7356 = (t7344 - t7349);
        t7357 = (t7345 + t7348);
        t7358 = (s4066 + s4098);
        t7359 = (s4067 + s4099);
        t7360 = (s4066 - s4098);
        t7361 = (s4067 - s4099);
        t7362 = (s4082 + s4114);
        t7363 = (s4083 + s4115);
        t7364 = (s4082 - s4114);
        t7365 = (s4083 - s4115);
        t7366 = (t7358 + t7362);
        t7367 = (t7359 + t7363);
        t7368 = (t7358 - t7362);
        t7369 = (t7359 - t7363);
        a10537 = (0.70710678118654757*(t7360 + t7365));
        a10538 = (0.70710678118654757*(t7361 - t7364));
        s4174 = (a10537 + a10538);
        s4175 = (a10538 - a10537);
        a10539 = (0.70710678118654757*(t7361 + t7364));
        a10540 = (0.70710678118654757*(t7360 - t7365));
        s4176 = (a10539 - a10540);
        s4177 = (a10540 + a10539);
        s4178 = (t7350 + t7366);
        s4179 = (t7351 + t7367);
        s4180 = (t7350 - t7366);
        s4181 = (t7351 - t7367);
        s4182 = (t7354 + s4174);
        s4183 = (t7355 + s4175);
        s4184 = (t7354 - s4174);
        s4185 = (t7355 - s4175);
        s4186 = (t7352 + t7369);
        s4187 = (t7353 - t7368);
        s4188 = (t7352 - t7369);
        s4189 = (t7353 + t7368);
        s4190 = (t7356 + s4176);
        s4191 = (t7357 - s4177);
        s4192 = (t7356 - s4176);
        s4193 = (t7357 + s4177);
        a10541 = (62*j1);
        a10542 = TW1[a10541];
        a10543 = TW1[(a10541 + 1)];
        a10544 = TW1[(a10541 + 2)];
        a10545 = TW1[(a10541 + 3)];
        a10546 = TW1[(a10541 + 4)];
        a10547 = TW1[(a10541 + 5)];
        a10548 = TW1[(a10541 + 6)];
        a10549 = TW1[(a10541 + 7)];
        a10550 = TW1[(a10541 + 8)];
        a10551 = TW1[(a10541 + 9)];
        a10552 = TW1[(a10541 + 10)];
        a10553 = TW1[(a10541 + 11)];
        a10554 = TW1[(a10541 + 12)];
        a10555 = TW1[(a10541 + 13)];
        a10556 = TW1[(a10541 + 14)];
        a10557 = TW1[(a10541 + 15)];
        a10558 = TW1[(a10541 + 16)];
        a10559 = TW1[(a10541 + 17)];
        a10560 = TW1[(a10541 + 18)];
        a10561 = TW1[(a10541 + 19)];
        a10562 = TW1[(a10541 + 20)];
        a10563 = TW1[(a10541 + 21)];
        a10564 = TW1[(a10541 + 22)];
        a10565 = TW1[(a10541 + 23)];
        a10566 = TW1[(a10541 + 24)];
        a10567 = TW1[(a10541 + 25)];
        a10568 = TW1[(a10541 + 26)];
        a10569 = TW1[(a10541 + 27)];
        a10570 = TW1[(a10541 + 28)];
        a10571 = TW1[(a10541 + 29)];
        a10572 = TW1[(a10541 + 30)];
        a10573 = TW1[(a10541 + 31)];
        a10574 = TW1[(a10541 + 32)];
        a10575 = TW1[(a10541 + 33)];
        a10576 = TW1[(a10541 + 34)];
        a10577 = TW1[(a10541 + 35)];
        a10578 = TW1[(a10541 + 36)];
        a10579 = TW1[(a10541 + 37)];
        a10580 = TW1[(a10541 + 38)];
        a10581 = TW1[(a10541 + 39)];
        a10582 = TW1[(a10541 + 40)];
        a10583 = TW1[(a10541 + 41)];
        a10584 = TW1[(a10541 + 42)];
        a10585 = TW1[(a10541 + 43)];
        a10586 = TW1[(a10541 + 44)];
        a10587 = TW1[(a10541 + 45)];
        a10588 = TW1[(a10541 + 46)];
        a10589 = TW1[(a10541 + 47)];
        a10590 = TW1[(a10541 + 48)];
        a10591 = TW1[(a10541 + 49)];
        a10592 = TW1[(a10541 + 50)];
        a10593 = TW1[(a10541 + 51)];
        a10594 = TW1[(a10541 + 52)];
        a10595 = TW1[(a10541 + 53)];
        a10596 = TW1[(a10541 + 54)];
        a10597 = TW1[(a10541 + 55)];
        a10598 = TW1[(a10541 + 56)];
        a10599 = TW1[(a10541 + 57)];
        a10600 = TW1[(a10541 + 58)];
        a10601 = TW1[(a10541 + 59)];
        a10602 = TW1[(a10541 + 60)];
        a10603 = TW1[(a10541 + 61)];
        a10604 = (64*j1);
        Y[a10604] = (t7266 + t7282);
        Y[(a10604 + 1)] = (t7267 + t7283);
        Y[(a10604 + 2)] = ((a10542*s4138) - (a10543*s4139));
        Y[(a10604 + 3)] = ((a10543*s4138) + (a10542*s4139));
        Y[(a10604 + 4)] = ((a10544*s4158) - (a10545*s4159));
        Y[(a10604 + 5)] = ((a10545*s4158) + (a10544*s4159));
        Y[(a10604 + 6)] = ((a10546*s4178) - (a10547*s4179));
        Y[(a10604 + 7)] = ((a10547*s4178) + (a10546*s4179));
        Y[(a10604 + 8)] = ((a10548*s4122) - (a10549*s4123));
        Y[(a10604 + 9)] = ((a10549*s4122) + (a10548*s4123));
        Y[(a10604 + 10)] = ((a10550*s4142) - (a10551*s4143));
        Y[(a10604 + 11)] = ((a10551*s4142) + (a10550*s4143));
        Y[(a10604 + 12)] = ((a10552*s4162) - (a10553*s4163));
        Y[(a10604 + 13)] = ((a10553*s4162) + (a10552*s4163));
        Y[(a10604 + 14)] = ((a10554*s4182) - (a10555*s4183));
        Y[(a10604 + 15)] = ((a10555*s4182) + (a10554*s4183));
        Y[(a10604 + 16)] = ((a10556*s4126) - (a10557*s4127));
        Y[(a10604 + 17)] = ((a10557*s4126) + (a10556*s4127));
        Y[(a10604 + 18)] = ((a10558*s4146) - (a10559*s4147));
        Y[(a10604 + 19)] = ((a10559*s4146) + (a10558*s4147));
        Y[(a10604 + 20)] = ((a10560*s4166) - (a10561*s4167));
        Y[(a10604 + 21)] = ((a10561*s4166) + (a10560*s4167));
        Y[(a10604 + 22)] = ((a10562*s4186) - (a10563*s4187));
        Y[(a10604 + 23)] = ((a10563*s4186) + (a10562*s4187));
        Y[(a10604 + 24)] = ((a10564*s4130) - (a10565*s4131));
        Y[(a10604 + 25)] = ((a10565*s4130) + (a10564*s4131));
        Y[(a10604 + 26)] = ((a10566*s4150) - (a10567*s4151));
        Y[(a10604 + 27)] = ((a10567*s4150) + (a10566*s4151));
        Y[(a10604 + 28)] = ((a10568*s4170) - (a10569*s4171));
        Y[(a10604 + 29)] = ((a10569*s4170) + (a10568*s4171));
        Y[(a10604 + 30)] = ((a10570*s4190) - (a10571*s4191));
        Y[(a10604 + 31)] = ((a10571*s4190) + (a10570*s4191));
        Y[(a10604 + 32)] = ((a10572*s4120) - (a10573*s4121));
        Y[(a10604 + 33)] = ((a10573*s4120) + (a10572*s4121));
        Y[(a10604 + 34)] = ((a10574*s4140) - (a10575*s4141));
        Y[(a10604 + 35)] = ((a10575*s4140) + (a10574*s4141));
        Y[(a10604 + 36)] = ((a10576*s4160) - (a10577*s4161));
        Y[(a10604 + 37)] = ((a10577*s4160) + (a10576*s4161));
        Y[(a10604 + 38)] = ((a10578*s4180) - (a10579*s4181));
        Y[(a10604 + 39)] = ((a10579*s4180) + (a10578*s4181));
        Y[(a10604 + 40)] = ((a10580*s4124) - (a10581*s4125));
        Y[(a10604 + 41)] = ((a10581*s4124) + (a10580*s4125));
        Y[(a10604 + 42)] = ((a10582*s4144) - (a10583*s4145));
        Y[(a10604 + 43)] = ((a10583*s4144) + (a10582*s4145));
        Y[(a10604 + 44)] = ((a10584*s4164) - (a10585*s4165));
        Y[(a10604 + 45)] = ((a10585*s4164) + (a10584*s4165));
        Y[(a10604 + 46)] = ((a10586*s4184) - (a10587*s4185));
        Y[(a10604 + 47)] = ((a10587*s4184) + (a10586*s4185));
        Y[(a10604 + 48)] = ((a10588*s4128) - (a10589*s4129));
        Y[(a10604 + 49)] = ((a10589*s4128) + (a10588*s4129));
        Y[(a10604 + 50)] = ((a10590*s4148) - (a10591*s4149));
        Y[(a10604 + 51)] = ((a10591*s4148) + (a10590*s4149));
        Y[(a10604 + 52)] = ((a10592*s4168) - (a10593*s4169));
        Y[(a10604 + 53)] = ((a10593*s4168) + (a10592*s4169));
        Y[(a10604 + 54)] = ((a10594*s4188) - (a10595*s4189));
        Y[(a10604 + 55)] = ((a10595*s4188) + (a10594*s4189));
        Y[(a10604 + 56)] = ((a10596*s4132) - (a10597*s4133));
        Y[(a10604 + 57)] = ((a10597*s4132) + (a10596*s4133));
        Y[(a10604 + 58)] = ((a10598*s4152) - (a10599*s4153));
        Y[(a10604 + 59)] = ((a10599*s4152) + (a10598*s4153));
        Y[(a10604 + 60)] = ((a10600*s4172) - (a10601*s4173));
        Y[(a10604 + 61)] = ((a10601*s4172) + (a10600*s4173));
        Y[(a10604 + 62)] = ((a10602*s4192) - (a10603*s4193));
        Y[(a10604 + 63)] = ((a10603*s4192) + (a10602*s4193));
    }
}