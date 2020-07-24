# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

@TVectSVE := @.cond(x->x.t=TVectSVE);
@ValueTVectSVE := @.cond(x->x.t=TVectSVE and ObjId(x)=Value);


Class(SVEUnparserMixin, rec(
    sve_loopn := (self, o, i, is) >> let(v := o.var.id,
        pg := self.opts.pg.id,
        lo := 0,
        hi := o.range,
        Print(Blanks(i), "int ", v, " = 0;\n",
            Blanks(i), pg, " = svwhilelt_b64(", v, ", ", hi, ");\n",
            Blanks(i), "do {\n",
            self(o.cmd, i + is, is),
            Blanks(i + is), v, " += svcntd();\n",
            Blanks(i + is), pg, " = svwhilelt_b64(", v, ", ", hi, ");\n",
            Blanks(i), "} while(svptest_any(svptrue_b64(), ", self.opts.pg.id, "));\n") 
    ),

# svfloat64_t svmulx[_n_f64]_z(svbool_t pg, svfloat64_t op1, float64_t op2)

# svcntd()
    sve_svcntd := (self, o, i, is) >> Print("svcntd()"),

# svfloat64_t
    TVectSVE := (self, t, vars, i, is) >> Print(When(IsBound(self.opts.TRealSVEtype), self.opts.TRealSVEtype, "svfloat64_t"), " ", self.infix(vars, ", ", i + is)),

# svbool_t
    TBoolSVE := (self, t, vars, i, is) >> Print("svbool_t ", self.infix(vars, ", ", i + is)),

# int64_t
    TInt64SVE := (self, t, vars, i, is) >> Print("int64t ", self.infix(vars, ", ", i + is)),

# svfloat64x2_t
    TVect := (self, t, vars, i, is) >> Print(When(IsBound(self.opts.TRealSVEtype), self.opts.TRealSVEtype, "svfloat64x2_t"), " ", self.infix(vars, ", ", i + is)),

    velem := (self, o, i, is) >> Print(self(o.loc, i, is), ".v", StringInt(o.idx.v)),
    
## svfloat64_t svld1_gather_[u64]offset[_f64](svbool_t pg, const float64_t *base, svuint64_t offsets)
    sve_gath := (self, o, i, is) >> Print("svld1_gather_u64offset_f64(", self(o.pg, i, is), ", ", self(o.loc, i, is), ", ", 
        "svindex_u64(0, (int64_t)(", self(o.stride, i, is), " * sizeof(float64_t)))", 
        ")"),

# svint64_t incx_vec = svindex_s64(0, incx);	
# svfloat64_t dx_vec = svld1_gather_index(pg, dx, incx_vec);
#    sve_gath := (self, o, i, is) >> Print("svld1_gather_index(", self(o.pg, i, is), ", ", self(o.loc, i, is), ", ", 
#        "svindex_s64(0, (int64_t)", self(o.stride, i, is), ")", 
#        ")"),

## void svst1_scatter_u64_offset_f64(svbool_t pg, svuint64_t bases, int64_t offset, svfloat64_t data)    
    sve_scat := (self, o, i, is) >> Print(Blanks(i), "svst1_scatter_u64offset_f64(", self(o.pg, i, is), ", ((float64_t  *)", self(o.loc, i, is), "), ", 
            "svindex_u64(0, (int64_t)(", self(o.stride, i, is), " * sizeof(float64_t)))", ", ", self(o.exp, i, is), ");\n"),
                                         
# svfloat64x2_t svld2[_f64](svbool_t pg, const float64_t *base)
    sve_ld2 := (self, o, i, is) >> Print("svld2_f64(", self(o.pg, i, is), ", ", self(o.loc, i, is), ")"),

# void svst2[_f64](svbool_t pg, float64_t *base, svfloat64x2_t data)
    sve_st2 := (self, o, i, is) >> Print(Blanks(i), "svst2_f64(", self(o.pg, i, is), ", ", self(o.loc, i, is), ", ", self(o.exp, i, is), ");\n"),
  
  
# Arithmetic
    mul := (self, o, i, is) >> let(n := Length(o.args), Cond(
        not o.t = TVectSVE,
            Print("(", self.pinfix(o.args, ")*("), ")"),
 	    n > 2 and n mod 2 <> 0,
            self(mul(o.args[1], ApplyFunc(mul, Drop(o.args, 1))), i, is),
        n > 2, 
            self(mul(ApplyFunc(mul, o.args{[1..n/2]}), ApplyFunc(mul, o.args{[n/2+1..n]})), i, is),
        CondPat(o,
# svfloat64_t svmul[_n_f64]_x(svbool_t pg, svfloat64_t op1, float64_t op2)
           [mul, @ValueTVectSVE, @TVectSVE],
               self.printf("svmul_n_f64_x($1, $2, $3)", [self.opts.pg, o.args[2], o.args[1]]),
           [mul, @TVectSVE, @ValueTVectSVE],
               self.printf("svmul_n_f64_x($1, $2, $3)", [self.opts.pg, o.args[1], o.args[2]]),
           [mul, @TReal, @TVectSVE],
               self.printf("svmul_n_f64_x($1, $2, $3)", [self.opts.pg, o.args[2], o.args[1]]),
           [mul, @TVectSVE, @TReal],
               self.printf("svmul_n_f64_x($1, $2, $3)", [self.opts.pg, o.args[1], o.args[2]]),
           [mul, @TInt, @TVectSVE],
               self(mul(_toReal(o.args[1]), o.args[2]), i, is),
           [mul, @TVectSVE, @TInt],
               self(mul(o.args[1], _toReal(o.args[2])), i, is),
# svfloat64_t svmul[_f64]_x(svbool_t pg, svfloat64_t op1, float64_t op2)
           [mul, @TVectSVE, @TVectSVE],
               self.printf("svmul_f64_x($1, $2, $3)", [self.opts.pg, o.args[1], o.args[2]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")))
    ),

# -- add --
    add := (self, o, i, is) >> let(n := Length(o.args), Cond(
        not o.t = TVectSVE,
            self.pinfix(o.args, " + "),
        n > 2 and n mod 2 <> 0,
            self(add(o.args[1], ApplyFunc(add, Drop(o.args, 1))), i, is),
        n > 2, 
            self(add(ApplyFunc(add, o.args{[1..n/2]}), ApplyFunc(add, o.args{[n/2+1..n]})), i, is),
        CondPat(o,
            [add, @TReal, @TVectSVE],
# svfloat64_t svadd[_n_f64]_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2)
                self.printf("svadd_n_f64_x($1, $2, $3)", [self.opts.pg, o.args[2], o.args[1]]),
            [add, @TVectSVE, @TReal],
                self.printf("svadd_n_f64_x($1, $2, $3)", [self.opts.pg, o.args[1], o.args[2]]),
            [add, @TInt, @TVectSVE],
                self(add(_toReal(o.args[1]), o.args[2]), i, is),
            [add, @TVectSVE, @TInt],
                self(add(o.args[1], _toReal(o.args[2])), i, is),
# svfloat64_t svadd[_f64]_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2)
            [add, @TVectSVE, @TVectSVE],
                self.printf("svadd_f64_x($1, $2, $3)", [self.opts.pg, o.args[1], o.args[2]]),
            Error("Don't know how to unparse <o>. Unrecognized type combination")))
    ),

    sub := (self, o, i, is) >> Cond(
        not o.t = TVectSVE,
            Inherited(o, i, is),
        CondPat(o,
            [sub, @TReal, @TVectSVE],
# svfloat64_t svsub[_n_f64]_x(svbool_t pg, svfloat64_t op1, float64_t op2)
                self.printf("svsubr_n_f64_x($1, $2, $3)", [self.opts.pg, o.args[2], o.args[1]]),
            [sub, @TVectSVE, @TReal],
                self.printf("svsub_n_f64_x($1, $2, $3)", [self.opts.pg, o.args[1], o.args[2]]),
            [sub, @TInt, @TVectSVE],
                self(add(_toReal(o.args[1]), o.args[2]), i, is),
            [sub, @TVectSVE, @TInt],
                self(add(o.args[1], _toReal(o.args[2])), i, is),
# svfloat64_t svadd[_f64]_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2)
            [sub, @TVectSVE, @TVectSVE],
                self.printf("svsub_f64_x($1, $2, $3)", [self.opts.pg, o.args[1], o.args[2]]),
            Error("Don't know how to unparse <o>. Unrecognized type combination"))
    ),

#    neg := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), 
#        self(mul(neg(o.t.one()), o.args[1]), i, is),
#        Inherited(o, i, is)),
#
                                         
    fma := (self, o, i, is) >> When(
        Length(o.args) <> 3, Error("fma is strictly ternary"),
        CondPat(o,
# svfloat64_t svmla[_n_f64]_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2, float64_t op3)
           [fma, @TVectSVE, @TVectSVE, @ValueTVectSVE],
                self.printf("svmla_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[2], o.args[3]]),
           [fma, @TVectSVE, @ValueTVectSVE, @TVectSVE],
                self.printf("svmla_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[3], o.args[2]]),
           [fma, @TVectSVE, @TVectSVE, @TReal],
                self.printf("svmla_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[2], o.args[3]]),
           [fma, @TVectSVE, @TReal, @TVectSVE],
                self.printf("svmla_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[3], o.args[2]]),
# svfloat64_t svmla[_f64]_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
           [fma, @TVectSVE, @TVectSVE, @TVectSVE],
                self.printf("svmla_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[2], o.args[3]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
        )),
                                         
    nfma := (self, o, i, is) >> When(
        Length(o.args) <> 3, Error("nfma is strictly ternary"),
        CondPat(o,
# svfloat64_t svnmls[_n_f64]_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2, float64_t op3)
           [nfma, @TVectSVE, @TVectSVE, @ValueTVectSVE],
                self.printf("svnmls_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[2], o.args[3]]),
           [nfma, @TVectSVE, @ValueTVectSVE, @TVectSVE],
                self.printf("svnmls_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[3], o.args[2]]),
           [nfma, @TVectSVE, @TVectSVE, @TReal],
                self.printf("svnmls_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[2], o.args[3]]),
           [nfma, @TVectSVE, @TReal, @TVectSVE],
                self.printf("svnmls_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[3], o.args[2]]),
# svfloat64_t svnmls[_f64]_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
           [nfma, @TVectSVE, @TVectSVE, @TVectSVE],
                self.printf("svnmls_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[2], o.args[3]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
        )),
                                         
    fms := (self, o, i, is) >> When(
        Length(o.args) <> 3, Error("fms is strictly ternary"),
        CondPat(o,
# svfloat64_t svmls[_n_f64]_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2, float64_t op3)
           [fms, @TVectSVE, @TVectSVE, @ValueTVectSVE],
                self.printf("svmls_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[2], o.args[3]]),
           [fms, @TVectSVE, @ValueTVectSVE, @TVectSVE],
                self.printf("svmls_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[3], o.args[2]]),
           [fms, @TVectSVE, @TVectSVE, @TReal],
                self.printf("svmls_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[2], o.args[3]]),
           [fms, @TVectSVE, @TReal, @TVectSVE],
                self.printf("svmls_n_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[3], o.args[2]]),
# svfloat64_t svmls[_f64]_m(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
           [fms, @TVectSVE, @TVectSVE, @TVectSVE],
                self.printf("svmls_f64_x($1, $2, $3, $4)", [self.opts.pg, o.args[1], o.args[2], o.args[3]]),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
        )),
 
));

Class(SVEUnparser, SVEUnparserMixin, CUnparser);
