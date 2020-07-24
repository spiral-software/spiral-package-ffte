# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

Declare(sve_gath, sve_scat, sve_ld2, sve_st2);

Class(TVectSVE, TReal);
TVectSVEx2 := TVect(TVectSVE, 2);

Class(TBoolSVE, TBool);
Class(TInt64SVE, TInt);

# SVE vector loop
Class(sve_loopn, loopn);

Class(sve_svcntd, Exp, rec(    
    computeType := self >> TInt
));


# svfloat64_t svld1_gather_[u64]offset[_f64](svbool_t pg, const float64_t *base, svuint64_t offsets) 

Class(sve_gath, Loc, rec(
    __call__ := (self, loc, n, pg, stride) >> WithBases(self,
        rec(operations := NthOps,
            loc := toExpArg(loc),
            n := toExpArg(n),
            pg := toExpArg(pg),
            stride := toExpArg(stride))).setType(),

    can_fold := False,

    rChildren := self >> [self.loc, self.n, self.pg, self.stride],
    rSetChild := rSetChildFields("loc", "n", "pg", "stride"),

    ev := self >> let(e := self.eval(), Cond(IsBound(e.v), e.v, e)),

    eval := self >> sve_gath(self.loc.eval(), self.n.eval(), self.pg.eval(), self.stride.eval()),

    computeType := self >> TVectSVE,

    isExpComposite := true
));


Class(sve_scat, Command, rec(
   __call__ := (self, loc, n, pg, stride, exp) >> WithBases(self,
       rec(operations := CmdOps,
       loc := toAssignTarget(loc),
       n := toExpArg(n),
       pg := toExpArg(pg),
       stride := toExpArg(stride),
       exp := toExpArg(exp)
       )),

   rChildren := self >> [self.loc, self.n, self.pg, self.stride, self.exp],
   rSetChild := rSetChildFields("loc", "n", "pg", "stride", "exp"),
   unroll := self >> self,

   print := (self,i,si) >> let(name := Cond(IsBound(self.isCompute) and self.isCompute,
                                            gap.colors.DarkYellow(self.__name__),
                                            IsBound(self.isLoad) and self.isLoad,
                                            gap.colors.DarkRed(self.__name__),
                                            IsBound(self.isStore) and self.isStore,
                                            gap.colors.DarkGreen(self.__name__),
                                            self.__name__),
                                Print(name, "(", self.loc, ", ", self.n, ", ", self.pg, ", ", self.stride, ", ", self.exp, ")"))
));


# svfloat64x2_t svld2[_f64](svbool_t pg, const float64_t *base)

Class(sve_ld2, Loc, rec(
    __call__ := (self, loc, n, pg) >> WithBases(self,
        rec(operations := NthOps,
            loc := toExpArg(loc),
            n := toExpArg(n),
            pg := toExpArg(pg))).setType(),

    can_fold := False,

    rChildren := self >> [self.loc, self.n, self.pg],
    rSetChild := rSetChildFields("loc", "n", "pg"),

    ev := self >> let(e := self.eval(), Cond(IsBound(e.v), e.v, e)),

    eval := self >> sve_ld2(self.loc.eval(), self.n.eval(), self.pg.eval()),

    computeType := self >> TVectSVEx2,

    isExpComposite := true
));


Class(sve_st2, Command, rec(
   __call__ := (self, loc, n, pg, exp) >> WithBases(self,
       rec(operations := CmdOps,
       loc := toAssignTarget(loc),
       n := toExpArg(n),
       pg := toExpArg(pg),
       exp := toExpArg(exp)
       )),

   rChildren := self >> [self.loc, self.n, self.pg, self.exp],
   rSetChild := rSetChildFields("loc", "n", "pg", "exp"),
   unroll := self >> self,

   print := (self,i,si) >> let(name := Cond(IsBound(self.isCompute) and self.isCompute,
                                            gap.colors.DarkYellow(self.__name__),
                                            IsBound(self.isLoad) and self.isLoad,
                                            gap.colors.DarkRed(self.__name__),
                                            IsBound(self.isStore) and self.isStore,
                                            gap.colors.DarkGreen(self.__name__),
                                            self.__name__),
                                Print(name, "(", self.loc, ", ", self.n, ", ", self.pg, ", ", self.exp, ")"))
));

