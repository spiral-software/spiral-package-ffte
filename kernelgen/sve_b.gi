# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

# Radix n kernel for FFTE, targeting ARM SVE

buildSVEKernel_b := function(n, kk, conf, opts)
    local vlen, name, suffix, filesuffix, t, j, k, l, m, tmp1, tmp2, tmp3, i1, i2, gf, sf, gath, scat,
        gi, g, gc, si, ss, s, sc, dft, rts, rt, opcnts, dfts, dftc, i, i3, twt, twf, twl, tws, twc,
        cl, cc, c, useFMA, _FMA, lp, mp, jj;

    name := DFTnameStr(n, kk);
    suffix := "b";
    filesuffix := "sve_";
#    suffix := "_jk";
    Print("\n\n// == Generating \"", name, "\" ==================================================\n\n");    

    vlen := TInt;
    useFMA := When(IsBound(conf.useFMA), conf.useFMA, true);
    _FMA := When(useFMA, FMA, p -> p);

    # variables
    t := var.fresh_t("t", TInt);
    j := var.fresh_t("j", TInt);
    k := var.fresh_t("k", TInt);
    l := var.fresh_t("l", TInt);
    m := var.fresh_t("m", TInt);
    lp := var.fresh_t("lp", TPtr(TInt));
    mp := var.fresh_t("mp", TPtr(TInt));

    # temp arrays
    tmp1 := var.fresh_t("R", TArray(TVectSVE, 2*n));
    tmp2 := var.fresh_t("S", TArray(TVectSVE, 2*n));
    tmp3 := var.fresh_t("T", TArray(TVectSVE, 2*n));

    # gather
    gi := _i -> VGath_L(1, vlen, 2*(k + j*m + _i*l*m));
    g := RulesSums(SumsSPL(VStack(List([0..n-1], gi)), opts));
    gc := Compile(opts.codegen(g, tmp1, X, opts), opts);
    
    # scatter
    si := _i -> VScat_L(1, vlen, 2*(k+n*j*m + _i*m));
    ss := SumsSPL(HStack(List([0..n-1], si)), opts);
    s := RulesSums(SubstTopDown(ss, @(1, ScatAcc), e->Scat(@(1).val.func)));
    sc := Compile(opts.codegen(s, Y, tmp3, opts), opts);
    
    # generate DFT kernel
    dft := RC(DFT(n, kk));
    rts := AllRuleTrees(dft, opts);
    opcnts := List(rts, r -> [ Length(Collect(CodeRuleTree(r, opts), @(1, [add, sub, mul], e->e.t=TReal))), r]);
    rt := Minimum(opcnts)[2];
    dfts := SumsRuleTree(rt, opts);
    dftc := BlockUnroll(opts.codegen(dfts, tmp2, tmp1, opts), opts);
    
    # twiddles as lookup table
    i := var.fresh_t("i", 2*n);
    twt := var.fresh_t("TW", TPtr(TReal)); 
    twf := Lambda(i, cond(eq(i, 0), 1, eq(i, 1), 0, nth(twt, 2*((n-1)*j) + 2 * idiv(i, 2) + (imod(i, 2) - 2))));
    
    # debug twiddles
    twl := Map(twf.tolist(), RulesStrengthReduce);
    DoForAll([0..Length(twl)-1], _i->Unparse(assign(nth(Y, _i), twl[_i+1]), CUnparser, 0, 1));
    tws := RCDiag(twf);
    twc :=BlockUnroll(unroll_cmd(opts.codegen(tws, tmp3, tmp2, opts)), opts);
    
    # stitch code fragments together
    cl := func(TVoid, "transform", [Y, X, twt, j, k, l, m, opts.pg], 
            decl([tmp1, tmp2, tmp3], chain(gc, dftc, twc, sc)));
    cc := Compile(cl, opts);
    
    # fixup to push decls inside (?)
    c := func(TVoid, "transform", [Y, X, twt, j, k, l, m, opts.pg], 
            decl(cc.vars, _FMA(cc.cmd.cmds[1].cmd)));
    
    # print Radix N FFTE kernel as function
    #PrintCode(name, c, opts);
    #PrintTo(name::".c", PrintCode(name, c, opts));
    
    ########################################################
    # version with j and k loop in the function call
    jj := var.fresh_t("j", TInt);
    cl := func(TVoid, "transform", [Y, X, twt, lp, mp], 
        decl(c.cmd.vars::c.free()::[opts.pg, l, m, j], 
            chain(
                assign(l, deref(lp)),
                assign(m, deref(mp)),
                loopn(jj, l-1, chain(
                    assign(j, add(jj, V(1))),  
                    sve_loopn(k, m, c.cmd.cmd)))
            )
        )
    );
    
    # print Radix N FFTE kernel as function
    PrintCode(opts.FortranIze(name::suffix), cl, opts);
    PrintTo(name::filesuffix::suffix::".c", PrintCode(opts.FortranIze(name::suffix), cl, opts));
end;





