# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

# Radix N kernel for FFTE

buildScalarKernel_b := function(n, kk, conf, opts)
    local name, suffix, filesuffix, j, k, l, m, tmp1, tmp2, tmp3, i1, i2, gf, sf, gath, scat,
        gc, sc, dft, rts, rt, opcnts, dfts, dftc, i3, twt, twf, twl, tws, twc,
        cl, cc, c, cx, lp, mp, jj;

    name := DFTnameStr(n, kk);
    suffix := "b";
    filesuffix := "c_";
#    suffix := "_jk";
    Print("\n\n// == Generating \"", name, "\" ==================================================\n\n");    

    # variables
    #t := var.fresh_t("t", TInt);
    j := var.fresh_t("j", TInt);
    k := var.fresh_t("k", TInt);
    l := var.fresh_t("l", TInt);
    m := var.fresh_t("m", TInt);
    lp := var.fresh_t("lp", TPtr(TInt));
    mp := var.fresh_t("mp", TPtr(TInt));

    # temp arrays
    tmp1 := var.fresh_t("R", TArray(TReal, 2*n));
    tmp2 := var.fresh_t("S", TArray(TReal, 2*n));
    tmp3 := var.fresh_t("T", TArray(TReal, 2*n));
    
    # gather and scatter
    i1 := var.fresh_t("i", n);
    i2 := var.fresh_t("i", n);
    gf := fTensor(Lambda(i1, k + j*m + i1*l*m), fId(2));
    sf := fTensor(Lambda(i2, k+n*j*m + i2*m), fId(2));
    gath := Gath(gf);
    scat := Scat(sf);

    # generate gather and scatter code
    gc := BlockUnroll(unroll_cmd(opts.codegen(gath, tmp1, X, opts)), opts);
    sc := BlockUnroll(unroll_cmd(opts.codegen(scat, Y, tmp3, opts)), opts);
    
    # generate DFT kernel
    dft := RC(DFT(n, kk));
    rts := AllRuleTrees(dft, opts);
    opcnts := List(rts, r -> [ Length(Collect(CodeRuleTree(r, opts), @(1, [add, sub, mul], e->e.t=TReal))), r]);
    rt := Minimum(opcnts)[2];
    dfts := SumsRuleTree(rt, opts);
    dftc := BlockUnroll(opts.codegen(dfts, tmp2, tmp1, opts), opts);
    
    # twiddles as lookup table
    i3 := var.fresh_t("i", 2*n);
    twt := var.fresh_t("TW", TPtr(TReal)); 
    twf := Lambda(i3, cond(eq(i3, 0), 1, eq(i3, 1), 0, nth(twt, 2*((n-1)*j) + 2 * idiv(i3, 2) + (imod(i3, 2) - 2))));
    
    # debug twiddles
    twl := Map(twf.tolist(), RulesStrengthReduce);
    DoForAll([0..Length(twl)-1], _i->Unparse(assign(nth(Y, _i), twl[_i+1]), CUnparser, 0, 1));

    tws := RCDiag(twf);
    twc :=BlockUnroll(unroll_cmd(opts.codegen(tws, tmp3, tmp2, opts)), opts);
    
    # stitch code fragments together
    cl := func(TVoid, "transform", [Y, X, twt, j, k, l, m], 
            decl([tmp1, tmp2, tmp3], chain(gc, dftc, twc, sc)));
    cc := Compile(cl, opts);
    
    # fixup to push decls inside (?)
    c := func(TVoid, "transform", [Y, X, twt, j, k, lp, mp], 
            decl(cc.vars::[l, m], 
                chain(
                    assign(l, deref(lp)),
                    assign(m, deref(mp)),
                    cc.cmd.cmds[1].cmd
                )
            )
         );
    
    # print Radix N FFTE kernel as function
#    PrintCode(name, c, opts);
#    PrintTo(name::".c", PrintCode(name, c, opts));
    
    # function with j and k loop in the function call
    jj := var.fresh_t("j", TInt);
    cl := func(TVoid, "transform", [Y, X, twt, lp, mp], 
        decl(c.cmd.vars::c.free()::[j], 
            chain(
                assign(l, deref(lp)),
                assign(m, deref(mp)),
                loopn(jj, l-1, chain(
                    assign(j, add(jj, V(1))), 
                    loopn(k, m, c.cmd.cmd.cmds[3]))
                )
            )
        )
    );
    
    # print Radix N FFTE kernel as function
    PrintCode(opts.FortranIze(name::suffix), cl, opts);
    PrintTo(name::filesuffix::suffix::".c", PrintCode(opts.FortranIze(name::suffix), cl, opts));
end;


