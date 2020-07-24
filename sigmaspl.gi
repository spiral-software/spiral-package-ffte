# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

Declare(ScatH, VScat_L);


Class(GathH, SumsBase, BaseMat, rec(
    rChildren := self >> [self.N, self.n, self.base, self.stride],
    rSetChild := rSetChildFields("N", "n", "base", "stride"),
    new := (self, N, n, base, stride) >> 
	SPL(WithBases(self, rec(
      	N:= N, n := n, base := base, stride := stride, dimensions := [n, N]))),
    dims := self >> self.dimensions,
    sums := self >> self,
    area := self >> self.n,
    isReal := self >> true,
    transpose := self >> ScatH(self.N, self.n, self.base, self.stride),
    conjTranspose := self >> self.transpose(),
    inverse := self >> self.transpose(),
    toAMat := self >> let(
	   n := self.n,
       N := self.N,
       base := self.base,
       stride := self.stride,
       AMatMat(List([0..n-1], row -> let(
         idx := base+row,
		 BasisVec(N, idx))))
    ),
    isIdentity := self >> (self.n=self.N) and self.base=0 and self.stride = 1,
));

Class(ScatH, SumsBase, BaseMat, rec(
    rChildren := self >> [self.N, self.n, self.base, self.stride],
    rSetChild := rSetChildFields("N", "n", "base", "stride"),
    new := (self, N, n, base, stride) >> 
SPL(WithBases(self, rec(
      	N:= N, n := n, base := base, stride := stride, dimensions := [N, n]))),
    dims := self >> self.dimensions,
    sums := self >> self,
    area := self >> self.n,
    isReal := self >> true,
    transpose := self >> GathH(self.N, self.n, self.base, self.stride),
    conjTranspose := self >> self.transpose(),
    inverse := self >> self.transpose(),
    toAMat := self >> let(
	   n := self.n,
       N := self.N,
       base := self.base,
       stride := self.stride,
       TransposedAMat(AMatMat(List([0..n-1], row -> let(
         idx := base+row*stride,
		 BasisVec(N, idx))))
    )),
    isIdentity := self >> (self.n=self.N) and self.base=0 and self.stride = 1,
));



Class(VGath_L, SumsBase, BaseMat, rec(
    rChildren := self >> [self.N, self.n, self.base],
    rSetChild := rSetChildFields("N", "n", "base"),
    new := (self, N, n, base) >> 
	SPL(WithBases(self, rec(
      	N:= N, n := n, base := base, dimensions := [2, N]))),
    dims := self >> self.dimensions,
    sums := self >> self,
    area := self >> self.n,
    isReal := self >> true,
    transpose := self >> VScat_L(self.N, self.n, self.base),
    conjTranspose := self >> self.transpose(),
    inverse := self >> self.transpose(),
    toAMat := self >> let(
	   n := self.n,
       N := self.N,
       base := self.base,
       AMatSPL(L(2*n, 2) * Tensor(SPLAMat(AMatMat(List([0..n-1], row -> let(
         idx := base+row,
		 BasisVec(N, idx))))), I(2))) 
    ),
    isIdentity := False
));

Class(VScat_L, SumsBase, BaseMat, rec(
    rChildren := self >> [self.N, self.n, self.base],
    rSetChild := rSetChildFields("N", "n", "base"),
    new := (self, N, n, base) >> 
SPL(WithBases(self, rec(
      	N:= N, n := n, base := base, dimensions := [N, 2]))),
    dims := self >> self.dimensions,
    sums := self >> self,
    area := self >> self.n,
    isReal := self >> true,
    transpose := self >> VGath_L(self.N, self.n, self.base),
    conjTranspose := self >> self.transpose(),
    inverse := self >> self.transpose(),
    toAMat := self >> TransposedAMat(self.transpose().toAMat()),
    isIdentity := False
));

