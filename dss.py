"""A DSS system."""

import numpy as np


class Data:
    """Decisions and weights."""

    xp: np.ndarray
    inc: np.ndarray
    wgt: np.ndarray
    kk: np.ndarray

    def __init__(
        self, xp: np.ndarray, inc: np.ndarray, wgt: np.ndarray, kk: np.ndarray
    ) -> None:
        assert (
            len(xp.shape) == 2
            and len(inc.shape) == 1
            and len(wgt.shape) == 1
            and len(kk.shape) == 1
        )
        p = xp.shape[1]
        assert inc.shape[0] == p and wgt.shape[0] == p and kk.shape[0] == 4
        self.xp = xp
        self.inc = inc
        self.wgt = wgt
        self.kk = kk

    def format(self) -> None:
        """Format weights of decisions."""
        xp = self.xp
        n = xp.shape[0]
        p = xp.shape[1]

        s = ["j"]
        s.extend(["%d" % (i + 1) for i in range(p)])
        print("VAR " + ",".join(s))
        for j in range(n):
            s = ["%d" % j]
            s.extend(["%.2f" % pv for pv in xp[j, :]])
            print(",".join(s))

        s = ["%d" % (i + 1) for i in range(p)]
        print("INC " + ",".join(s))
        s = ["1" if iv else "0" for iv in self.inc]
        print(",".join(s))

        s = ["%d" % (i + 1) for i in range(p)]
        print("WGT " + ",".join(s))
        s = ["%.2f" % wv for wv in self.wgt]
        print(",".join(s))

        s = ["%d" % (i + 1) for i in range(4)]
        print("KK " + ",".join(s))
        s = ["%.2f" % kv for kv in self.kk]
        print(",".join(s))


def get_bmatrix(w: np.ndarray, increase: bool) -> np.ndarray:
    """Create binary matrix for weights of a specific decision.

    BR(x,y) = 1 if x >= y
            = 0 otherwise
    """
    assert len(w.shape) == 1
    n = w.shape[0]
    bm = np.zeros((n, n), dtype=int)
    for i in range(n):
        mask = w[i] >= w if increase else w[i] <= w
        bm[i, mask] = 1
        bm[i, i] = 0
    return bm


def get_rates(scores: np.ndarray) -> np.ndarray:
    """Get rates of decisions."""
    assert len(scores.shape) == 1
    iv = [(i, v) for i, v in enumerate(scores)]
    iv.sort(key=lambda it: -it[1])
    n = scores.shape[0]
    rt = np.zeros(n, dtype=int)
    p = 0
    for i in range(1, n):
        if iv[i][1] < iv[i-1][1]:
            p += 1
        rt[iv[i][0]] = p
    return rt + 1


class Solver:
    ways: list[str]
    scores: list[np.ndarray]

    def __init__(self) -> None:
        self.ways = []
        self.scores = []

    def format(self) -> None:
        rates = [get_rates(sc) for sc in self.scores]
        n = self.scores[0].shape[0] if self.scores else 0
        title = ["j"]
        for w in self.ways:
            title.extend([w + "-score", w + "-rate"])
        print(",".join(title))
        for j in range(n):
            row = ["%d" % j]
            for sc, sr in zip(self.scores, rates):
                row.extend(["%.2f" % sc[j], "%d" % sr[j]])
            print(",".join(row))

    def solve(self, xp: np.ndarray, inc: np.ndarray, wgt: np.ndarray) -> None:
        """Find scores of decisions."""
        raise AssertionError("ENOTSUP")


class Kmax(Solver):
    """K-maximum solver."""

    kk: np.ndarray

    def __init__(self, kk: np.ndarray) -> None:
        super().__init__()
        self.kk = kk

    def get_smatrix(self, bm: np.ndarray) -> np.ndarray:
        """Get K-maximum vectors."""
        assert len(bm.shape) == 2 and bm.shape[0] == bm.shape[1]
        n = bm.shape[0]
        sm = np.zeros((n, 4), dtype=int)
        mk = np.ones(n, dtype=bool)
        for i in range(n):
            mk[i] = False
            HR0 = (bm[i, mk] == 1) * (bm[mk, i] == 0)
            ER = (bm[i, mk] == 1) * (bm[mk, i] == 1)
            NR = (bm[i, mk] == 0) * (bm[mk, i] == 0)
            sm[i, :] = np.array(
                [
                    np.sum(HR0 + ER + NR),
                    np.sum(HR0 + NR),
                    np.sum(HR0 + ER),
                    np.sum(HR0),
                ],
                dtype=int,
            )
            mk[i] = True
        return sm

    def get_mmask(self, sm: np.ndarray, oi=-1) -> np.ndarray:
        """Caclulate a mask to select k-maximum variants.

        Let consider that the i-th variant is "optimal" under an SRk profile
        if it SRk(i) == max SRk(j). It seams to be more reasonable that to
        require that SRk(i) should be equal (n - 1), i.e. to requre it to be
        the 0-maximal for SR1 and SR3 and to be 0-maximal and comparable with
        all others for SR2 and SR4.

        For example, for SR1 profile, let x1 is defeated by x2 but is at least
        as good or incomparable with other n - 2 variants, SR1(x1) = n - 2. And
        x2 defeats x1, but it is defeated by x3 and is incomparable with others,
        SR1(x2) = n - 2. If so, consider that both x1 and x2 are "optimal",
        in the sense that they are 1-maximal and there are no one 0-maximal one.
        """
        assert len(sm.shape) == 2 and sm.shape[1] == 4
        assert oi >= -1 and oi < 3
        smx = np.max(sm, axis=0)
        if oi == -1:
            mm = np.array(sm >= smx, dtype=int)
        else:
            mm = np.zeros(sm.shape, dtype=int)
            mm[:,oi] = np.array(sm[:,oi] >= smx[oi], dtype=int)
        return mm

    def format_kmax_p(self, p: int, sm: np.ndarray, sjp: np.ndarray, sjm: np.ndarray):
        n = sm.shape[0]
        for j in range(n):
            s = ["%d" % p, "%d" % j]
            s.extend(["%.0f" % v for v in sm[j]])
            s.append("%.2f" % sjp[j])
            s.append("%.2f" % sjm[j])
            print(",".join(s))

    def solve(self, xp: np.ndarray, inc: np.ndarray, wgt: np.ndarray) -> None:
        """Find KJ and KM socres.
        """
        kk = self.kk
        n = xp.shape[0]
        p = xp.shape[1]
        tjp = np.zeros(n, dtype=float)
        tjm = np.zeros(n, dtype=float)
        # -1 all optimals, 0..3 smax,max,slar,lar only
        op = -1
        for pi in range(p):
            bm = get_bmatrix(xp[:, pi], inc[pi])
            sm = self.get_smatrix(bm)
            sj = np.dot(sm, kk) * wgt[pi]
            mm = self.get_mmask(sm, op)
            sjm = np.dot(sm * mm, kk) * wgt[pi]
            # self.format_kmax_p(pi, sm, sj, sjm)
            tjp += sj
            tjm += sjm
        self.ways = ["KJ", "KM"]
        self.scores = [tjp, tjm]


class Dominate(Solver):
    "Dominate solver."

    def __init__(self) -> None:
        super().__init__()

    def get_score(self, matrix: np.ndarray) -> np.ndarray:
        """Get dominating scores of decisions.

        CR(x) = {x: (Ay)xRy}
        """
        assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
        n = matrix.shape[0]
        scores = np.zeros(n)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            mask[i] = False
            scores[i] += 1 if np.all(matrix[i, mask] == 1) else 0
            mask[i] = True
        return scores

    def solve(self, xp: np.ndarray, inc: np.ndarray, wgt: np.ndarray) -> None:
        """Find dominating scores."""
        n = xp.shape[0]
        p = xp.shape[1]
        sc = np.zeros(n, dtype=float)
        for pi in range(p):
            bm = get_bmatrix(xp[:, pi], inc[pi])
            sp = self.get_score(bm)
            sc += sp * wgt[pi]
        self.ways = ["DOM"]
        self.scores = [sc]


class Block(Solver):
    """Block solver."""

    def __init__(self) -> None:
        super().__init__()

    def get_score(self, matrix: np.ndarray) -> np.ndarray:
        """Get block scores of decisions.

        CR(x) = |x: (Ay)y(!R)x|
        """
        assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
        n = matrix.shape[0]
        scores = np.zeros(n)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            mask[i] = False
            scores[i] += 1 if np.all(matrix[mask, i] == 0) else 0
            mask[i] = True
        return scores

    def solve(self, xp: np.ndarray, inc: np.ndarray, wgt: np.ndarray) -> None:
        """Find block scores."""
        n = xp.shape[0]
        p = xp.shape[1]
        sc = np.zeros(n, dtype=float)
        for pi in range(p):
            bm = get_bmatrix(xp[:, pi], inc[pi])
            sp = self.get_score(bm)
            sc += sp * wgt[pi]
        self.ways = ["BLK"]
        self.scores = [sc]


class Tourney(Solver):
    """Tournament solver."""

    def __init__(self) -> None:
        super().__init__()

    def get_score(self, matrix: np.ndarray) -> np.ndarray:
        """Get scores of tornament of variants.

        fR(x) = sum fR(x,y)
        fR(x,y) = 1, xRy and y(!R)x
                = 0, yRx and x(!R)y
                = 0.5, otherwise
        """
        assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
        n = matrix.shape[0]
        scores = np.zeros(n)
        for i in range(n):
            s1 = np.sum(matrix[i, :] > matrix[:, i])
            s2 = np.sum(matrix[:, i] > matrix[i, :])
            scores[i] = s1 + (n - s1 - s2 - 1) / 2
        return scores

    def solve(self, xp: np.ndarray, inc: np.ndarray, wgt: np.ndarray) -> None:
        """Find tournament scores."""
        n = xp.shape[0]
        p = xp.shape[1]
        sc = np.zeros(n, dtype=float)
        for pi in range(p):
            bm = get_bmatrix(xp[:, pi], inc[pi])
            sp = self.get_score(bm)
            sc += sp * wgt[pi]
        self.ways = ["TRN"]
        self.scores = [sc]


class DSS:
    solvers: list[Solver]
    data: Data

    def __init__(self, data: Data) -> None:
        self.solvers = [
            Kmax(data.kk),
            Dominate(),
            Block(),
            Tourney(),
        ]
        self.data = data

    def solve(self) -> None:
        for slv in self.solvers:
            slv.solve(self.data.xp, self.data.inc, self.data.wgt)

    def format(self) -> None:
        ways: list[str] = []
        scores: list[np.ndarray] = []
        for slv in self.solvers:
            ways.extend(slv.ways)
            scores.extend(slv.scores)
        rates = [get_rates(sc) for sc in scores]
        n = scores[0].shape[0] if scores else 0
        p = len(scores)

        gr = np.empty((n, p), dtype=int)
        gr[:] = n + 1
        for i, r in enumerate(rates):
            gr[:, i] -= r
        tg = np.sum(gr, axis=1)
        bj = np.argmax(tg)

        title = ["j"]
        title.extend([w for w in ways])
        title.extend(["total", "best"])
        print(",".join(title))
        for j in range(n):
            row = ["%d" % j]
            row.extend(["%.2f" % g for g in gr[j]])
            row.append("%.2f" % tg[j])
            row.append("*" if j == bj else "")
            print(",".join(row))

    @classmethod
    def deside(
        cls, xp: np.ndarray, inc: np.ndarray, wgt: np.ndarray, kk: np.ndarray
    ) -> None:
        """Get variants and take a decision."""
        data = Data(xp, inc, wgt, kk)
        data.format()
        print()

        self = cls(data)
        self.solve()

        for slv in self.solvers:
            slv.format()
            print()

        self.format()
