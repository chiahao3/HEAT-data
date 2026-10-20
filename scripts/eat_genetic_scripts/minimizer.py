import torch


class minimizer:
    """
    Projected limited-memory BFGS optimizer.
    """

    def __init__(
        self,
        fun_jac,
        x0,
        args,
        project,
        proj_args,
        callback=None,
        BFGS_hist=50,
        maxit=100,
        tol=1e-10,
        max_line_search=3,
    ):
        self.fun_jac = fun_jac
        self.x0 = x0
        self.BFGS_hist = BFGS_hist
        self.args = args
        self.proj_args = proj_args
        self.callback = callback
        self.project = project
        self.maxit = maxit
        self.tol = tol
        self.max_line_search = max_line_search

    def PBFGS(self):
        device, dtype = self.x0.device, self.x0.dtype
        x = self.x0.clone()
        f_new = None
        g_new = None
        k = 1
        S_list = []
        R_list = []
        converged = False
        reset = True

        while k <= self.maxit:
            if reset:
                f, g = self.fun_jac(x, *self.args)
                err = torch.norm(self.project(x - g, *self.proj_args) - x).item()
                if isinstance(f, torch.Tensor):
                    f = f.item()

                alpha = 1e-4
                x = self.project(x - alpha * g, *self.proj_args)
                f_new, g_new = self.fun_jac(x, *self.args)

                s = -alpha * g
                y = g_new - g
                r = y.clone()

                delta = (y.dot(y) / y.dot(s)).item() if y.dot(s).item() > 0 else 1.0
                N = torch.zeros((len(x), 1), device=device, dtype=dtype)
                M_inv = torch.zeros((1, 1), device=device, dtype=dtype)
                reset = False
            else:
                f = f_new
                g = g_new

                err = torch.norm(self.project(x - g, *self.proj_args) - x).item()
                if err < self.tol:
                    converged = True
                    break

                x_star, SPG_converged = self._SPG(
                    x0=x,
                    f0=f,
                    g0=g,
                    delta=delta,
                    N=N,
                    M_inv=M_inv,
                    SPG_hist=3,
                )

                if not SPG_converged:
                    S_list = []
                    R_list = []
                    reset = True
                    continue

                x_star = self.project(x_star, *self.proj_args)
                d = x_star - x

                alpha, f_temp, g_temp = self._line_min(
                    fun_jac=self.fun_jac,
                    x0=x,
                    f0=f,
                    g0=g,
                    p=d,
                    f_b=f,
                    args=self.args,
                    amax=1,
                    c1=1e-4,
                    maxiter=self.max_line_search,
                )
                if (alpha is None) or (alpha < 0) or isinstance(alpha, complex):
                    reset = True
                    continue

                g_new = g_temp
                f_new = f_temp
                x = self.project(x + alpha * d, *self.proj_args)
                s = alpha * d
                y = g_new - g
                r = self._construct_r(delta, N, M_inv, s, y)
                delta, N, M_inv, S_list, R_list, reset = self._construct_B(s, r, S_list, R_list)

            if self.callback is not None:
                err_new = torch.norm(self.project(x - g_new, *self.proj_args) - x).item()
                self.callback(x, f_new, err_new, k, converged, *self.args)
            k += 1

        return f, x

    def _construct_B(self, s, y, S_list, Y_list):
        device = s.device
        dtype = s.dtype
        if len(S_list) < self.BFGS_hist:
            S_list.append(s.clone())
            Y_list.append(y.clone())
        else:
            S_list.pop(0)
            S_list.append(s.clone())
            Y_list.pop(0)
            Y_list.append(y.clone())

        S = torch.stack(S_list, dim=1)
        Y = torch.stack(Y_list, dim=1)
        gamma = torch.dot(s, y).item() / torch.dot(y, y).item()
        if abs(gamma) > 0:
            delta = 1.0 / gamma
        else:
            return None, None, None, None, None, True

        L = torch.zeros((S.shape[1], S.shape[1]), device=device, dtype=dtype)
        for i in range(S.shape[1]):
            for j in range(S.shape[1]):
                if i > j:
                    L[i, j] = torch.dot(S_list[i], Y_list[j]).item()

        D = torch.zeros((S.shape[1], S.shape[1]), device=device, dtype=dtype)
        for i in range(S.shape[1]):
            D[i, i] = torch.dot(S_list[i], Y_list[i]).item()

        N = torch.cat([delta * S, Y], dim=1)
        A = delta * S.t().matmul(S)
        top = torch.cat([A, L], dim=1)
        bottom = torch.cat([L.t(), -D], dim=1)
        M = torch.cat([top, bottom], dim=0)
        M_inv = torch.inverse(M)
        return delta, N, M_inv, S_list, Y_list, False

    def _construct_r(self, delta, N, M_inv, s, y):
        t = N.t() @ s
        u = M_inv @ t
        v = N @ u
        Bs = delta * s - v

        if s @ y >= 0.2 * s @ Bs:
            theta = 1
        else:
            theta = 0.8 * s @ Bs / (s @ Bs - s @ y)

        r = theta * y + (1 - theta) * Bs
        return r

    def _SPG(self, x0, f0, g0, delta, N, M_inv, SPG_hist, maxit=50, tol=1e-3):
        def quad(x):
            r = N.t().matmul(x - x0)
            r = M_inv.matmul(r)
            r = N.matmul(r)
            r = delta * (x - x0) - r
            return (0.5 * torch.dot(x - x0, r) + torch.dot(g0, x - x0)).item() + f0, r + g0

        a_max = 1e10
        a_min = 1e-10
        x = self.project(x0, *self.proj_args)
        f, g = quad(x)
        a_bb = 1.0
        f_hist = [f]

        for _ in range(maxit):
            a_k = min(a_max, max(a_min, a_bb))
            d_k = self.project(x - a_k * g, *self.proj_args) - x
            f_b = max(f_hist)
            a, f_new_val, g_new = self._line_min(
                fun_jac=quad,
                x0=x,
                f0=f,
                g0=g,
                p=d_k,
                f_b=f_b,
                args=(),
                amax=1,
                c1=1e-4,
                maxiter=10,
            )
            if (a is None) or (a < 0) or isinstance(a, complex):
                a = 1.0
                f_new_val, g_new = quad(x + a * d_k)

            if len(f_hist) < SPG_hist:
                f_hist.append(f_new_val)
            else:
                f_hist = f_hist[1:] + [f_new_val]

            x_new = x + a * d_k
            s = x_new - x
            y = g_new - g
            num = torch.dot(s, s).item()
            den = torch.dot(s, y).item()
            a_bb = num / den if den > 0 else 1.0

            err = torch.norm(self.project(x_new - g_new, *self.proj_args) - x_new).item()
            if err < tol:
                return x_new, True

            x, g = x_new, g_new

        return x, False

    def _line_min(self, fun_jac, x0, f0, g0, p, f_b, args, amax=1, c1=1e-4, maxiter=5):
        a_list = [0.0, float(amax)]
        f_list = [f0]
        g_list = [g0]
        i = 1
        while i <= maxiter:
            x_new = x0 + a_list[i] * p
            try:
                f_val, g_val = fun_jac(x_new, *args)
            except Exception:
                f_list.append(f_list[i - 1])
                g_list.append(g_list[i - 1])
                a_list.append(a_list[i] / 2.0)
                i += 1
                continue

            if isinstance(f_val, torch.Tensor):
                f_val = f_val.item()
            f_list.append(f_val)
            g_list.append(g_val)

            if f_list[i] <= f_b + c1 * a_list[i] * torch.dot(g_list[0], p).item():
                return a_list[i], f_list[i], g_list[i]

            a_new = self._cubic_interp(
                a_list[i],
                f_list[i],
                torch.dot(g_list[i], p).item(),
                a_list[0],
                f_list[0],
                torch.dot(g_list[0], p).item(),
            )
            if isinstance(a_new, complex) or a_new < 0:
                a_new = a_list[i] / 2.0
            a_list.append(a_new)
            i += 1

        return None, f_list[maxiter], g_list[maxiter]

    def _cubic_interp(self, an, fn, gn, ao, fo, go):
        try:
            d1 = go + gn - 3.0 * (fo - fn) / (ao - an)
            d2 = (1.0 if (an - ao) >= 0 else -1.0) * ((d1 ** 2 - go * gn) ** 0.5)
            res = an - (an - ao) * ((gn + d2 - d1) / (gn - go + 2.0 * d2))
        except Exception:
            res = an
        return res
