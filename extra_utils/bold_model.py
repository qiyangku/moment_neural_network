import numpy as np


class BOLD:
    def __init__(self, input_shape, epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02,
                 delta_t=1e-3, init_f_in=None, init_s=None, init_v=None, init_q=None):
        """
        bold = BOLD(epsilon=10, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02) by 曾龙斌
        bold = BOLD(epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02) by 张文勇
        """
        self.epsilon = epsilon
        self.tao_s = tao_s
        self.tao_f = tao_f
        self.tao_0 = tao_0
        self.E_0 = E_0
        self.V_0 = V_0
        self.delta_t = delta_t
        self.div_alpha = 1 / alpha
        self.f_in = init_f_in
        self.s = init_s
        self.v = init_v
        self.q = init_q
        self.input = None
        self.input_shape = input_shape
        self.stable_params()

    def update(self, f_str, df):
        f = self.__getattribute__(f_str)
        if f is None:
            self.__setattr__(f_str, df * self.delta_t)
        else:
            f += df * self.delta_t

    def run(self, u):
        assert isinstance(u, np.ndarray)
        if self.s is None:
            self.s = np.zeros_like(u)
        if self.q is None:
            self.q = np.zeros_like(u)
        if self.v is None:
            self.v = np.ones_like(u)
        if self.f_in is None:
            self.f_in = np.ones_like(u)
        d_s = self.epsilon * u - self.s/self.tao_s - (self.f_in-1)/self.tao_f
        q_part = np.where(self.f_in > 0, 1 - (1-self.E_0)**(1/self.f_in), np.ones_like(self.f_in))
        self.update('q', (self.f_in * q_part/self.E_0 - self.q * self.v ** (self.div_alpha - 1))/self.tao_0)
        self.update('v', (self.f_in - self.v ** self.div_alpha)/self.tao_0)
        self.update('f_in', self.s)
        self.f_in = np.where(self.f_in > 0, self.f_in, np.zeros_like(self.f_in))
        self.update('s', d_s)
        out = self.V_0 * (7 * self.E_0 * (1 - self.q) + 2 *
                          (1 - self.q / self.v) + (2 * self.E_0 - 0.2) * (1 - self.v))
        self.input = u
        return out

    def stable_params(self):
        seed = np.random.rand(1) * 0.07 + 0.01
        for _ in range(800):
            _ = self.get_bold(seed)

    def print_variable(self):
        print("f_in: ", self.f_in)
        print("s:", self.s)
        print("q:", self.q)
        print("v:", self.v)
    
    def get_bold(self, u):
        for i in range(799):
            _ = self.run(u)
        return self.run(u)

    def get_bold_frozen(self, u):
        cache_f = np.copy(self.f_in)
        cache_q = np.copy(self.q)
        cache_v = np.copy(self.v)
        cache_s = np.copy(self.s)
        for i in range(799):
            _ = self.run(u)
        bold = self.run(u)
        self.f_in = cache_f
        self.q = cache_q
        self.v = cache_v
        self.s = cache_s
        return bold

    def stable_bold(self, u):
        cache_f = np.copy(self.f_in)
        cache_q = np.copy(self.q)
        cache_v = np.copy(self.v)
        cache_s = np.copy(self.s)
        for i in range(20000):
            _ = self.run(u)
        bold = self.run(u)
        self.f_in = cache_f
        self.q = cache_q
        self.v = cache_v
        self.s = cache_s
        return bold


if __name__ == "__main__":
    model = BOLD(1)
    print(model.run(np.array(0.01)))
    model.print_variable()
    print("===============")
    print(model.stable_bold(np.array(1)))



