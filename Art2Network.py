import numpy as np
import sys


class ART2:
    a = 10
    b = 10
    c = 0.1
    d = 0.9
    e = sys.float_info.epsilon
    theta = 0
    alpha = 0
    vigilance = 0.99
    B = list()
    T = list()

    classes = 1

    def __init__(self, M, N):
        self.theta = 1/np.sqrt(M)
        self.alpha = 1/np.sqrt(M)

        self.T = np.zeros([N, M])
        self.B = np.random.rand(N, M) * (1/(1-self.d) * self.theta)

    def present(self, s, learn):
        norm = self.norm
        classes = self.classes

        w = s
        x = np.divide(w, (norm(w) + self.e))
        v = self.f(x)

        u = np.divide(v, (norm(v) + self.e))
        w = s + self.a * u
        x = np.divide(w, (norm(w) + self.e))
        p = u
        q = np.divide(p, (norm(p) + self.e))
        v = self.f(x) + self.f(q) * self.b

        y = np.dot(self.B, p)[0:max(classes, 1)]
        reset = True
        J = 0
        while reset:
            if (np.max(y) == -1) or (classes == 0):
                if classes == len(self.B) - 1:
                    return -1
                else:
                    self.classes += 1
                    classes += 1
                    J = classes
                    reset = False
            else:
                J = np.argmax(y)
                u = np.divide(v, (norm(v) + self.e))
                p = u + self.T[J] * self.d
                r = np.add(u, self.c * p) / (self.e + norm(u) + self.c * norm(p))
                n = norm(r)
                if n < (self.vigilance - self.e):
                    y[J] = -1
                else:
                    reset = False
        if learn:
            self.T[J] = self.alpha * self.d * u + (1 + self.alpha * self.d * (self.d - 1)) * self.T[J]
            self.B[J] = self.alpha * self.d * u + (1 + self.alpha * self.d * (self.d - 1)) * self.B[J]

        return J

    def f(self, vector):
        return np.array([v if np.abs(v) > self.theta else 0 for v in vector])

    @staticmethod
    def norm(vector):
        return np.sqrt(np.power(vector, 2).sum())
