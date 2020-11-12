import numpy as np
# Parameters description: http://www.inf.ufsc.br/~aldo.vw/patrec/SNNS/UserManual/node206.html#SECTION0010122220000000000000
class ARTNetwork:
    Weights_T = list() #Output layer -> M x N -> Input layer
    Weights_W = list() #Input layer -> N x M -> Output layer
    Probability = list() #
    Mask = list()
    Scaler = list()
#
     # 0 <= x <= 1, 0.7 <= x <= 1 suggested
# Strength of the influence of the lower level in F1 by the middle level
# Parameter a defines the importance of the expection of F2 , propagated to F1.
# Affects speed of stabilization in F1
     # a > 0,  normally a >> 1
# Strength of the influence of the middle level in F1  by the upper level.
# For parameter b things are similar to parameter a.  A high value for b is even more important, because otherwise the network could become instable
    # b > 0, suggested b >> 1
# Used to compute the error
     # 0 < c < 1
#Output value of the F2  winner unit
    #d = 0.2 # 0 < d < 1
#Theta - kind of threshold
     # 0 < theta < 1

    def __init__(self, _input_size: int, _init_output_size:int = 1, _vp:float = 0.9, _a: float = 10, _b: float = 10, _c: float = 0.1, _theta: float = 0.0):
        self.vigilance_parameter = _vp
        self.a =_a
        self.b =_b
        self.c =_c
        self.theta =_theta
        for i in range(_input_size):
            self.Weights_W.append(list())
            for j in range(_init_output_size):
                self.Weights_W[i].append(1/(1+_input_size))
        for j in range(_init_output_size):
            self.Weights_T.append(np.ones(_input_size))

    def __insert_new_category(self):
        n = len(self.Weights_W)
        for i in range(n):
            self.Weights_W[i].append(1/(1+n))
        self.Weights_T.append(np.ones(n))

    def normalize(self, x):
        return

    def learn(self, data_set, cycles: int):
        for i in range(cycles):
            current_x = data_set[i % len(data_set)]
            self.Mask = np.ones(len(self.Weights_T))
            go_back = True
            while(go_back):
                best_category_index = self.__choose_best_category(current_x)
                if self.__compare_input(current_x, pattern=self.Weights_T[best_category_index], threshold=self.vigilance_parameter) == False:
                    # Not recognized, try again with another category (if available). If not possible - create new category
                    self.Mask[best_category_index] = 0
                    if max(self.Mask) > 0:
                        go_back = True
                        continue
                    else:
                        self.__insert_new_category()
                        best_category_index = len(self.Mask)
                self.__update_weights_T(current_x, best_category_index)
                self.__update_weights_W(current_x, best_category_index)

    @staticmethod
    def __compare_input(x, pattern, threshold):
        return np.divide(np.multiply(x,pattern),np.sum(x)) > threshold

    def __choose_best_category(self, x):
        for j in range(len(self.Weights_T)):
            self.Probability[j] = 0.0
            if self.Mask[j] == 0:
                continue
            for i in range(len(x)):
                self.Probability[j] += self.Weights_W[i,j] * x[i]
        return self.Probability.index(max(self.Probability))

    def __update_weights_W(self, x, best_category_index: int):
        for i in range(len(x)):
            self.Weights_W[i, best_category_index] = (self.Weights_T[best_category_index,i] * x[i]) / (self.a + np.sum(np.multiply(self.Weights_T[best_category_index], x)))

    def __update_weights_T(self, x, best_category_index: int):
        for i in range(len(x)):
            self.Weights_T[best_category_index, i] *= x[i]