import numpy as np
from ensemble_net.calc import fss, probability_matched_mean


# Test FSS
dim = 40
test_m = np.zeros((2, dim, dim))
test_m[0, :, 8] = 1.
test_m[1, :, 9] = 1.
test_o = np.zeros((2, dim, dim))
test_o[:, :, 10] = 1.

print(fss(test_m, test_o, 0.5, neighborhood=1))


# Test PMM
field = np.random.rand(20, 20, 100)
pmm = probability_matched_mean(field, axis=-1)

