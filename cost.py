import math as mt


def cost(a_final, real_ans):
    c = [(a_final[i] - real_ans[i]) ** 2 for i in range(0, len(a_final))]
    return sum(c)


def der_cost(a_final, real_ans):
    return 2 * (a_final - real_ans)


def cross_entropy(a_final, real_ans):
    a_final = a_final.flatten()
    real_ans = real_ans.flatten()
    a_final = [0.99999 if n == 1 else n for n in a_final]

    c = []
    for j in range(len(a_final)):
        c.append(real_ans[j] * mt.log(a_final[j], mt.e) + (1 - real_ans[j]) * mt.log(1 - a_final[j], mt.e))
    return - sum(c)


def der_cross_entropy(a_final, real_ans):
    return - real_ans / a_final + (1 - real_ans) / (1 - a_final)
