def cal_m(p, q, y_ngang, x_ngang, s):
    res = 0
    for y in range(1, s.__len__()):
        for x in range(1, s[1].__len__()):
            res += (x - x_ngang) ** p * (y - y_ngang) ** q * s[y][x]
    return res


s = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [
    0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

m00 = 0
for y in range(1, s.__len__()):
    for x in range(1, s[1].__len__()):
        m00 += s[y][x]
print('m00=', m00)

x_ngang = 0
y_ngang = 0
for y in range(1, s.__len__()):
    for x in range(1, s[1].__len__()):
        x_ngang += x * s[y][x]
        y_ngang += y * s[y][x]
x_ngang = x_ngang / m00
y_ngang = y_ngang / m00
print(f"x_ngang = {x_ngang}, y_ngang = {y_ngang}")


def cal_M(p, q):
    return cal_m(p, q, y_ngang, x_ngang, s) / (m00 ** ((p + q) / 2 + 1))


S1 = cal_M(2, 0) + cal_M(0, 2)
print("S1", S1)
S2 = cal_M(2, 0) * 2 - cal_M(0, 2) * 2 + 4 * (cal_M(1, 1) ** 2)
print("S2", S2)
S3 = (cal_M(3, 0) - 3 * cal_M(1, 2)) ** 2 + \
    cal_M(3, 0) - (3 * cal_M(2, 1)) ** 2
print("S3", S3)
S4 = (cal_M(3, 0) + cal_M(1, 2)) * 2 + (cal_M(0, 3) + cal_M(2, 1)) * 2
print("S4", S4)
S5 = (cal_M(3, 0) - 3*cal_M(1, 2)) * (cal_M(3, 0) +
                                      cal_M(1, 2)) * ((cal_M(3, 0) + cal_M(1, 2))**2 - 3*(cal_M(0, 3) + cal_M(2, 1))**2) + (3*cal_M(2, 1) - cal_M(0, 3)) * (cal_M(0, 3) + cal_M(2, 1)) * (3*(cal_M(3, 0) + cal_M(1, 2))**2 - (cal_M(0, 3) + cal_M(2, 1))**2)
print("S5", S5)
S6 = (cal_M(2, 0) - cal_M(0, 2)) * ((cal_M(3, 0) + cal_M(1, 2))**2 - (cal_M(0, 3) + cal_M(2, 1))
                                    ** 2) + 4 * cal_M(1, 1) * (cal_M(3, 0) + cal_M(1, 2)) * (cal_M(0, 3) + cal_M(2, 1))
print("S6", S6)
S7 = (3*cal_M(2, 1) - cal_M(0, 3)) * (cal_M(3, 0) + cal_M(1, 2)) * ((cal_M(3, 0) + cal_M(1, 2))**2 - 3*(cal_M(0, 3) + cal_M(2, 1))**2) + \
    (cal_M(3, 0) - 3*cal_M(1, 2)) * (cal_M(2, 1) + cal_M(0, 2)) * \
    (3*(cal_M(3, 0) + cal_M(1, 2))**2 - (cal_M(0, 3) + cal_M(2, 1))**2)
print("S7", S7)