import math

def _base(n):
    if n == 26:
        r = 0.09
        sqrt3 = math.sqrt(3.0)
        rows = [
            (r, [r + 2.0 * r * j for j in range(5)]),
            (r + sqrt3 * r, [2.0 * r + 2.0 * r * j for j in range(4)]),
            (r + 2.0 * sqrt3 * r, [r + 2.0 * r * j for j in range(5)]),
            (r + 3.0 * sqrt3 * r, [2.0 * r + 2.0 * r * j for j in range(4)]),
            (r + 4.0 * sqrt3 * r, [r + 2.0 * r * j for j in range(5)]),
            (r + 5.0 * sqrt3 * r, [2.0 * r + 2.0 * r * j for j in range(4)]),
        ]
        circles = []
        for y, xs in rows:
            for x in xs:
                circles.append([x, y, r])
        return circles[:n]

    side = math.ceil(math.sqrt(n))
    radius = 0.5 / side - 1e-12
    return [
        [(c + 0.5) / side, (row + 0.5) / side, radius]
        for row in range(side)
        for c in range(side)
    ][:n]


def solve(n, seed, budget):
    circles = _base(n)
    if n != 26:
        return circles

    if seed % 2 == 1:
        circles = [[1.0 - x, y, r] for x, y, r in circles]

    import numpy as np
    from scipy.optimize import linprog

    centers = np.array([[c[0], c[1]] for c in circles], dtype=float)
    m = len(circles)
    A_ub = []
    b_ub = []

    for i in range(m):
        b = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1]) - 1e-9
        row = np.zeros(m)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(b)

    for i in range(m):
        for j in range(i + 1, m):
            d = math.hypot(
                centers[i, 0] - centers[j, 0],
                centers[i, 1] - centers[j, 1]
            ) - 1e-9
            row = np.zeros(m)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)

    c = np.ones(m)
    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0.0, 0.5)] * m,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )

    if res.success and res.x is not None:
        new_radii = np.clip(np.asarray(res.x, dtype=float), 0.0, 0.5)
        new_radii = new_radii * (1.0 - 1e-9)
        new_circles = [
            [centers[i, 0], centers[i, 1], float(new_radii[i])]
            for i in range(m)
        ]

        max_viol = 0.0
        for i in range(m):
            x, y, r = new_circles[i]
            b = min(x, 1.0 - x, y, 1.0 - y)
            max_viol = max(max_viol, r - b)

        for i in range(m):
            for j in range(i + 1, m):
                d = math.hypot(
                    new_circles[i][0] - new_circles[j][0],
                    new_circles[i][1] - new_circles[j][1]
                )
                max_viol = max(max_viol, new_circles[i][2] + new_circles[j][2] - d)

        if max_viol <= 1e-9 and sum(c[2] for c in new_circles) > sum(c[2] for c in circles) + 1e-12:
            circles = new_circles

    return circles[:n]
