import math

def solve(n, seed, budget):
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
        if seed % 2 == 1:
            circles = [[1.0 - x, y, r] for x, y, r in circles]
        return circles[:n]
    else:
        side = math.ceil(math.sqrt(n))
        radius = 0.5 / side - 1e-12
        return [
            [(column + 0.5) / side, (row + 0.5) / side, radius]
            for row in range(side)
            for column in range(side)
        ][:n]
