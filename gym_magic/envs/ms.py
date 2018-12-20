import numpy as np

all_3x3_magic_squares = [
            [8, 1, 6, 3, 5, 7, 4, 9, 2],
            [6, 1, 8, 7, 5, 3, 2, 9, 4],
            [4, 9, 2, 3, 5, 7, 8, 1, 6],
            [2, 9, 4, 7, 5, 3, 6, 1, 8],
            [8, 3, 4, 1, 5, 9, 6, 7, 2],
            [4, 3, 8, 9, 5, 1, 2, 7, 6],
            [6, 7, 2, 1, 5, 9, 8, 3, 4],
            [2, 7, 6, 9, 5, 1, 4, 3, 8],
            ]

all_moves = [
            (0,4),
            (1,4),
            (2,4),
            (3,4),
            (5,4),
            (6,4),
            (7,4),
            (8,4),
            ]

def random_ms(scramble=0):

    n = len(all_3x3_magic_squares)
    ix = np.random.randint(0, n)
    ms = np.array(all_3x3_magic_squares[ix]).copy()

    for i in range(scramble):
        n = len(all_moves)
        ix = np.random.randint(0, n)
        x, y = all_moves[ix]
        tmp_val =  ms[y]
        ms[y] = ms[x]
        ms[x] = tmp_val

    return ms

def print_ms(ms):
    print "========="
    print ms[0:3]
    print ms[3:6]
    print ms[6:9]
    print "========="


if __name__ == "__main__":
     print_ms(random_ms(0))
     print_ms(random_ms(1))
     print_ms(random_ms(2))
