import multiprocessing as mp
from time import sleep


def f(v):
    sleep(1)
    print('toto {}'.format(v))


def main():
    p = mp.Pool(2)

    res = []
    for v in (1, 2, 3, 42):
        res.append(p.apply_async(f, (v,)))

    for r in res:
        r.wait()


if __name__ == '__main__':
    main()