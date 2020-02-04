import pygame
from pygame.locals import *
import random
import threading
import argparse
import sys


HEIGHT = None
WIDTH = None
BLOCK_HEIGHT = None
BLOCK_WIDTH = None
INSERSION_CUTOFF = 8
MEDIAN3_CUTOFF = 40

screen = None
lock = threading.RLock()
initialised = False
sort = None


def shuffle_sort(a):
    global lock
    for j in range(HEIGHT // BLOCK_HEIGHT):
        for i in range(WIDTH // BLOCK_WIDTH):
            rand_i = random.randint(0, len(a) - 1)
            k = i + j * (WIDTH // BLOCK_WIDTH)
            lock.acquire()
            a[k], a[rand_i] = a[rand_i], a[k]
            draw_changes(a, (k, rand_i))
            lock.release()
    sort(a)


def bogo_sort(a):
    global lock

    def in_order(a):
        for i in range(len(a) - 1):
            lock.acquire()
            if not a[i] <= a[i + 1]:
                return False
            lock.release()
        return True

    while not in_order(a):
        lock.acquire()
        random.shuffle(a)
        draw_changes(a, [i for i in range(len(a))])
        lock.release()


def bubble_sort(a):
    global lock
    lock.acquire()
    n = len(a)
    lock.release()
    for _ in range(n):
        for i in range(1, n):
            lock.acquire()
            if a[i - 1] > a[i]:
                a[i - 1], a[i] = a[i], a[i - 1]
                draw_changes(a, (i - 1, i))
            lock.release()


def shell_sort(a):
    global lock
    n = len(a)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            j = i
            lock.acquire()
            temp = a[i]
            temp2 = a[j - gap]
            lock.release()
            while j >= gap and temp2 > temp:
                lock.acquire()
                a[j] = temp2
                draw_changes(a, [j])
                j -= gap
                temp2 = a[j - gap]
                lock.release()
            lock.acquire()
            a[j] = temp
            draw_changes(a, [j])
            lock.release()
        gap //= 2


def heap_sort(a):
    global lock
    lock.acquire()
    n = len(a)
    lock.release()

    for i in range(n, -1, -1):
        heapify(a, n, i)

    for i in range(n - 1, 0, -1):
        lock.acquire()
        a[i], a[0] = a[0], a[i]
        draw_changes(a, (0, i))
        lock.release()
        heapify(a, i, 0)


def heapify(a, n, i):
    global lock

    hi = i
    L = 2 * i + 1
    R = 2 * i + 2

    lock.acquire()
    if L < n and a[i] < a[L]:
        hi = L
    if R < n and a[hi] < a[R]:
        hi = R
    lock.release()

    if hi != i:
        lock.acquire()
        a[i], a[hi] = a[hi], a[i]
        draw_changes(a, (i, hi))
        lock.release()
        heapify(a, n, hi)


def merge_sort(a):
    global lock
    if len(a) > 1:
        mid = len(a) // 2
        lock.acquire()
        L = a[:mid]
        R = a[mid:]
        lock.release()

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            lock.acquire()
            if L[i] < R[j]:
                a[k] = L[i]
                i += 1
            else:
                a[k] = R[j]
                j += 1
            draw_changes(a, [k])
            lock.release()
            k += 1

        while i < len(L):
            lock.acquire()
            a[k] = L[i]
            draw_changes(a, [k])
            lock.release()
            i += 1
            k += 1

        while j < len(R):
            lock.acquire()
            a[k] = R[j]
            draw_changes(a, [k])
            lock.release()
            j += 1
            k += 1


def cocktail_sort(a):
    global lock
    lock.acquire()
    n = len(a)
    lock.release()
    swapped = True
    start = 0
    end = n - 1
    while swapped:
        swapped = False

        for i in range(start, end):
            lock.acquire()
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                draw_changes(a, (i, i + 1))
                swapped = True
            lock.release()
        if not swapped:
            break
        swapped = False
        end -= 1
        for i in range(end - 1, start - 1, -1):
            lock.acquire()
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                draw_changes(a, (i, i + 1))
                swapped = True
            lock.release()
        start += 1


def quick_sort(a, lo=0, hi=None):
    global lock
    if hi is None:
        lock.acquire()
        hi = len(a) - 1
        lock.release()
    if hi <= lo:
        return

    j = partition(a, lo, hi)
    quick_sort(a, lo, j - 1)
    quick_sort(a, j + 1, hi)


def partition(a, lo, hi):
    global lock
    i = lo
    j = hi + 1
    lock.acquire()
    v = a[lo]
    lock.release()

    while True:
        lock.acquire()
        while a[i + 1] < v:
            i += 1
            if i == hi:
                i -= 1
                break
        lock.release()
        i += 1
        lock.acquire()
        while v < a[j - 1]:
            j -= 1
            if j == lo:
                j += 1
                break
        lock.release()
        j -= 1

        if i >= j:
            break

        lock.acquire()
        a[i], a[j] = a[j], a[i]
        draw_changes(a, (i, j))
        lock.release()
    lock.acquire()
    a[lo], a[j] = a[j], a[lo]
    draw_changes(a, (lo, j))
    lock.release()
    return j


def quick_sort2O(a, lo=0, hi=None):
    global lock
    if hi is None:
        lock.acquire()
        hi = len(a) - 1
        lock.release()
    if hi <= lo:
        return

    n = hi - lo + 1
    if n <= INSERSION_CUTOFF:
        insertion_sort(a, lo, hi + 1)
        return
    j = partition2O(a, lo, hi)
    quick_sort2O(a, lo, j - 1)
    quick_sort2O(a, j + 1, hi)


def partition2O(a, lo, hi):
    global lock
    n = hi - lo + 1
    lock.acquire()
    m = median3(a, lo, lo + n // 2, hi)
    a[m], a[lo] = a[lo], a[m]
    draw_changes(a, (m, lo))
    lock.release()
    i = lo
    j = hi + 1
    lock.acquire()
    v = a[lo]
    v1 = a[i + 1]
    v2 = a[j - 1]
    lock.release()

    while v1 < v:
        i += 1
        if i == hi:
            lock.acquire()
            a[lo], a[hi] = a[hi], a[lo]
            draw_changes(a, (lo, hi))
            lock.release()
            return hi
        lock.acquire()
        v1 = a[i + 1]
    i += 1
    while v < v2:
        j -= 1
        if j == lo + 1:
            return lo
        lock.acquire()
        v2 = a[j - 1]
        lock.release()
    j -= 1

    while i < j:
        lock.acquire()
        a[i], a[j] = a[j], a[i]
        draw_changes(a, (i, j))
        v1 = a[i + 1]
        lock.release()
        while v1 < v:
            i += 1
            lock.acquire()
            v1 = a[i + 1]
            lock.release()
        i += 1
        lock.acquire()
        v2 = a[j - 1]
        lock.release()
        while v < v2:
            j -= 1
            lock.acquire()
            v2 = a[j - 1]
            lock.release()
        j -= 1
    lock.acquire()
    a[lo], a[j] = a[j], a[lo]
    draw_changes(a, (lo, j))
    lock.release()
    return j


def quick_sort3(a, lo=0, hi=None):
    global lock
    if hi is None:
        lock.acquire()
        hi = len(a) - 1
        lock.release()
    if hi > lo:
        lt = lo
        gt = hi
        lock.acquire()
        val = a[lo]
        lock.release()
        i = lo + 1
        while i <= gt:
            lock.acquire()
            if a[i] < val:
                a[lt], a[i] = a[i], a[lt]
                draw_changes(a, (i, lt))
                lt += 1
                i += 1
            elif a[i] > val:
                a[i], a[gt] = a[gt], a[i]
                draw_changes(a, (i, gt))
                gt -= 1
            else:
                i += 1
            lock.release()
        quick_sort3(a, lo, lt - 1)
        quick_sort3(a, gt + 1, hi)


def median3(a, i, j, k):
    global lock
    mid = 0
    lock.acquire()
    if a[i] < a[j]:
        if a[j] < a[k]:
            mid = j
        elif a[i] < a[k]:
            mid = k
        else:
            mid = i
    else:
        if a[k] < a[j]:
            mid = j
        elif a[k] < a[i]:
            mid = k
        else:
            mid = i
    lock.release()
    return mid


def quick_sort3O(a, lo=0, hi=None):
    global lock
    if hi is None:
        lock.acquire()
        hi = len(a) - 1
        lock.release()

    n = hi - lo + 1
    if n <= INSERSION_CUTOFF:
        insertion_sort(a, lo, hi)
        return
    elif n <= MEDIAN3_CUTOFF:
        m = median3(a, lo, lo + n // 2, hi)
        lock.acquire()
        a[m], a[lo] = a[lo], a[m]
        draw_changes(a, (m, lo))
        lock.release()
    else:
        eps = n // 8
        mid = lo + n // 2
        m1 = median3(a, lo, lo + eps, lo + eps + eps)
        m2 = median3(a, mid - eps, mid, mid + eps)
        m3 = median3(a, hi - eps - eps, hi - eps, hi)
        ninther = median3(a, m1, m2, m3)
        lock.acquire()
        a[ninther], a[lo] = a[lo], a[ninther]
        draw_changes(a, (ninther, lo))
        lock.release()

    i = p = lo
    j = q = hi + 1
    lock.acquire()
    v = a[lo]
    lock.release()
    while True:
        lock.acquire()
        while a[i + 1] < v:
            i += 1
            if i == hi:
                i -= 1
                break
        lock.release()
        i += 1
        lock.acquire()
        while v < a[j - 1]:
            j -= 1
            if j == lo:
                j += 1
                break
        lock.release()
        j -= 1

        lock.acquire()
        if i == j and a[i] == v:
            p += 1
            a[p], a[i] = a[i], a[p]
            draw_changes(a, (i, p))
        lock.release()
        if i >= j:
            break

        lock.acquire()
        a[i], a[j] = a[j], a[i]
        draw_changes(a, (i, j))
        if a[i] == v:
            p += 1
            a[p], a[i] = a[i], a[p]
            draw_changes(a, (i, p))
        if a[j] == v:
            q -= 1
            a[j], a[q] = a[q], a[j]
            draw_changes(a, (j, q))
        lock.release()

    i = j + 1
    for k in range(lo, p + 1):
        lock.acquire()
        a[j], a[k] = a[k], a[j]
        draw_changes(a, (j, k))
        lock.release()
        j -= 1
    for k in range(hi, q - 1, -1):
        lock.acquire()
        a[i], a[k] = a[k], a[i]
        draw_changes(a, (i, k))
        lock.release()
        i += 1

    quick_sort3O(a, lo, j)
    quick_sort3O(a, i, hi)


def selection_sort(a):
    global lock
    for i in range(len(a)):
        lo = i
        for j in range(i + 1, len(a)):
            lock.acquire()
            if a[j] < a[lo]:
                lo = j
            lock.release()
        lock.acquire()
        a[i], a[lo] = a[lo], a[i]
        draw_changes(a, (i, lo))
        lock.release()


def insertion_sort(a, lo=0, end=None):
    global lock
    if end is None:
        lock.acquire()
        end = len(a) - 1
        lock.release()
    for i in range(lo, end + 1):
        j = i
        while j > lo and a[j] < a[j - 1]:
            lock.acquire()
            a[j], a[j - 1] = a[j - 1], a[j]
            draw_changes(a, (j, j - 1))
            lock.release()
            j -= 1


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def draw_changes(a, changes):
    global screen
    for change in changes:
        val = a[change] % 360
        color = pygame.Color(0)
        hsva = (val, 100, 100, 0)
        color.hsva = hsva
        w = HEIGHT // BLOCK_HEIGHT
        y = (change % w) * BLOCK_WIDTH
        x = (change // w) * BLOCK_HEIGHT
        rect = pygame.Rect(x, y, BLOCK_WIDTH, BLOCK_HEIGHT)
        try:
            pygame.draw.rect(screen, color, rect)
        except Exception:
            break
    # pygame.display.update()


def init(items):
    global initialised, lock
    increment = (360 * BLOCK_WIDTH) / WIDTH
    for i, c in enumerate(frange(0, 360, increment)):
        for j in range(HEIGHT // BLOCK_HEIGHT):
            color = pygame.Color(0)
            hsva = (c, 100, 100, 0)
            color.hsva = hsva
            rect = pygame.Rect(i * BLOCK_WIDTH, j * BLOCK_HEIGHT,
                               BLOCK_WIDTH, BLOCK_HEIGHT)
            lock.acquire()
            items.append(round(c + 360 * i, 3))
            try:
                pygame.draw.rect(screen, color, rect)
            except Exception:
                break
            lock.release()
    initialised = True


def main():
    global lock, initialised, sort, screen
    global WIDTH, HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT
    SORT_TYPES = {'bubble': bubble_sort, 'cocktail': cocktail_sort,
                  'selection': selection_sort, 'insertion': insertion_sort,
                  'shell': shell_sort, 'merge': merge_sort,
                  'heap': heap_sort, 'quick': quick_sort,
                  'quickO': quick_sort3O}
    parser = argparse.ArgumentParser()
    parser.add_argument('-sort', choices=SORT_TYPES.keys(),
                        default='bubble',
                        help=("choose sorting algorithm from bubble," +
                              " cocktail, selection, insertion, shell, " +
                              " merge, heap, quick, quick optimised"))
    parser.add_argument('-bsize', type=int,
                        default=4, help=("choose integer block size," +
                                         " preferreably power of 2"))
    args = vars(parser.parse_args(sys.argv[1:]))
    sort = SORT_TYPES[args['sort']]
    BLOCK_HEIGHT = BLOCK_WIDTH = args['bsize']
    if sort is not None:
        pygame.init()
        info = pygame.display.Info()
        HEIGHT = info.current_h
        WIDTH = info.current_w
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((WIDTH, HEIGHT),
                                         pygame.DOUBLEBUF |
                                         pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
        pygame.display.set_caption("Sorts")
        items = []
        init_t = threading.Thread(target=init,
                                  args=(items, ),
                                  daemon=True)
        shuffle_t = threading.Thread(target=shuffle_sort,
                                     args=(items, ),
                                     daemon=True)
        init_t.start()
        mode = 1
        while(True):
            event = pygame.event.poll()
            if initialised:
                initialised = False
                shuffle_t.start()
            if (event.type == QUIT or
                    (event.type == KEYDOWN and event.key == K_ESCAPE)):
                pygame.display.quit()
                sys.exit(0)
            elif event.type == MOUSEBUTTONDOWN:
                lock.acquire()
                if mode == 0:
                    pygame.display.set_mode((WIDTH, HEIGHT),
                                            pygame.DOUBLEBUF |
                                            pygame.FULLSCREEN)
                    pygame.mouse.set_visible(False)
                    mode = 1
                else:
                    pygame.display.set_mode((WIDTH // 2, HEIGHT // 2),
                                            pygame.DOUBLEBUF)
                    pygame.mouse.set_visible(True)
                    mode = 0
                draw_changes(items, [i for i in range(len(items))])
                lock.release()
            pygame.display.update()
            clock.tick(60)


if __name__ == "__main__":
    main()
