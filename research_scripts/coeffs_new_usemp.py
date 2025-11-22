import os
import csv
import sys
import tempfile
import multiprocessing as mp

from utils.libration_sense import (
    get_xf, du2km, initial_state_parser,
)
from utils.formula_creators import (
   alpha_finder_of_n, n_finder
)

ORBIT_TYPE = "L1"
ORBIT_MIN, ORBIT_MAX = 1, 251
GRID_DENSITY = 11
SEED = None
REUSE_NOISE = True
AMOUNT_OF_POINTS = 11_000

OUTPATH = "data/output/L1_test.csv"
HEADER = ["Orbit Number", "T", "Alpha1", "Alpha2", "n", "Deviation Max"]


def sort_csv_inplace(path: str, header: list[str]):
    """Сортирует CSV по 'Orbit Number' (первый столбец) и перезаписывает файл атомарно."""
    # читаем
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    # гарантируем заголовок
    if not rows or rows[0] != header:
        rows = [header] + [r for r in rows if r]

    data = []
    for r in rows[1:]:
        if not r:
            continue
        try:
            orb = int(r[0])
        except Exception:
            # пропустим битые строки
            continue
        data.append((orb, r))

    # сортировка по номеру орбиты
    data.sort(key=lambda t: t[0])

    # атомарная перезапись
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".tmp_sort_", suffix=".csv")
    os.close(fd)
    try:
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for _, row in data:
                w.writerow(row)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def compute_one(orbit_number: int):
    """
    Считает все величины для одной орбиты и возвращает кортеж для CSV:
    (orbit_number, T, alpha1, alpha2, n_perf, deviation_max_km)
    """
    try:
        _, z0, _, T, _, _ = initial_state_parser(ORBIT_TYPE, orbit_number)
        xf = get_xf(ORBIT_TYPE, orbit_number)

        n_perf, A_normed, y_du = n_finder(
            ORBIT_TYPE,
            orbit_number,
            xf,
            grid_density=GRID_DENSITY,
            seed=SEED,
            reuse_noise=REUSE_NOISE,
            amount_of_points=AMOUNT_OF_POINTS,
        )
        alpha, deviation_max_du = alpha_finder_of_n(A_normed, y_du, n_perf)
        deviation_max_km = du2km(deviation_max_du)

        alpha1 = float(alpha[0])
        alpha2 = float(alpha[1])

        return (orbit_number, float(T), alpha1, alpha2, float(n_perf), float(deviation_max_km), None)
    except Exception as e: 
        # вернём ошибку, чтобы главный процесс мог залогировать и идти дальше
        return (orbit_number, None, None, None, None, None, str(e))


def ensure_outfile(path: str):
    """Создать папку и CSV с заголовком, если его нет; вернуть множество уже посчитанных орбит (для резюма)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    processed = set()
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            rows = list(csv.reader(f))
        if rows and rows[0] == HEADER:
            for r in rows[1:]:
                if r and r[0].isdigit():
                    processed.add(int(r[0]))
        else:
            # перезапишем с заголовком, сохранив валидные строки данных
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(HEADER)
                for r in rows:
                    if r and r[0].isdigit():
                        w.writerow(r)
                        processed.add(int(r[0]))
    else:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

    return processed


def main():
    processed = ensure_outfile(OUTPATH)
    todo = [i for i in range(ORBIT_MIN, ORBIT_MAX + 1) if i not in processed]
    if not todo:
        print("Все орбиты уже посчитаны — делать нечего.")
        return

    print(f"К расчёту: {len(todo)} орбит(ы). Результат будет писаться по мере готовности в {OUTPATH}")

    ctx = mp.get_context("spawn")
    # чтобы не было овер-сабскрайба BLAS/NumPy в дочерних процессах, можно (опционально) перед запуском:
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")

    with open(OUTPATH, "a", newline="") as f:
        writer = csv.writer(f)

        # пул процессов
        with ctx.Pool(processes=os.cpu_count()) as pool:
            # imap_unordered возвращает результаты по мере готовности
            total = len(todo)
            done = 0
            # chunksize подберите эмпирически (1–4). При «тяжёлых» тасках разницы почти нет.
            for res in pool.imap_unordered(compute_one, todo, chunksize=1):
                orbit_n, T, a1, a2, nperf, dmax_km, err = res
                if err is None:
                    writer.writerow([orbit_n, T, a1, a2, nperf, dmax_km])
                    f.flush()
                else:
                    print(f"[ERROR] orbit {orbit_n}: {err}", file=sys.stderr)
                done += 1
                if done % 5 == 0 or done == total:
                    print(f"[{done}/{total}] записано; последняя орбита: {orbit_n}{' (error)' if err else ''}")

    print("Готово.")


if __name__ == "__main__":

    import time
    mp.freeze_support()
    start = time.time()
    main()
    print(f"Время выполнения {(time.time() - start) / 60} минут")

    sort_csv_inplace(OUTPATH, HEADER)
    print("CSV отсортирован по Orbit Number.")
