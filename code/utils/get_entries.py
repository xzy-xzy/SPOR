from corpusreader.benchmark_reader import Benchmark, select_files, select_tfile
from corpusreader.benchmark_reader_e2e import get_benchmark_e2e
from construction.check import delete_comp

seen = {'University', 'City', 'Artist', 'Politician', 'Airport', 'WrittenWork', 'ComicsCharacter',
        'MeanOfTransportation', 'SportsTeam', 'Building', 'Company', 'CelestialBody',
        'Monument', 'Athlete', 'Food', 'Astronaut'}


def get_entries(root, corp):
    if corp == "webnlg":
        print("Processing WebNLG entries...")
        corp = select_files(root + "train")
        bm = Benchmark()
        bm.fill_benchmark(corp)
        train = [e for e in bm.entries]
        corp_d = select_files(root + "dev")
        bm_d = Benchmark()
        bm_d.fill_benchmark(corp_d)
        dev = [e for e in bm_d.entries]
        corp_t = select_tfile(root + "test", "rdf-to-text-generation-test-data-with-refs-en.xml")
        bm_t = Benchmark()
        bm_t.fill_benchmark(corp_t)
        test = [e for e in bm_t.entries]
        test = delete_comp(train, dev + test, corp)
        return train, test
    elif corp == "e2e":
        print("Processing E2E entries...")
        train = get_benchmark_e2e(root + "train-fixed.no-ol.csv", True)
        dev = get_benchmark_e2e(root + "devel-fixed.no-ol.csv", True)
        test = get_benchmark_e2e(root + "test-fixed.csv", True)
        test = delete_comp(train, dev + test, corp)
        return train, test


