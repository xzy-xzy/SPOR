from utils.get_entries import get_entries
from construction.entry import Entry
from copy import deepcopy
import random
from tqdm import tqdm


def coefficient(a, b):
    r, s = 0, sum([a[x] for x in a])
    for x in a:
        r += ((a[x] / s) ** 0.5) * ((b[x] / s) ** 0.5)
    return 1 - r


def deal_sys(con):
    print("Construction for Systematicity")
    sys_type = con.sys
    train, test = get_entries(con.corp_folder, con.corp)
    cluster = { }
    mr = (con.corp == 'e2e')

    # To accelerate the construction in WebNLG, we first cluster the samples by main entity

    for corp in train, test:
        for e in corp:
            tri = Entry(e, mr).triples
            s = [x[0] for x in tri]
            o = [x[2] for x in tri]
            cnt = { }
            for x in s:
                cnt[x] = cnt[x] + 1 if x in cnt.keys( ) else 1
            for x in o:
                cnt[x] = cnt[x] + 1 if x in cnt.keys( ) else 1
            ent, max_cnt = None, -1
            for x in s:
                if cnt[x] > max_cnt:
                    ent, max_cnt = x, cnt[x]
            attr = ent
            if attr not in cluster.keys( ):
                cluster[attr] = [Entry(e, mr)]
            else:
                cluster[attr].append(Entry(e, mr))

    # It is harder to construct a large systematicity test set with a small number of distinct data units (e.g. E2E)
    # Therefore, we perform 20,000 constructions and choose the largest test set
    # Step when reaching the maximum: （WebNLG: 12,333 / E2E: 968)

    try_count = 20000
    step_max = 12333 if con.corp == "webnlg" else 968
    final_atom_set, final_comb_set, final_test_set = [ ], [ ], [ ]
    final_sum_lens, final_comb_pairs = 0, 0
    final_tc = -1

    for tc in tqdm(range(try_count)):

        for k, arr in cluster.items( ):
            random.shuffle(arr)

        '''
        # To reproduce the final construction directly without trying multiple constructions, insert this code block

        if tc != step_max - 1:
            continue
        '''

        atom_set, comb_set, test_set = [ ], [ ], [ ]
        comb_num_pairs = 0
        all_a_cnt, all_c_cnt = { }, { }

        for k, arr in cluster.items( ):
            train_atom, test = set( ), [ ]
            all_atoms = set( )
            blocked = set( )
            atom_candi, blocked_candi = [ ], [ ]
            v = sorted(arr, key=lambda x: len(x.triples))
            random.shuffle(arr)

            # Construct training set Atom(A) and test set

            while len(v) != 0:

                aim = v.pop( )      # Try to add "aim" to test set
                if aim.data in train_atom or len(aim.triples) == 1:
                    continue

                atoms = set(aim.triples)        # Atoms in "aim"
                flag = True
                for x in atom_candi:
                    in_atoms = [y for y in x.triples if y in atoms]
                    if len(in_atoms) > 1:       # At least one sample in A has more than one atom in "aim"
                        flag = False
                        break

                if flag:
                    # Collect samples that have only one atom in "aim" and have not been blocked
                    atom1 = [x for x in v if len([y for y in x.triples if y in atoms]) == 1 and x.data not in blocked]
                    cnt = {atom: 0 for atom in atoms}
                    for x in atom1:
                        for y in x.triples:
                            if y in atoms:
                                cnt[y] += 1

                    for atom in atoms:
                        if cnt[atom] == 0:      # At least one atom in "aim" cannot be covered
                            flag = False
                            break

                if flag:
                    # Add "aim" to test set
                    test.append(aim)

                    # Add "atom1" to A (Those already in A will not be added repeatedly)
                    for a in atoms:
                        all_atoms.add(a)
                    for x in atom1:
                        if x.data not in train_atom:
                            atom_candi.append(x)
                        train_atom.add(x.data)

                    # Block samples that have more than one atom in "aim"
                    for x in v:
                        in_atoms = [y for y in x.triples if y in atoms]
                        if len(in_atoms) > 1:
                            if x.data not in blocked:
                                blocked_candi.append(x)
                            blocked.add(x.data)

            # Construct training set Combination(C)

            test_cnt = {x: 0 for x in all_atoms}
            for x in test:
                for y in x.triples:
                    if y in all_atoms:
                        test_cnt[y] += 1

            # For E2E (a whole cluster), we remove atoms appear too few times in test set
            if con.corp == "e2e":
                all_atoms = {x for x in all_atoms if test_cnt[x] > 3}
                del_idx = [ ]
                for i in range(len(test)):
                    y = test[i].triples
                    if not all([x in all_atoms for x in y]):
                        del_idx.append(i)
                for i in reversed(del_idx):
                    test.pop(i)

            cnt = {x: 0 for x in all_atoms}

            def add(x, val):
                suc = True
                for y in x.triples:
                    if y in all_atoms:
                        cnt[y] += val
                        if cnt[y] <= 0:
                            suc = False
                return suc

            for x in atom_candi:
                for y in x.triples:
                    if y in all_atoms:
                        cnt[y] += 1

            atom_cnt = deepcopy(cnt)

            original_atom_candi = deepcopy(atom_candi)
            atom_candi = [(x, len([y for y in x.triples if y in all_atoms])) for x in atom_candi]
            atom_candi.sort(key=lambda x: x[1])
            combination_add = [ ]

            test_s = set([x.data for x in test])

            while len(blocked_candi) != 0:
                # Select a blocked sample with maximum V(x)
                blocked_candi.sort(key=lambda x: sum([0] + [atom_cnt[y] - cnt[y] for y in x.triples if y in all_atoms]))
                x = blocked_candi.pop( )
                if x.data in test_s:        # Do not select 0-atom sample
                    continue
                add(x, 1)
                num = len([y for y in x.triples if y in all_atoms])

                # Try to replace samples in A (traversing in ascending order of V(y))
                atom_candi.sort(key=lambda x: sum([0] + [cnt[y] - atom_cnt[y] for y in x[0].triples if y in all_atoms]))
                remove_idx = [ ]
                for i in range(len(atom_candi) - 1, -1, -1):
                    if atom_candi[i][1] > num or atom_candi[i][1] == 0:     # Atom number exceeds or 0-atom sample
                        continue
                    ret = add(atom_candi[i][0], -1)
                    if not ret:     # Some atoms won't be covered after removing this sample
                        add(atom_candi[i][0], 1)
                    else:
                        remove_idx.append(i)
                        num -= atom_candi[i][1]

                c = coefficient(atom_cnt, cnt)
                if num == 0 and c <= con.sys_co_limit:      # Keep equal atom number and low divergence
                    combination_add.append(x)
                    for i in remove_idx:
                        atom_candi.pop(i)
                else:
                    add(x, -1)
                    for i in remove_idx:
                        add(atom_candi[i][0], 1)

            # Validation and statistics

            atom_set_p = original_atom_candi
            comb_set_p = [x[0] for x in atom_candi] + combination_add
            sum1 = sum([len([y for y in x.triples if y in all_atoms]) for x in atom_set_p])
            sum2 = sum([len([y for y in x.triples if y in all_atoms]) for x in comb_set_p])
            assert sum1 == sum2
            pair_set = set()
            for x in test:
                for i in range(len(x.triples)):
                    for j in range(i + 1, len(x.triples)):
                        pair_set.add((x.triples[i], x.triples[j]))
            pr1, pr2 = 0, 0
            for x in atom_set_p:
                for i in range(len(x.triples)):
                    for j in range(i + 1, len(x.triples)):
                        if (x.triples[i], x.triples[j]) in pair_set:
                            pr1 += 1
            for x in comb_set_p:
                for i in range(len(x.triples)):
                    for j in range(i + 1, len(x.triples)):
                        if (x.triples[i], x.triples[j]) in pair_set:
                            pr2 += 1
            assert pr1 == 0

            for x in atom_cnt:
                all_a_cnt[x] = atom_cnt[x]
                all_c_cnt[x] = cnt[x]

            test_set += [x.data for x in test]
            atom_set += [x.data for x in atom_set_p]
            comb_set += [x.data for x in comb_set_p]
            comb_num_pairs += pr2

        # Update the construction with the largest test set

        lens = [len(Entry(x).triples) for x in test_set]
        sum_lens = sum(lens)
        if sum_lens > final_sum_lens:
            final_sum_lens = sum_lens
            final_atom_set = atom_set
            final_comb_set = comb_set
            final_test_set = test_set
            final_comb_num_pairs = comb_num_pairs
            final_tc = tc

    print("Divergence:", coefficient(all_a_cnt, all_c_cnt))
    print("#Atoms (A/C):", sum([all_a_cnt[x] for x in all_a_cnt]), sum([all_c_cnt[x] for x in all_c_cnt]))
    print("#Samples (A/C):", len(final_atom_set), len(final_comb_set))
    print("#Units (A/C):", sum([len(Entry(x).triples) for x in final_atom_set]),
          sum([len(Entry(x).triples) for x in final_comb_set]))
    print("#Combination Pairs (C):", final_comb_num_pairs)
    lens = [len(Entry(x).triples) for x in final_test_set]
    sum_lens = sum(lens)
    print("#Samples (Test):", len(final_test_set))
    print("#Units (Test):", sum_lens)
    print("Step when reaching the maximum #Units (Test):", final_tc + 1)

    if sys_type == "atom":
        return (final_atom_set, final_test_set), (final_test_set,)
    elif sys_type == "combination":
        return (final_comb_set, final_test_set), (final_test_set,)
