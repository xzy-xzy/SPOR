def gen_name(con):
    name = [con.corp]
    name += [con.aspect]
    name += [con.model]
    name += ["seed=%d" % con.seed]
    if con.aspect == "systematicity":
        name += [con.sys]
    if con.aspect == "productivity":
        name += ["N=%d" % con.pro_len]
        name += [con.pro]
    if con.aspect == "order-invariance":
        name += [con.ord]
    if con.aspect == "substitutivity":
        name += [con.sub, con.sub_setting]

    # main_name is used for folders that save trained models and corresponding results
    main_name = ",".join(name)

    # train_name is used for folders that save training data
    con.train_name = "_".join(["train"] + name[0:2] + name[4:])

    rec = name[0:1] + name[4:]

    name = [con.corp, con.aspect]
    if con.aspect == "productivity":
        name += ["N=%d" % con.pro_len]
    if con.aspect == "order-invariance":
        name += [con.ord_aspect]

    # test_name is used for folders that save test data
    con.test_name = "_".join(["test"] + name)

    # sub_name is used for files that save test results (the same trained models can be used for different test)
    sub_name = "_".join(name)

    if con.aspect == "productivity":
        name = name[:-1]
    rec += name[2:]
    con.rec_name = "_".join(rec)

    return main_name, sub_name
