def paraphase_check(y):
    # Exclude data units with (possible) paraphrase so as not to affect the calculation of PARENT
    # predicate paraphrase
    check_list = ["nationality", "date", "country"]
    for x in check_list:
        if x in y[1]:
            return False
    # numerical paraphrase
    cnt = 0
    for x in y[2]:
        if x.isdigit( ):
            cnt += 1
            if cnt == 4:
                return False
        else:
            cnt = 0
    return True
