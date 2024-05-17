from preprocess import preprocess
from utils.evaluation import evaluation


def inference(args=None):
    con, ctrl, loader = preprocess(False, args)
    ctrl.is_train(False)
    testlist = [ ]
    for task in ctrl.task_list:
        res = task.test(loader)
        test_res = evaluation(res, con, ctrl.output_folder, "test")
        test_res[0] = "Test:\n" + test_res[0] + "\n"
        testlist.append(test_res)
    ctrl.deal_result(None, testlist)


if __name__ == "__main__":
    inference( )