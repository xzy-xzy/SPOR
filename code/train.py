from preprocess import preprocess
from utils.evaluation import evaluation
from tqdm import tqdm


def train(args=None):

    con, ctrl, loader = preprocess(True, args)
    t_loader, d_loader = loader
    epoch = ctrl.range_now
    no_upd_steps = 0
    ctrl.cal_decay(len(t_loader) * con.epoch)

    while True:

        if con.stop_method == "epoch":
            print("epoch: %d / %d" % (epoch, con.epoch))
            if epoch > con.epoch:
                break
        if con.stop_method == "patience":
            print("patience: %d / %d" % (no_upd_steps, con.patience))
            if no_upd_steps >= con.patience:
                break
        ctrl.is_train(True)
        print("epoch %d:" % epoch)
        num = 0

        for data in tqdm(t_loader):
            num += len(data)
            loss_list = [ ]
            for task in ctrl.task_list:
                loss_list.append(task.get_loss(data))
            loss = sum(loss_list)
            loss = loss / con.div_factor
            loss.backward( )
            if num >= con.batch_size:
                ctrl.step( )
                ctrl.zero_grad( )
                num = 0

        if num > 0:
            ctrl.step( )
            ctrl.zero_grad( )

        if not con.without_dev:
            print("")
            ctrl.is_train(False)
            devlist = [ ]
            for task in ctrl.task_list:
                res = task.test(d_loader)
                dev_res = evaluation(res, con, ctrl.output_folder, str(epoch))
                dev_res[0] = ("epoch %d:\n" % epoch) + dev_res[0] + "\n"
                devlist.append(dev_res)
            upd = ctrl.deal_result(devlist, None)
        else:
            upd = False

        if upd:
            no_upd_steps = 0
        else:
            no_upd_steps += 1

        ctrl.save_model(epoch, upd)
        epoch += 1

    ctrl.remove_opt( )


if __name__ == "__main__":
    train( )
