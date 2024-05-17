import torch as tch
import torch.optim as optim
import os


class ModelControl:
    def __init__(self, t_name, con):
        self.name = t_name.replace('/', '_')
        self.con = con
        self.model_folder = con.trained_model_folder + self.name + '/'
        self.res_folder = con.res_folder + self.name + '/'
        self.output_folder = self.res_folder + 'output/'
        self.checkpoint = self.model_folder + 'checkpoint.txt'
        self.best_point = self.model_folder + 'bestpoint.txt'
        self.model_dir = [ ]
        self.dev_dir = [ ]
        self.test_dir = [ ]
        self.model_list = [ ]
        self.model_name_list = [ ]
        self.task_list = [ ]
        self.opt_list = [ ]
        self.scheduler_list = [ ]
        self.change_list = [ ]
        self.param_list = [ ]
        self.from_best_list = [ ]
        self.lr_list = [ ]
        self.decay_list = [ ]
        self.model_number = 0
        self.tot_steps = 0

        self.from_checkpoint = True
        self.best_mode = (not con.train) and (not con.without_dev)
        self.metric = -1e9

        self.device = con.device
        print("Device: ", end='')
        print(con.device)

        self.range_now = 1
        self.range_next = con.epoch

        if self.from_checkpoint == 1 and os.path.exists(self.checkpoint):
            x = open(self.checkpoint, 'r')
            s = x.readline( ).strip( )
            if s != "":
                self.range_now = int(s) + 1

        if os.path.exists(self.best_point):
            x = open(self.best_point, 'r')
            s = x.readline( ).strip( )
            if s != "":
                self.metric = float(s.split( )[1])

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        if not os.path.exists(self.res_folder):
            os.makedirs(self.res_folder)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def insert_model(self, x, name, change, lr):
        self.model_list.append(x)
        x.to(self.device)
        param_list = [y for y in x.parameters( ) if y.requires_grad]
        if param_list:
            opt = optim.Adam(param_list, lr=lr)
            self.opt_list.append(opt)
            self.param_list.append(param_list)
        else:
            self.opt_list.append(None)
            self.param_list.append(None)
        self.lr_list.append(lr)
        self.model_dir.append(self.model_folder + name)
        self.change_list.append(change)
        self.model_name_list.append(name)
        self.from_best_list.append(True if self.best_mode else (not change))
        if not change:
            self.model_list[self.model_number].train(mode=False)
        self.model_number += 1

    def insert_task(self, x, task_name):
        self.task_list.append(x)
        self.dev_dir.append(self.res_folder + task_name + "_dev_result.txt")
        self.test_dir.append(self.res_folder + task_name + "_test_result.txt")

    def is_train(self, mode):
        for i in range(self.model_number):
            if self.change_list[i]:
                self.model_list[i].train(mode=mode)

    def zero_grad(self):
        for i in range(self.model_number):
            self.model_list[i].zero_grad( )

    def step(self):
        for i in range(self.model_number):
            if self.change_list[i]:
                self.opt_list[i].step( )

    def cal_decay(self, tot_steps):
        for x in self.lr_list:
            if x:
                self.decay_list.append(x / tot_steps if tot_steps != 0 else 0)
            else:
                self.decay_list.append(None)

    def schedule(self):
        if self.con.lr_decay:
            for i in range(self.model_number):
                if self.change_list[i]:
                    for x in self.opt_list[i].param_groups:
                        x['lr'] -= self.decay_list[i]

    @staticmethod
    def get_dict(x):
        res = {k: v for k, v in x.named_parameters( ) if v.requires_grad}
        return res

    def save_model(self, epoch, upd):
        for i in range(self.model_number):
            if self.change_list[i]:
                tch.save(self.get_dict(self.model_list[i]), self.model_dir[i] + '_LATEST')
                tch.save(self.opt_list[i].state_dict( ), self.model_dir[i] + '_opt_LATEST')
        checkpoint_file = open(self.checkpoint, 'w')
        checkpoint_file.write("%d" % epoch)
        checkpoint_file.close( )
        if upd:
            for i in range(self.model_number):
                if self.change_list[i]:
                    tch.save(self.get_dict(self.model_list[i]), self.model_dir[i] + '_BEST')
                    tch.save(self.opt_list[i].state_dict( ), self.model_dir[i] + '_opt_BEST')
            best_point_file = open(self.best_point, 'w')
            best_point_file.write("%d %.8f\n" % (epoch, self.metric))
            best_point_file.close( )
        print("CheckPoint Updated\n")

    def load(self):
        for i in range(self.model_number):
            try:
                suffix = '_BEST' if self.from_best_list[i] else '_LATEST'
                self.model_list[i].load_state_dict(tch.load(self.model_dir[i] + suffix, map_location=self.device),
                                                   strict=not self.con.use_lora)
                if self.range_now <= self.range_next and self.con.train:
                    self.opt_list[i].load_state_dict(tch.load(self.model_dir[i] + '_opt' + suffix,
                                                              map_location=self.device))
                print("Checkpoint Found")
            except Exception as e:
                print(e)
                print("Checkpoint Not Found")

    def remove_opt(self):
        for i in range(self.model_number):
            for suffix in ['_BEST', '_LATEST']:
                name = self.model_dir[i] + '_opt' + suffix
                if os.path.isfile(name):
                    os.remove(name)
        print("Opt Removed")

    def remove_model(self):
        for i in range(self.model_number):
            for suffix in ['_BEST', '_LATEST']:
                name = self.model_dir[i] + suffix
                if os.path.isfile(name):
                    os.remove(name)
        print("Model Removed")

    def show_param(self):
        for i in range(self.model_number):
            model = self.model_list[i]
            for name, param in model.named_parameters( ):
                print(name, param)

    def deal_result(self, devlist, testlist, aim=0):
        if devlist:
            for x, y in zip(devlist, self.dev_dir):
                g = open(y, 'a')
                g.write(x[0])
            acc = devlist[aim][1]
            upd = (acc > self.metric)
            if upd:
                self.metric = acc
                if self.con.write_test_result_in_training:
                    for x, y in zip(devlist, self.test_dir):
                        g = open(y, 'w')
                        g.write(x[0])
            return upd
        if testlist:
            for x, y in zip(testlist, self.test_dir):
                g = open(y, 'w')
                g.write(x[0])
