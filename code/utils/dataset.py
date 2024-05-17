from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):

    def __init__(self, con, corp, train):
        self.corp = [ ]
        self.length = 0
        tz = con.tokenizer
        for data, data_f, texts in corp:
            if con.model_type == "causal":
                data = data + " Output:"
            input_id = tz(data)["input_ids"]
            if train:
                for text in texts:
                    if con.model_type == "seq2seq":
                        label = tz(text)["input_ids"]
                        self.corp.append({"input": input_id, "output": label, "data": data_f})
                    else:
                        src_tgt = tz(data + text)["input_ids"] + [tz.eos_token_id]
                        src_length = len(input_id)
                        label = [-100] * src_length + src_tgt[src_length:]
                        self.corp.append({"input": src_tgt, "output": label, "data": data_f})
                    self.length += 1
            else:
                self.corp.append({"input": input_id, "output": texts, "data": data_f})
                self.length += 1
        if con.pre_check_mode:
            self.corp = self.corp[:con.pre_check_len]
            self.length = con.pre_check_len

    def __getitem__(self, index):
        return self.corp[index]

    def __len__(self):
        return self.length


def create_loader(corp, con):
    if con.train:
        t_set = MyDataset(con, corp[0], True)
        d_set = MyDataset(con, corp[1], False)
        t_loader = DataLoader(dataset=t_set, batch_size=con.in_size_train, shuffle=True, collate_fn=lambda x: x)
        d_loader = DataLoader(dataset=d_set, batch_size=con.in_size_test, shuffle=False, collate_fn=lambda x: x)
        return t_loader, d_loader
    else:
        if con.score_type in ["order-main", "consistency"]:
            t1, t2 = MyDataset(con, corp[2], False), MyDataset(con, corp[3], False)
            t1 = DataLoader(dataset=t1, batch_size=con.in_size_test, shuffle=False, collate_fn=lambda x: x)
            t2 = DataLoader(dataset=t2, batch_size=con.in_size_test, shuffle=False, collate_fn=lambda x: x)
            return t1, t2
        else:
            t_set = MyDataset(con, corp[2], False)
            t_loader = DataLoader(dataset=t_set, batch_size=con.in_size_test, shuffle=False, collate_fn=lambda x: x)
            return t_loader



