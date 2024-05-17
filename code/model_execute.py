import torch as tch
import torch.nn as nn
from generation.model_for_generation import model_for_generation
from tqdm import tqdm


class ModelExecute(nn.Module):

    def __init__(self, con, ctrl):
        super(ModelExecute, self).__init__()
        self.device = con.device
        self.model = model_for_generation(con)
        ctrl.insert_model(x=self.model, name="model", change=True, lr=con.lr)
        self.type = con.score_type
        self.con = con

    def get_loss(self, corp):
        inputs = [x["input"] for x in corp]
        text = [x["output"] for x in corp]
        return self.model.get_loss(inputs, text)

    def get_output(self, ld, res):
        for corp in tqdm(ld):
            inputs = [x["input"] for x in corp]
            output = self.model.generate(inputs)
            res += output
        print( )

    def test(self, loader):
        # order-main
        if self.type == "order-main":
            with tch.no_grad( ):
                t1, t2 = loader[0], loader[1]
                o1, o2 = [ ], [ ]
                self.get_output(t1, o1)
                self.get_output(t2, o2)
            print( )
            return {"output1": o1, "output2": o2}

        # order-cwio
        elif self.type == "order-cwio":
            output = [ ]
            with tch.no_grad( ):
                for corp in tqdm(loader):
                    x = [x["input"] for x in corp]
                    output += self.model.generate(x)
            print( )
            return {"output": output}

        # rate (rule-learnability)
        elif self.type == "rate":
            output, data = [ ], [ ]
            with tch.no_grad( ):
                for corp in tqdm(loader):
                    x = [x["input"] for x in corp]
                    y = self.model.generate(x)
                    data += [x["data"] for x in corp]
                    output += y
            print( )
            return {"data": data, "output": output}

        # performance
        elif self.type == "performance":
            preds, refs, data = [ ], [ ], [ ]
            with tch.no_grad( ):
                for corp in tqdm(loader):
                    inputs = [x["input"] for x in corp]
                    ref = [x["output"] for x in corp]
                    output = self.model.generate(inputs)
                    preds.extend(output)
                    refs.extend(ref)
                    data.extend([x["data"] for x in corp])
            print( )
            return {'preds': preds, 'refs': refs, 'data': data}
