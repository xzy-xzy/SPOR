import torch as tch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from generation.partial_grad_embbedings import UniteEmbedding


def padding(batch, padding_value, padding_method):
    max_len = max([len(x) for x in batch])
    if padding_method == "left":
        return [[padding_value] * (max_len - len(x)) + x for x in batch]  # Left-padding
    elif padding_method == "right":
        return [x + [padding_value] * (max_len - len(x)) for x in batch]  # Right-padding


class Seq2Seq(nn.Module):
    def __init__(self, con):
        super(Seq2Seq, self).__init__( )
        name = con.model
        self.tz = AutoTokenizer.from_pretrained(con.model_folder + name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(con.model_folder + name)

        if con.extra_tokens:
            print("Extra Tokens:")
            print(con.extra_tokens)
            self.tz.add_special_tokens({'additional_special_tokens': con.extra_tokens})
            self.model.resize_token_embeddings(len(self.tz))

        if con.use_lora:
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32,
                                     lora_dropout=0.1)
            self.model = get_peft_model(self.model, peft_config)
            if con.extra_tokens:
                emb = UniteEmbedding(self.model.get_input_embeddings( ), len(con.extra_tokens))
                self.model.set_input_embeddings(emb)
                if not self.model.config.tie_word_embeddings:
                    emb = UniteEmbedding(self.model.get_output_embeddings( ), len(con.extra_tokens))
                    self.model.set_output_embeddings(emb)

        self.device = con.device
        self.dim = self.model.config.hidden_size
        self.con = con
        self.type = "Seq2Seq"

    def to_dv(self, x):
        return tch.tensor(x).to(self.device)

    def get_loss(self, inputs, ref):
        input_ids = self.to_dv(padding(inputs, self.tz.pad_token_id, "right"))
        ref_ids = self.to_dv(padding(ref, -100, "right"))
        mask = input_ids.ne(self.tz.pad_token_id)
        result = self.model(input_ids=input_ids, attention_mask=mask, labels=ref_ids)
        gen_loss = result.loss
        return gen_loss

    def generate(self, inputs):
        input_ids = self.to_dv(padding(inputs, self.tz.pad_token_id, "right"))
        mask = input_ids.ne(self.tz.pad_token_id)
        output = self.model.generate(input_ids=input_ids, attention_mask=mask,
                                     max_length=128, num_beams=self.con.beam_width)
        output = self.tz.batch_decode(output, skip_special_tokens=True)
        return output


class Causal(nn.Module):
    def __init__(self, con):
        super(Causal, self).__init__( )
        name = con.model
        self.tz = AutoTokenizer.from_pretrained(con.model_folder + name)
        self.model = AutoModelForCausalLM.from_pretrained(con.model_folder + name)

        if con.extra_tokens:
            print("Extra Tokens:")
            print(con.extra_tokens)
            self.tz.add_special_tokens({'additional_special_tokens': con.extra_tokens})
            self.model.resize_token_embeddings(len(self.tz))

        if con.use_lora:
            if "Mistral" in name:
                target_modules = ["q_proj", "v_proj"]       # As default for other models
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                         lora_dropout=0.1, target_modules=target_modules)
            else:
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                         lora_dropout=0.1)
            self.model = get_peft_model(self.model, peft_config)
            if con.extra_tokens:
                emb = UniteEmbedding(self.model.get_input_embeddings( ), len(con.extra_tokens))
                self.model.set_input_embeddings(emb)
                if not self.model.config.tie_word_embeddings:
                    emb = UniteEmbedding(self.model.get_output_embeddings( ), len(con.extra_tokens))
                    self.model.set_output_embeddings(emb)

        if self.tz.pad_token_id is None:
            self.tz.pad_token = self.tz.eos_token
            self.tz.pad_token_id = self.tz.eos_token_id
            self.pad_equal_eos = True
        else:
            self.pad_equal_eos = (self.tz.pad_token_id == self.tz.eos_token_id)
        self.device = con.device
        self.con = con
        self.type = "Causal"

    def to_dv(self, x):
        return tch.tensor(x).to(self.device)

    def get_loss(self, inputs, ref):
        input_ids = self.to_dv(padding(inputs, self.tz.pad_token_id, "left"))
        ref_ids = self.to_dv(padding(ref, -100, "left"))
        mask = input_ids.ne(self.tz.pad_token_id)
        if self.pad_equal_eos:
            mask[:, -1] = True
        result = self.model(input_ids=input_ids, attention_mask=mask, labels=ref_ids, output_hidden_states=True)
        gen_loss = result.loss
        return gen_loss

    def generate(self, inputs):
        input_ids = self.to_dv(padding(inputs, self.tz.pad_token_id, "left"))
        mask = input_ids.ne(self.tz.pad_token_id)
        output = self.model.generate(input_ids=input_ids, attention_mask=mask,
                                     max_new_tokens=128, num_beams=self.con.beam_width)
        output = self.tz.batch_decode(output[:, input_ids.size(1):], skip_special_tokens=True)
        return output


def model_for_generation(con):
    if con.model_type == "seq2seq":
        return Seq2Seq(con)
    elif con.model_type == "causal":
        return Causal(con)

