import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoModel, AutoConfig


'''
https://stackoverflow.com/questions/73948214/how-to-convert-a-pytorch-nn-module-into-a-huggingface-pretrainedmodel-object/74109727#74109727

'''
class MyConfig(PretrainedConfig):
    model_type = 'mymodel'
    def __init__(self, important_param=42, **kwargs):
        super().__init__(**kwargs)
        self.important_param = important_param

class MyModel(PreTrainedModel):
    config_class = MyConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = nn.Sequential(
                          nn.Linear(3, self.config.important_param),
                          nn.Sigmoid(),
                          nn.Linear(self.config.important_param, 1),
                          nn.Sigmoid()
                          )
    def forward(self, input):
        return self.model(input)

    # Now you can create (and obviously train a new model), save and then load your model locally


class Llama410mModel(PreTrainedModel):
    config_class = MyConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # self.model = nn.Sequential(
        #     nn.Linear(3, self.config.important_param),
        #     nn.Sigmoid(),
        #     nn.Linear(self.config.important_param, 1),
        #     nn.Sigmoid()
        # )

        ################## MODEL ###################
        from model_design.lit_llama_model import LLaMA, LLaMAConfig
        llama_size = '410M'
        # model_name = f'Llama_{llama_size}'
        _config = LLaMAConfig.from_name(llama_size)
        config.block_size = 1024
        config.vocab_size = 151643

        self.model = LLaMA(_config)
        # model.apply(model._init_weights)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids)





def main():

    config = MyConfig(4)
    # model = MyModel(config)
    model = Llama410mModel(config)

    model.save_pretrained('./my_model_dir')

    new_model = Llama410mModel.from_pretrained('./my_model_dir')
    # new_model


if __name__=='__main__':

    main()

