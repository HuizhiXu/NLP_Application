# -*- encoding: utf-8 -*-
"""
@Author : Sophia Xu
@File : t5.py
@Time : 2025/03/18 09:42:13
@Desc : T5模型的前向传播和预训练任务
"""

from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoConfig,AutoModel

# 模型和分词器的初始化
def initialize_model_and_tokenizer(model_ckpt,if_train=False):
    if if_train:
        model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
    else:
        model = AutoModel.from_pretrained(model_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    config = AutoConfig.from_pretrained(model_ckpt)
    return model, tokenizer, config

# 前向传播方式一
def t5_forward_one(model, input_ids, decoder_input_ids):
    """
    手动调用编码器和解码器
    """
    encoder_outputs = model.encoder(input_ids=input_ids)
    hidden_states = encoder_outputs[0]
    decoder_outputs = model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=hidden_states)
    return decoder_outputs.last_hidden_state

# 前向传播方式二
def t5_forward_two(model, input_ids, decoder_input_ids):
    """
    使用模型的前向传播方法
    """
    model.eval()
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    return outputs.last_hidden_state

# 无监督去噪训练
def unsupervised_denoising_training(model, tokenizer):
    input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
    labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    loss = model(input_ids=input_ids, labels=labels).loss
    print(f"Unsupervised Denoising Training Loss: {loss.item()}")

# 有监督的序列到序列训练
def supervised_seq2seq_training(model, tokenizer):
    input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
    labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids
    loss = model(input_ids=input_ids, labels=labels).loss
    print(f"Supervised Seq2Seq Training Loss: {loss.item()}")

#多数据对的训练
def multi_sentence_pairs_training(model,tokenizer):
    max_source_length = 512
    max_target_length = 128

    input_sequence_1 = "Welcome to NYC"
    output_sequence_1 = "Bienvenue a NYC"

    input_sequence_2 = "HuggingFace is a company"
    output_sequence_2 = "HuggingFace est une entreprise"

    # encode the inputs
    task_prefix = "translate English to French: "
    input_sequences = [input_sequence_1, input_sequence_2]

    encoding = tokenizer(
        [task_prefix + sequence for sequence in input_sequences],
        padding = "longest",
        max_length = max_source_length,
        truncation = True,
        return_tensors = "pt",
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # encode the targets
    target_encoding = tokenizer(
        [output_sequence_1, output_sequence_2],
        padding = "longest",
        max_length = max_target_length,
        truncation = True,
        return_tensors = "pt",

    )
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels[labels==tokenizer.pad_token_id] = -100

    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    print(f"multi sentence pairs Training Loss: {loss.item()}")

if __name__ == "__main__":
    model_ckpt = "t5-small"
    model, tokenizer, config = initialize_model_and_tokenizer(model_ckpt,if_train=False)

    # 前向传播测试
    input_ids = tokenizer("Studies have been shown that owning a cat is good for you", return_tensors="pt").input_ids
    decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids
    decoder_input_ids = model._shift_right(decoder_input_ids)  # _shift_right 是T5模型的一个内部方法

    last_hidden_state_one = t5_forward_one(model=model, input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    last_hidden_state_two = t5_forward_two(model=model, input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    print(f"Last Hidden State (Method One): {last_hidden_state_one}")
    print(f"Last Hidden State (Method Two): {last_hidden_state_two}")


    # 预训练
    model, tokenizer, config = initialize_model_and_tokenizer(model_ckpt,if_train=True)
    # 无监督去噪训练
    # unsupervised_denoising_training(model, tokenizer)

    # 有监督的序列到序列训练
    supervised_seq2seq_training(model, tokenizer)

    # 多数据对训练
    # multi_sentence_pairs_training(model,tokenizer)


    # 推理
    input_ids = tokenizer("translate English to German: My dog is cute.", return_tensors="pt").input_ids
   
    outputs = model.generate(input_ids, max_length=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))