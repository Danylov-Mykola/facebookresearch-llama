# import sys
import torch
import torch.optim as optim
import fire
from llama import Llama, Dialog
from typing import List, Optional
import sentencepiece as spm
import torch.nn as nn

# loss_function = nn.CrossEntropyLoss()


def load_tokenizer(tokenizer_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer


# Цю функцію треба розробити
def compute_loss_const(param):  # (output, target):
    return torch.tensor(param, requires_grad=True)


# import torch
import torch.nn.functional as functional
def compute_loss(outputs, targets, satisfaction_score):
    # Використовуємо крос-ентропію як основну функцію втрати
    # Припускаємо, що outputs - це логіти, а targets - це індекси відповідних токенів
    base_loss = functional.cross_entropy(outputs.view(-1, outputs.size(-2)), targets.view(-1))
    # Використовуємо задоволеність для модифікації основної втрати
    # Якщо задоволеність висока, втрата зменшується, і навпаки
    # Тут ми припускаємо, що satisfaction_score вже нормалізовано до [0, 1]
    loss = base_loss * (1 - satisfaction_score)
    return loss


def train_on_dialog(model, dialog_history, tokenizer, optimizer, satisfaction_score):
    full_dialog = "Повний текст. Питання. Відповіді. Остання відповідь."
    last_responce = "Остання відповідь."
    satisfaction_score = 0.5

    encoded_full_dialog = torch.tensor(tokenizer.encode(full_dialog), dtype=torch.long).unsqueeze(0)
    encoded_last_response = torch.tensor(tokenizer.encode(last_responce), dtype=torch.long).unsqueeze(0)
    # ... (попередній код)

    model.train()
    print("Checking if model parameters require gradients...")
    for name, param in model.named_parameters():
        print(f"{name} requires_grad: {param.requires_grad}")
    optimizer.zero_grad()
    outputs = model(encoded_full_dialog, start_pos=0)
    print(f"Does outputs have grad_fn after model training? {outputs.grad_fn is not None}")
    # Обрізаємо outputs до останніх 7 токенів
    # outputs має форму [1, 21, 32000], отже, ми беремо останні 7 елементів з другого виміру
    outputs_trimmed = outputs[:, -7:, :]

    # Виконуємо зворотне розповсюдження тільки якщо у outputs є grad_fn
    if outputs.grad_fn:
        base_loss = functional.cross_entropy(outputs_trimmed.view(-1, 32000), encoded_last_response.view(-1))
        loss = base_loss * (1 - satisfaction_score)
        loss.backward()
        optimizer.step()
        model.eval()
    else:
        print("Cannot perform backpropagation: outputs do not have grad_fn")

    # model.train()
    # optimizer.zero_grad()
    # outputs = model(encoded_full_dialog, start_pos=0)
    # # Зауваження: targets тут має бути частиною encoded_full_dialog, а не окремим викликом моделі
    # # Наприклад, якщо last_response є останнім реченням в full_dialog, targets може бути
    # # всіма токенами в encoded_full_dialog, крім першого, або підмножиною токенів
    # print(f"outputs: {outputs}")
    # # print(f"Shape of outputs: {outputs.size()}")
    # # print(f"Shape of targets (encoded_last_response): {encoded_last_response.size()}")
    #
    # # Обрізаємо outputs до останніх 7 токенів
    # # outputs має форму [1, 21, 32000], отже, ми беремо останні 7 елементів з другого виміру
    # outputs_trimmed = outputs[:, -7:, :]
    #
    # # Тут передбачається, що outputs має форму [batch_size, num_classes, seq_length]
    # # і targets має форму [batch_size, seq_length]
    # # base_loss = functional.cross_entropy(outputs.view(-1, outputs.size(-2)), encoded_last_response.view(-1))
    # base_loss = functional.cross_entropy(outputs_trimmed.view(-1, 32000), encoded_last_response.view(-1))
    # loss = base_loss * (1 - satisfaction_score)
    # loss.backward()
    # optimizer.step()
    # model.eval()


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 2048,  # maximum available 4096
        max_batch_size: int = 1,  # 8 is available at tje least, 1 consumes less memory
        max_gen_len: Optional[int] = None,
        train_interval: int = 2,
        dialog_log_path: str = "dialog_log.txt",
):
    llama_instance = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size
    )
    tokenizer = load_tokenizer(tokenizer_path)
    optimizer = optim.AdamW(llama_instance.model.parameters(), lr=0.001)

    dialog_history: List[Dialog] = [[]]
    user_input = ""
    iterations = 0

    while user_input.lower() != "exit":
        user_input = input("User: ")
        dialog_history[0].append({"role": "user", "content": user_input})

        # Генерація відповіді моделлю
        response = llama_instance.chat_completion(
            dialogs=dialog_history,  # dictionary
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )
        print(f"Assistant: {response}")
        for item in response:
            if 'generation' in item:
                dialog_history[0].append(item['generation'])

        # Періодичне навчання моделі. Увага: береться лише нульовий діалог! Якщо їх декілька - треба переробити.
        satisfaction_score = 0.5  # тимчасове значення, яке пізніше визначатиме модель самостійно
        if iterations % train_interval == 0:
            train_on_dialog(llama_instance.model, dialog_history[0], tokenizer, optimizer, satisfaction_score)
            # torch.save(llama_instance.model.state_dict(), 'ckpt_dir/consolidated.00.pth')

        iterations += 1


if __name__ == "__main__":
    fire.Fire(main)
