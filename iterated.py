# import sys
import torch
import torch.optim as optim
import fire
from llama import Llama, Dialog
from typing import List, Optional
import sentencepiece as spm
import torch.nn as nn

loss_function = nn.CrossEntropyLoss()


def load_tokenizer(tokenizer_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer


# Цю функцію треба розробити
def compute_loss(param):  # (output, target):
    return torch.tensor(param, requires_grad=True)


# import torch
# import torch.nn.functional as F
# def compute_loss(outputs, targets, satisfaction_score):
#     # Використовуємо крос-ентропію як основну функцію втрати
#     # Припускаємо, що outputs - це логіти, а targets - це індекси відповідних токенів
#     base_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
#     # Використовуємо задоволеність для модифікації основної втрати
#     # Якщо задоволеність висока, втрата зменшується, і навпаки
#     # Тут ми припускаємо, що satisfaction_score вже нормалізовано до [0, 1]
#     loss = base_loss * (1 - satisfaction_score)
#     return loss


def train_on_dialog(model, dialog_history, tokenizer, optimizer):
    print(f"dialog_history:\\n {dialog_history}")
    # Об'єднання усього діалогу в один текст
    full_dialog = " ".join([dialog["content"] for dialog in dialog_history[0]])
    print(f"full_dialog:\\n {full_dialog}")

    # Перетворення об'єднаного діалогу в тензор
    encoded_full_dialog = torch.tensor(tokenizer.encode(full_dialog), dtype=torch.long).unsqueeze(0)
    # Обчислення втрати та оновлення ваг
    model.train()
    optimizer.zero_grad()
    outputs = model(encoded_full_dialog, 0)  # формує тензор. Тут 0 - це start_pos, обов'язковий параметр
    loss = compute_loss(1.0)  # Функція compute_loss потрібно розробити
    loss.backward()
    optimizer.step()
    model.eval()
    return loss


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

        # Періодичне навчання моделі
        if iterations % train_interval == 0:
            train_on_dialog(llama_instance.model, dialog_history, tokenizer, optimizer)
            # torch.save(llama_instance.model.state_dict(), 'ckpt_dir/consolidated.00.pth')

        iterations += 1


if __name__ == "__main__":
    fire.Fire(main)
