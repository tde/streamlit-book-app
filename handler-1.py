import json
import os
import logging
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

storage_path = os.getenv("STORAGE_PATH", "/runpod-volume")
model_path = f"{storage_path}/models/deepsex-34b.Q4_K_M.gguf" 
book_dir = f"{storage_path}/book"

if not os.path.exists(book_dir):
    logger.info(f"Создание директории: {book_dir}")
    os.makedirs(book_dir)

# Загрузка модели
llm = None
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    logger.info("Загрузка модели...")
    llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=35)
    logger.info("Модель загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {str(e)}")

def handler(event, context):
    try:
        logger.info(f"Получен запрос: {event}")
        body = json.loads(event.get("body", "{}"))
        plot = body.get("plot", "")
        chapter_prompt = body.get("chapter_prompt", "")
        chapter_number = body.get("chapter_number", 1)
        custom_vocab = body.get("custom_vocab", [])
        max_length = body.get("max_length", 1000)
        temperature = body.get("temperature", 0.7)

        context = f"Сюжет книги:\n{plot}\n\n"
        if custom_vocab:
            context += "Используй слова: " + ", ".join(custom_vocab) + "\n\n"
        prev_chapter_path = os.path.join(book_dir, f"chapter_{chapter_number-1}.txt")
        if os.path.exists(prev_chapter_path):
            with open(prev_chapter_path, "r") as f:
                prev_chapter = f.read()
            context += f"Предыдущая глава:\n{prev_chapter}\n\n"
        context += f"Текущая глава {chapter_number}:\n{chapter_prompt}"

        token_count = len(llm.tokenize(context.encode()))
        logger.info(f"Токенов в контексте: {token_count}")
        if token_count > 8192:
            logger.error("Контекст превышает лимит токенов")
            return {"statusCode": 400, "body": "Контекст превышает лимит токенов (8192)"}

        logger.info("Генерация текста...")
        output = llm(
            context,
            max_tokens=max_length,
            temperature=temperature,
            stop=["</s>", "\n\n\n"],
            echo=False
        )
        generated_text = output["choices"][0]["text"]

        chapter_path = os.path.join(book_dir, f"chapter_{chapter_number}.txt")
        with open(chapter_path, "w") as f:
            f.write(generated_text)
        logger.info(f"Глава сохранена: {chapter_path}")

        response = {
            "statusCode": 200,
            "body": json.dumps({
                "chapter_number": chapter_number,
                "generated_text": generated_text,
                "token_count": token_count,
                "chapter_path": chapter_path
            })
        }
        logger.info("Запрос выполнен")
        return response

    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}")
        return {"statusCode": 500, "body": f"Ошибка: {str(e)}"}