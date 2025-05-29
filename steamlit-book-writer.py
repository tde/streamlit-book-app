import streamlit as st
import os
import json
import logging
from llama_cpp import Llama

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Переменные окружения
model_path = os.getenv("MODEL_PATH", "/workspace/models/deepsex-34b.Q4_K_M.gguf")
book_dir = os.getenv("BOOK_DIR", "/workspace/book")

# Проверка путей
if not os.path.exists(model_path):
    st.error(f"Модель не найдена: {model_path}")
    st.stop()
if not os.path.exists(book_dir):
    os.makedirs(book_dir)

# Загрузка модели
@st.cache_resource
def load_model():
    logger.info("Загрузка модели...")
    llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=35)
    logger.info("Модель загружена")
    return llm

llm = load_model()

# Интерфейс Streamlit
st.title("NSFW Генератор Книг")
st.header("Создайте главу вашей книги")

# Форма ввода
with st.form("book_form"):
    plot = st.text_area("Сюжет книги", placeholder="Эротика: Алина анализирует ETH/HYN (1 ETH = 12,429 HYN)...", height=150)
    chapter_prompt = st.text_area("Промпт для главы", placeholder="Напиши сцену с анализом ETH/HYN, 1000 слов...", height=150)
    chapter_number = st.number_input("Номер главы", min_value=1, value=1, step=1)
    custom_vocab = st.text_input("Кастомный словарь (через запятую)", value="страсть, ETH/HYN, TradingView")
    max_length = st.slider("Максимальная длина (токены)", min_value=500, max_value=2000, value=1000)
    temperature = st.slider("Температура", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    submit = st.form_submit_button("Сгенерировать главу")

# Обработка запроса
if submit:
    with st.spinner("Генерация текста..."):
        custom_vocab_list = [word.strip() for word in custom_vocab.split(",") if word.strip()]
        context = f"Сюжет книги:\n{plot}\n\n"
        if custom_vocab_list:
            context += "Используй слова: " + ", ".join(custom_vocab_list) + "\n\n"
        prev_chapter_path = os.path.join(book_dir, f"chapter_{chapter_number-1}.txt")
        if os.path.exists(prev_chapter_path):
            with open(prev_chapter_path, "r") as f:
                prev_chapter = f.read()
            context += f"Предыдущая глава:\n{prev_chapter}\n\n"
        context += f"Текущая глава {chapter_number}:\n{chapter_prompt}"

        token_count = len(llm.tokenize(context.encode()))
        if token_count > 8192:
            st.error("Контекст превышает лимит токенов (8192)")
            st.stop()

        logger.info(f"Генерация текста (токенов: {token_count})")
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

        st.success(f"Глава {chapter_number} сгенерирована!")
        st.subheader("Результат")
        st.write(generated_text)
        st.download_button(
            label="Скачать главу",
            data=generated_text,
            file_name=f"chapter_{chapter_number}.txt",
            mime="text/plain"
        )