from fastapi import FastAPI, File, UploadFile
import xlsxwriter
from uuid import uuid4
from huggingface_hub import snapshot_download
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
import pandas as pd
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings
from llama_cpp import Llama

from fastapi import FastAPI, Body, UploadFile, File
from fastapi.responses import FileResponse
from typing import Annotated

import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os


SYSTEM_PROMPT = "Ты — СмартКонсалт, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


repo_name = "IlyaGusev/saiga_13b_lora_llamacpp"
model_name = "model-q4_K.gguf"
embedder_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

snapshot_download(repo_id=repo_name, local_dir=".", allow_patterns=model_name)

model = Llama(
    model_path=model_name,
    n_ctx=2000,
    n_parts=1,
)

max_new_tokens = 1500
embeddings = HuggingFaceEmbeddings(model_name=embedder_name)

def get_uuid():
    return str(uuid4())

def build_index(file_paths, chunk_size, chunk_overlap):
    if file_paths:
        print('класс')
        documents = [load_single_document(path) for path in file_paths]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(documents)
        fixed_documents = []
        for doc in documents:
            doc.page_content = process_text(doc.page_content)
            if not doc.page_content:
                continue
            fixed_documents.append(doc)

        db = Chroma.from_documents(
            fixed_documents,
            embeddings,
            client_settings=Settings(
                anonymized_telemetry=False
            )
        )
        return db
    return None


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    assert ext in LOADER_MAPPING
    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    return loader.load()[0]


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


def upload_files(files):
    file_paths = ["resources/" + f.filename for f in files]
    return file_paths


def process_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines).strip()
    if len(text) < 10:
        return None
    return text





def user(message, history):
    new_history = history + [[message, None]]
    return "", new_history


def retrieve(history, db, k_documents):
    if db:
        last_user_message = history[-1][0]
        retriever = db.as_retriever(search_kwargs={"k": k_documents})
        docs = retriever.get_relevant_documents(last_user_message)
        retrieved_docs = "\n\n".join([doc.page_content for doc in docs])
    return retrieved_docs


def bot(
    history,
    retrieved_docs,
    top_p,
    top_k,
    temp
):
    if not history:
        return

    tokens = get_system_tokens(model)[:]
    tokens.append(LINEBREAK_TOKEN)

    for user_message, bot_message in history[:-1]:
        message_tokens = get_message_tokens(model=model, role="user", content=user_message)
        tokens.extend(message_tokens)
        if bot_message:
            message_tokens = get_message_tokens(model=model, role="bot", content=bot_message)
            tokens.extend(message_tokens)

    last_user_message = history[-1][0]
    if retrieved_docs:
        last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя контекст, ответь на вопрос: {last_user_message}"
    message_tokens = get_message_tokens(model=model, role="user", content=last_user_message)
    tokens.extend(message_tokens)

    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens.extend(role_tokens)
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temp
    )
    partial_text = ""
    for i, token in enumerate(generator):
        print(token)
        if token == model.token_eos() or (max_new_tokens is not None and i >= max_new_tokens):
            break
        partial_text += model.detokenize([token]).decode("utf-8", "ignore")
        history[-1][1] = partial_text
    return partial_text

app = FastAPI()

origins = [""]
chunk_size = 250
chunk_overlap = 30
k_documents = 2
top_p = 0.9
top_k = 30
temp = 0.1  
file_paths = ["resources/" + f for f in os.listdir("resources")]
db = build_index(file_paths, chunk_size, chunk_overlap)


chatbot = []



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)


@app.post("/uploadfiles")
def create_upload_files(files: list[UploadFile]):
    global file_paths, db
    file_paths = file_paths + upload_files(files)
    for file in files:
        with open("resources/" + file.filename, "w", encoding='utf-8') as fin:
            fin.write(file.file.read().decode('utf-8'))
    db = build_index(file_paths, chunk_size, chunk_overlap)
    return 'Файлы добавлены'

    
@app.delete("/del_files")
def del_files(data=Body()):
    global file_paths, db
    del_paths = ["resources/" + f for f in data['msg']]
    for file in del_paths:
        if os.path.exists(file):
            os.remove(file)
    file_paths = [file for file in file_paths if file not in del_paths]
    db = build_index(file_paths, chunk_size, chunk_overlap)
    return 'Файлы удалены'


@app.post("/answer")
def question_answering(data=Body()):
    global chatbot
    print(chatbot)
    msg = data['msg']
    if file_paths:
        msg, chatbot = user(msg, chatbot)
        retrieved_docs = retrieve(chatbot, db, k_documents)
        result = bot(chatbot,retrieved_docs,top_p,top_k,temp)
        return {'msg':result}
    return {'msg':'Модель пуста, добавьте файлы в модель'}


@app.post("/clear_chat")
def clear_chat():
    global chatbot
    chatbot = [["", ""]]


@app.get("/files_list")
def get_files_list():
    global file_paths
    return {'msg':file_paths}


@app.get("/dialog")
def get_dialog():
    global chatbot
    return chatbot



@app.post("/excel")
def create_excel(file: UploadFile):
    global chatbot, db
    with open(file.filename, "wb") as fin:
            fin.write(file.file.read())
    df = pd.read_excel(file.filename)
    answers = []
    writer = pd.ExcelWriter("output.xlsx", engine="xlsxwriter")
    for msg in df['Вопрос']:
        msg, chatbot = user(msg, chatbot)
        retrieved_docs = retrieve(chatbot, db, k_documents)
        result = bot(chatbot,retrieved_docs,top_p,top_k,temp)
        answers.append(result)
        # df['Ответ'] = answers
        # df.to_excel(writer)
        # writer.save()
        print('Записал!')
    df['Ответ'] = answers
    df.to_excel(writer, index=False)
    writer.close()


    
    



# with gr.Blocks(
#     theme=gr.themes.Soft()
# ) as demo:
#     db = gr.State(None)
#     conversation_id = gr.State(get_uuid)
#     favicon = '<img src="https://cdn.midjourney.com/b88e5beb-6324-4820-8504-a1a37a9ba36d/0_1.png" width="48px" style="display: inline">'
#     gr.Markdown(
#         f"""<h1><center>{favicon}Saiga 13B llama.cpp: retrieval QA</center></h1>
#         """
#     )

#     with gr.Row():
#         with gr.Column(scale=5):
#             file_output = gr.File(file_count="multiple", label="Загрузка файлов")
#             file_paths = gr.State([])
#             file_warning = gr.Markdown(f"Фрагменты ещё не загружены!")

#         with gr.Column(min_width=200, scale=3):
#             with gr.Tab(label="Параметры нарезки"):
#                 chunk_size = gr.Slider(
#                     minimum=50,
#                     maximum=2000,
#                     value=250,
#                     step=50,
#                     interactive=True,
#                     label="Размер фрагментов",
#                 )
#                 chunk_overlap = gr.Slider(
#                     minimum=0,
#                     maximum=500,
#                     value=30,
#                     step=10,
#                     interactive=True,
#                     label="Пересечение"
#                 )


#     with gr.Row():
#         k_documents = gr.Slider(
#             minimum=1,
#             maximum=10,
#             value=2,
#             step=1,
#             interactive=True,
#             label="Кол-во фрагментов для контекста"
#         )
#     with gr.Row():
#         retrieved_docs = gr.Textbox(
#             lines=6,
#             label="Извлеченные фрагменты",
#             placeholder="Появятся после задавания вопросов",
#             interactive=False
#         )
#     with gr.Row():
#         with gr.Column(scale=5):
#             system_prompt = gr.Textbox(label="Системный промпт", placeholder="", value=SYSTEM_PROMPT, interactive=False)
#             chatbot = gr.Chatbot(label="Диалог").style(height=400)
#         with gr.Column(min_width=80, scale=1):
#             with gr.Tab(label="Параметры генерации"):
#                 top_p = gr.Slider(
#                     minimum=0.0,
#                     maximum=1.0,
#                     value=0.9,
#                     step=0.05,
#                     interactive=True,
#                     label="Top-p",
#                 )
#                 top_k = gr.Slider(
#                     minimum=10,
#                     maximum=100,
#                     value=30,
#                     step=5,
#                     interactive=True,
#                     label="Top-k",
#                 )
#                 temp = gr.Slider(
#                     minimum=0.0,
#                     maximum=2.0,
#                     value=0.1,
#                     step=0.1,
#                     interactive=True,
#                     label="Temp"
#                 )

#     with gr.Row():
#         with gr.Column():
#             msg = gr.Textbox(
#                 label="Отправить сообщение",
#                 placeholder="Отправить сообщение",
#                 show_label=False,
#             ).style(container=False)
#         with gr.Column():
#             with gr.Row():
#                 submit = gr.Button("Отправить")
#                 stop = gr.Button("Остановить")
#                 clear = gr.Button("Очистить")

#     # Upload files
#     upload_event = file_output.change(
#         fn=upload_files,
#         inputs=[file_output, file_paths],
#         outputs=[file_paths],
#         queue=True,
#     ).success(
#         fn=build_index,
#         inputs=[file_paths, db, chunk_size, chunk_overlap, file_warning],
#         outputs=[db, file_warning],
#         queue=True
#     )

#     # Pressing Enter
#     submit_event = msg.submit(
#         fn=user,
#         inputs=[msg, chatbot, system_prompt],
#         outputs=[msg, chatbot],
#         queue=False,
#     ).success(
#         fn=retrieve,
#         inputs=[chatbot, db, retrieved_docs, k_documents],
#         outputs=[retrieved_docs],
#         queue=True,
#     ).success(
#         fn=bot,
#         inputs=[
#             chatbot,
#             retrieved_docs,
#             top_p,
#             top_k,
#             temp
#         ],
#         outputs=chatbot,
#         queue=True,
#     )

#     # Pressing the button
#     submit_click_event = submit.click(
#         fn=user,
#         inputs=[msg, chatbot, system_prompt],
#         outputs=[msg, chatbot],
#         queue=False,
#     ).success(
#         fn=retrieve,
#         inputs=[chatbot, db, retrieved_docs, k_documents],
#         outputs=[retrieved_docs],
#         queue=True,
#     ).success(
#         fn=bot,
#         inputs=[
#             chatbot,
#             system_prompt,
#             conversation_id,
#             retrieved_docs,
#             top_p,
#             top_k,
#             temp
#         ],
#         outputs=chatbot,
#         queue=True,
#     )

#     # Stop generation
#     stop.click(
#         fn=None,
#         inputs=None,
#         outputs=None,
#         cancels=[submit_event, submit_click_event],
#         queue=False,
#     )

#     # Clear history
#     clear.click(lambda: None, None, chatbot, queue=False)

# demo.queue(max_size=128, concurrency_count=1)
# demo.launch()
# files = [File()]
# upload_files(files, file_paths)
