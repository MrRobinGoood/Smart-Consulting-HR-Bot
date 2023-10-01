import shutil
from asyncio import sleep

from aiogram import Bot, Router
from aiogram import F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Filter
from requests_api import get_output, get_list_actual_files, add_files, delete_files, clear_dialog
from consts import ADMIN_ID
from telegram_bot.keyboards import admin_panel, complete_panel
from aiogram.fsm.context import FSMContext
from telegram_bot.state import AllStates
import os

router = Router()


@router.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        f"Привет, {message.from_user.first_name}, я HR-бот компании Smart Consulting. Ты можешь задавать мне вопросы, связанные с HR!\n\n<i>В целях демонстрации, функционал администратора доступен всем по команде /admin </i>", parse_mode="HTML")


@router.message(F.text == "/help")
async def cmd_help(message: Message):
    await message.answer("Бот предназначен для ведения диалога в контексте корпоративной базы знаний компании Smart Consulting.\n<b>Как использовать:</b>\nВы можете задать вопрос и получить ответ на него исходя из нормативной документации компании.\nВы также можете очистить историю диалога с ботом командой /clear.\nСкорость ответа бота варьируется от размера вопроса и в среднем составляет ~ 30 секунд.", parse_mode="HTML")


@router.message(F.text == "/clear")
async def cmd_help(message: Message):
    await clear_dialog()
    await message.answer("История диалога успешно удалена!💬")


class Admin(Filter):
    async def __call__(self, message: Message) -> bool:
        return message.from_user.id in [ADMIN_ID]


@router.message(Admin(), F.text == "/admin")
async def admin(message: Message, bot: Bot):
    await message.answer("Панель администратора", reply_markup=admin_panel)


@router.callback_query(Admin(), F.data == "list_files")
async def get_list_files(call: CallbackQuery):
    file_names = await get_list_actual_files()
    await call.message.answer('Список текущих файлов:\n' + '\n'.join(file_names))


@router.callback_query(Admin(), F.data == "add_file")
async def add_file(call: CallbackQuery, state: FSMContext):
    await call.message.answer("Подгрузите необходимые файлы, а затем нажмите кнопку готово:",
                              reply_markup=complete_panel)
    await state.set_state(AllStates.GET_FILES)


@router.message(Admin(), F.document, AllStates.GET_FILES)
async def cmd_document_admin(message: Message, bot: Bot):
    doc = message.document
    file = await bot.get_file(doc.file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "../company_data/" + doc.file_name)
    await message.reply("Файл загружен✅")


@router.callback_query(Admin(), F.data == "delete_file")
async def add_file(call: CallbackQuery, state: FSMContext):
    file_names = await get_list_actual_files()
    await call.message.answer(
        "Список текущих файлов:\n" + '\n'.join(file_names) + "\nНапишите через пробел файлы, которые вы хотите удалить:")
    await state.set_state(AllStates.DELETE_FILES)


@router.message(Admin(), F.text, AllStates.DELETE_FILES)
async def cmd_document_admin(message: Message, bot: Bot):
    file_names = message.text.split()
    await delete_files(file_names)
    await message.reply("Выбранные файлы удалены🗑")


@router.callback_query(Admin(), F.data == "complete", AllStates.GET_FILES)
async def add_file(call: CallbackQuery, state: FSMContext):
    files = []
    path = '../company_data'
    dirs = os.listdir(path)
    if dirs:
        for direct in dirs:
            files.append(('files', (direct, open(f'{path}/{direct}', 'rb'))))
            os.remove(os.path.join(path, direct))
        print(files)
        await add_files(files)

        await call.message.answer("Все файлы успешно добавлены!")
        await state.clear()
    else:
        await call.message.answer("Вы не добавили ни одного файла! Пожалуйста прикрепите файлы и нажмите готово.")


@router.callback_query(Admin(), F.data == "add_file")
async def add_file(call: CallbackQuery, state: FSMContext):
    await call.message.answer("Подгрузите необходимые файлы, а затем нажмите кнопку готово:")
    await state.set_state(AllStates.GET_FILES)


@router.message(F.photo)
async def cmd_photo(message: Message):
    await message.answer("Извините, я не понял ваш вопрос, пожалуйста напишите его текстом!")


@router.message(F.sticker or F.emoji)
async def cmd_sticker(message: Message):
    await message.answer("Извините, я не понял ваш вопрос, пожалуйста напишите его текстом!")


@router.message(F.video)
async def cmd_video(message: Message):
    await message.answer("Извините, я не понял ваш вопрос, пожалуйста напишите его текстом!")


@router.message(F.voice)
async def cmd_voice(message: Message):
    await message.answer("Извините, я не понял ваш вопрос, пожалуйста напишите его текстом!")


@router.message(F.document)
async def cmd_document(message: Message):
    await message.answer("Извините, я не понял ваш вопрос, пожалуйста напишите его текстом!")


@router.message(F.text)
async def text(message: Message, state: FSMContext):
    path = '../company_data'
    dirs = os.listdir(path)
    if dirs:
        await message.answer("Добавление файлов отменено!")
        for direct in dirs:
            os.remove(os.path.join(path, direct))
        await state.clear()

    a = await message.answer("Ваш вопрос принят в обработку♻")

    input_text = message.text
    await a.edit_text(await(get_output(input_text)))
