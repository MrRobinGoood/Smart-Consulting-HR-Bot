from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

admin_panel= InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Список файлов', callback_data='list_files')],
    [InlineKeyboardButton(text='Добавить файлы', callback_data='add_file')],
    [InlineKeyboardButton(text='Удалить файлы', callback_data='delete_file')]

])

complete_panel = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Готово', callback_data='complete')]

])