from aiogram import Bot, Dispatcher
import asyncio
from handlers import router
from consts import TOKEN
from aiogram.methods.delete_webhook import DeleteWebhook

bot = Bot(token=TOKEN)
dp = Dispatcher()


async def main():
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен")
