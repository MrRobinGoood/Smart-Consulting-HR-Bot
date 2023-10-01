from aiogram.fsm.state import StatesGroup, State

class AllStates(StatesGroup):
    GET_FILES = State()
    DELETE_FILES = State()
