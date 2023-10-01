import requests

ADDRESS = 'http://127.0.0.1'
PORT = '8000'
URL = f"{ADDRESS}:{PORT}"
async def get_output(input_text):
    response = requests.post(f"{URL}/answer", json={'msg': input_text})
    return response.json()["msg"]

async def get_list_actual_files():
    try:
        file_names = requests.get(f"{URL}/files_list")
        return file_names.json()['msg']
    except requests.exceptions.ConnectionError:
        return "Сервер не отвечает, попробуйте повторить запрос позже."

async def add_files(files):
    requests.post("http://127.0.0.1:8000/uploadfiles", files = files)


async def delete_files(file_names):
    requests.delete("http://127.0.0.1:8000/del_files", json={'msg': file_names})

async def clear_dialog():
    requests.post("http://127.0.0.1:8000/clear_chat")




