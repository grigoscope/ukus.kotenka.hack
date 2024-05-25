import asyncio
from aiohttp import web
from aiohttp.web_middlewares import middleware

import _MainData as _md

@middleware
async def cors_middleware(request, handler):
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

async def handle(request):
    if request.method == 'OPTIONS':
        return web.Response()
    
    # Извлечение данных запроса
    data = await request.json()

    time_start = data['datetime1'] 
    time_stop = data['datetime2'] 
    print(f"Received request data: {data}")

    ret = _md.main(time_start, time_stop)

    # Возврат ответа
    response_data = {"range_0": str(ret[0]), "range_1": str(ret[1]), "range_2": str(ret[2]), "range_3": str(ret[3]) , 
                     "data_0" : list(ret[4]), "data_1" : list(ret[5]), "data_2" : list(ret[6]), "data_3" : list(ret[7])}
    
    print("Send Responce")
    return web.json_response(response_data)

async def init_app():
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_post('/echo', handle)
    app.router.add_options('/echo', handle)  # Добавьте обработку OPTIONS-запросов
    return app

if __name__ == '__main__':
    web.run_app(init_app(), host='localhost', port=8090)