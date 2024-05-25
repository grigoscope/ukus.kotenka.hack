import clickhouse_connect
import numpy as np

#Можно выбрать - использовать локальную БД или в облаке
LOCAL = False

#Создание клиента БД
if LOCAL:
    client = clickhouse_connect.get_client(
        host='localhost',
        user='default',
        password='vano')
else:
    client = clickhouse_connect.get_client(
        host='y8crjtwlil.europe-west4.gcp.clickhouse.cloud',
        user='default',
        password='ZGroY.kB4tVoQ',
        secure=True)

#Выполнение запроса и переформатирование данных
def sql_query_execute(query):
    try:
        result = list(client.query(query).result_rows)
        dtime = np.array([item[0] for item in result])
        array = np.array([item[1] for item in result])
        return dtime, array
    except Exception as e:
        print("Failed to unload data from the database:\n",e)
        return None, None

#Однообразные запросы на расчет метрик за указанный период
#Метрики:
#Web Response - время ответа сервиса на внешний http-запрос
#Throughput - пропускная способность сервиса. Измеряется в запросах в минуту.
#APDEX - сводный синтетический показатель “здоровья” сервиса. Изменяется от 0 до 1. Чем ближе к 1, тем лучше
#Error - процент ошибок в обработанных запросах. 
def get_web_response(time_startm:str, time_end: str):
    sql_query = f"""
        SELECT
            point as time,
            sumOrNull(total_call_time) / sumOrNull(call_count) as " "
        FROM
            metrics_collector
        WHERE
            language = 'java'
            AND app_name = '[GMonit] Collector'
            AND scope = ''
            AND name = 'HttpDispatcher'
            AND time >= '{time_startm}' AND time <='{time_end}'
        GROUP BY 
            time
        ORDER BY
            time
    """
    return sql_query_execute(sql_query)


def get_throughput(time_startm:str, time_end: str):
    sql_query = f"""
        SELECT
            point as time,
            sumOrNull(call_count)
        FROM
            metrics_collector
        WHERE
            language = 'java'
            AND app_name = '[GMonit] Collector'
            AND scope = ''
            AND name = 'HttpDispatcher'
            AND time >= '{time_startm}' AND time <='{time_end}'
        GROUP BY 
            time
        ORDER BY
            time
    """
    return sql_query_execute(sql_query)

def get_apdex(time_startm:str, time_end: str):
    sql_query = f"""
        WITH
            sumOrNull(call_count) as s,
            sumOrNull(total_call_time) as t,
            sumOrNull(total_exclusive_time) as f
        SELECT
            point as time,
            (s + t/2) / (s + t + f) as " "
        FROM
            metrics_collector
        WHERE
            language = 'java'
            AND app_name = '[GMonit] Collector'
            AND scope = ''
            AND name = 'Apdex'
            AND time >= '{time_startm}' AND time <='{time_end}'
        GROUP BY 
            time
        ORDER BY
            time
    """
    return sql_query_execute(sql_query)

def get_error(time_startm:str, time_end: str):
    sql_query = f"""
        SELECT
            point as time,
            sumOrNullIf(call_count, name='Errors/allWeb') / sumOrNullIf(call_count, name='HttpDispatcher') as " "
        FROM
            metrics_collector
        WHERE
            language = 'java'
            AND app_name = '[GMonit] Collector'
            AND scope = ''
            AND name in ('HttpDispatcher', 'Errors/allWeb')
            AND time >= '{time_startm}' AND time <='{time_end}'
        GROUP BY 
            time
        ORDER BY
            time
    """
    dtime, _error = sql_query_execute(sql_query)

    #В данных "процента ошибок в обработанных запросах" есть пустые значения, заполним их нулями
    if(dtime is None or _error is None):
        return None, None

    for i in range(len(_error)):
        _error[i] = _error[i] if _error[i] is not None else 0.0
    
    #Из-за наличия None в оригнальном массиве, тип данных устанавливается некорректно
    _error = _error.astype("float64")

    return dtime, _error