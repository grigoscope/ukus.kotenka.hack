
import numpy as np
import matplotlib.pyplot as plt
import sys

from pyESN import ESN

import _DBLoadMetrics as _ld
import _DataManipulate as _dm

TEMP_TIME_START = "2024-04-30 21:21:00"
TEMP_TIME_STOP  = "2024-05-01 23:44:00"

#При проверке без веба, можно вывести графики локально
VISUALISATION = False

def get_data(t_start = TEMP_TIME_START, t_stop = TEMP_TIME_STOP):
    print("Accessing the database")

    #Получение данных метрик за указанный период
    #_xxx - исходные данные, без обработки
    _dt , _response = _ld.get_web_response(t_start, t_stop)
    _ , _throughput = _ld.get_throughput(t_start, t_stop)
    _ , _apdex = _ld.get_apdex(t_start, t_stop)
    _ , _error = _ld.get_error(t_start, t_stop)

    #Проверка на наличие валидных данных
    if(_dt is None or _response is None or _throughput is None or _apdex is None or _error is None ):
        print("Check the connection to the database! Terminating the Program...")
        sys.exit(1)
    print("Data has been successfully loaded from the database")

    #Приведение данных к "стабильной" длинне, в данном случае 5000 элементов
    response = _dm.data_linearization(_response)
    throughput = _dm.data_linearization(_throughput)
    apdex = _dm.data_linearization(_apdex)
    error = _dm.data_linearization(_error)
 
    #Удаление одиночных выбросов значений медианным фильтром
    response = _dm.data_filtering_med(response, kernel_size=5)
    throughput = _dm.data_filtering_med(throughput, kernel_size=5)
    apdex = _dm.data_filtering_med(apdex, kernel_size=5)
    error = _dm.data_filtering_med(error, kernel_size=5)

    #xxx_ - данные прошедшие медианный фильтр
    response_ = response.copy()
    throughput_ = throughput.copy()
    apdex_ = apdex.copy()
    error_ = error.copy()

    #Дополнительная фильтрация фильтром Баттерворта
    #xxx - итоговые предобработанные данные
    response =  _dm.butterworth_filter(response)
    throughput =  _dm.butterworth_filter(throughput)
    apdex =  _dm.butterworth_filter(apdex)
    error =  _dm.butterworth_filter(error)

    print("Data preprocessing is complete")

    #Вернем все полученные данные
    return (_dt, [_response, _throughput, _apdex, _error], [response_, throughput_, apdex_, error_], [response, throughput, apdex, error])

def esn_work(esn: ESN, _med_data, _filt_data, minimal_limit, maximum_limit):
    
    #Для работы ESN требуются нормализованные данные
    med_data  = _dm.data_norm(_med_data)
    filt_data = _dm.data_norm(_filt_data)

    print("Start ESN work")

    #Имитация рабоы в режиме реального времени
    #ESN принимает полностью профильтрованное значение и пытается предсказать следующее значение, но так, как будто оно без фильтра
    #Типо такого  ESN (filt_data(i)) => med_data(i+1)
    #Чем-то напоминает работу автоэнкодера, сеть пытается по меньшему кол-ву информации, чем доступно, воспроизвести изначательное поведение ряда
    #Прогрев
    esn.reset_all()
    prediction = []
    for i in range(50):
        prediction.append(esn.fit(np.array([filt_data[i]]), np.array([med_data[i]]), alpha=0.99)[1])
    for i in range(50,len(filt_data)):
        prediction.append(esn.fit(np.array([filt_data[i]]), np.array([med_data[i]]), alpha=0.01)[1])

    print("ESN work complete")

    #Разравниваем полученные предсказания
    prediction = np.array(prediction).flatten()

    #Всего используется 4 критерия оценивания:

    #Первый - вычисляем ошибку предсказания, по полученному масиву находим 3*(стандартное отклонение) и используем его в качестве порога обнаружения аномалии 
    errors_1_2 = np.abs(prediction - med_data)
    #Нормализуем ошибку
    errors_1_2 = _dm.data_norm(errors_1_2)
    #Считаем порог
    threshold_1 = 3 * np.std(errors_1_2)
    
    #Второй - доп. усиление первого, если отклонение ошибки предсказания более 0.9, то вероятность аномалии усиливается
    threshold_2 = 0.9

    #Третий - используем Z-score с порогом 3
    #threshold_3 = 3
    mean = np.mean(prediction)
    std_dev = np.std(prediction)
    z_scores = np.array([(x - mean) / std_dev for x in prediction])
    threshold_3 = np.percentile(np.abs(z_scores), 99)

    #Четвертный - вычисляем ошибку в абсолютных величинах и делим на стандартное отклонение
    threshold_4 = 0.9
    d_std_a = np.std(_filt_data) 
    d_max = _filt_data.max()
    d_min = _filt_data.min()
    error_4 = np.fabs((prediction * (d_max-d_min) + d_min) - _med_data) / d_std_a
    #Нормализуем
    error_4 = _dm.data_norm(error_4)

    #Суммируем вероятности возникновения аномалии
    #Дополнительно проверяем, что бы значение ряда лежало в допустимом диапазоне
    anomalies_ultra = np.zeros_like(prediction)
    for i in range(len(anomalies_ultra)):
        if(_filt_data[i] < minimal_limit or _filt_data[i] > maximum_limit or i<50):
            anomalies_ultra[i] = 0
        else:
            anomalies_ultra[i] = 0.3 * (1) if errors_1_2[i] > threshold_1 else 0
            anomalies_ultra[i] += 0.2 * (1) if errors_1_2[i] > threshold_2 else 0
            anomalies_ultra[i] += 0.25 * (1) if np.abs(z_scores[i]) > threshold_3 else 0
            anomalies_ultra[i] += 0.25 * (1) if error_4[i] > threshold_4 else 0

    print("Anomalies preprocessing is complete")

    #Сглаживание и нормализация ряда вероятностей 
    anomalies_ultra = _dm.exponential_moving_average(anomalies_ultra, 0.07)
    anomalies_ultra =  _dm.data_norm(anomalies_ultra)

    #Определение диапазонов аномалий
    #Диапазоном считается ненулевая послежовательность вероятностей аномалии
    ranges = []
    current_range = []
    current_probabilities = []

    for i, prob in enumerate(anomalies_ultra):
        #Просто отсекаем нули
        if prob >= 0.001:
            if not current_range:
                current_range.append(i)
            current_probabilities.append(prob)
        else:
            if current_range:
                current_range.append(i-1)
                #Средняя вероятность аномалии по диапазону
                avg_probability = np.mean(current_probabilities)
                ranges.append((current_range[0], current_range[1], avg_probability))
                current_range = []
                current_probabilities = []

    #Закрываем последний диапазон, если он не закрыт
    if current_range:
        current_range.append(len(anomalies_ultra) - 1)
        avg_probability = np.mean(current_probabilities)
        ranges.append((current_range[0], current_range[1], avg_probability))

    print("The creation of anamaly probability ranges has been completed")

    return ranges, prediction, anomalies_ultra

def main(t_start = TEMP_TIME_START, t_stop = TEMP_TIME_STOP):
    #Получим все необходимоые данные за указанный период
    all_data = get_data(t_start = t_start, t_stop = t_stop)

    #Распределим данные
    #_dt, [_response, _throughput, _apdex, _error], [response_, throughput_, apdex_, error_], [response, throughput, apdex, error])
    datetime = all_data[0]
    raw_data = np.array(all_data[1])
    med_data = np.array(all_data[2])
    filt_data = np.array(all_data[3])

    #Создание ESN
    #В контексте этой задачи, ESN (эхо нейрость), будет проходиться по всему временному ряду и предсказывать следующие значение
    #Сеть будет постоянно доучиваться в процессе, что позволит обнаруживать не стандартное "поведение" временного ряда
    #По аналогии с фильтром Калмана, можно сказать, что сеть является динамической моделью поведения
    esn = ESN(n_inputs=1, n_outputs=1, n_reservoir=479, spectral_radius=0.85, random_state=42)

    #Получение списка диапазонов возможных аномалий, ряд предсказаний и ряд вероятности аномалии
    #Для веб запросов нет рамок по абсолютной величине
    ranges_request, prediction_request, anomalies_ultra_request = esn_work(esn, med_data[0], filt_data[0], 0, float('inf'))
    #Для пропускной способности нет рамок по абсолютной величине
    ranges_throughput, prediction_throughput, anomalies_ultra_throughput = esn_work(esn, med_data[1], filt_data[1], 0, float('inf'))
    #Нас не интересуют аномалии, если здоровье сервиса больше 0.98 из 1 
    ranges_apdex, prediction_apdex, anomalies_ultra_apdex = esn_work(esn, med_data[2], filt_data[2], 0, 0.98)
    #Нас не интересуют аномалии, если ошибки менее 5% из 100%
    ranges_error, prediction_error, anomalies_error  = esn_work(esn, med_data[3], filt_data[3], 0.05, float('inf')) 

    print("Prepare responce")

    #Устанавливаем дату по индексам
    ranges_request_ = [("","",0.0)]
    ranges_throughput_ = [("","",0.0)]
    ranges_apdex_ = [("","",0.0)]
    ranges_error_ = [("","",0.0)]

    #Поправочный коэфициент масштабирования
    k = 5000/len(datetime)
    for i in range(len(ranges_request)):
        ranges_request_.append( (datetime[int(ranges_request[i][0]/k)], datetime[int(ranges_request[i][1]/k)], ranges_request[i][2]) )
    for i in range(len(ranges_throughput)):
        ranges_throughput_.append( (datetime[int(ranges_throughput[i][0]/k)],datetime[int(ranges_throughput[i][1]/k)], ranges_throughput[i][2]) )
    for i in range(len(ranges_apdex)):
        ranges_apdex_.append( (datetime[int(ranges_apdex[i][0]/k)],datetime[int(ranges_apdex[i][1]/k)], ranges_apdex[i][2]) )
    for i in range(len(ranges_error)):
        ranges_error_.append( (datetime[int(ranges_error[i][0]/k)],datetime[int(ranges_error[i][1]/k)], ranges_error[i][2]) )

    #Визуализация результатов
    if(VISUALISATION):
        #Создаем графики
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        #Веб запросы
        axs[0, 0].plot(range(len(prediction_request)), _dm.data_norm(med_data[0]), label='WebRequest')
        axs[0, 0].plot(range(len(prediction_request)), prediction_request, label='Predicted', linestyle='-')
        axs[0, 0].plot(range(len(prediction_request)), _dm.data_norm(filt_data[0]), label='Filtered', linestyle='--')
        axs[0, 0].plot(range(len(prediction_request)), anomalies_ultra_request, label='Anomalies Ultra', linestyle='-')
        axs[0, 0].legend()

        #Веб запросы
        axs[0, 1].plot(range(len(prediction_throughput)), _dm.data_norm( med_data[1]), label='Throughput')
        axs[0, 1].plot(range(len(prediction_throughput)), prediction_throughput, label='Predicted', linestyle='-')
        axs[0, 1].plot(range(len(prediction_throughput)), _dm.data_norm(filt_data[1]), label='Filtered', linestyle='--')
        axs[0, 1].plot(range(len(prediction_throughput)), anomalies_ultra_throughput, label='Anomalies Ultra', linestyle='-')
        axs[0, 1].legend()

        #Веб запросы
        axs[1, 0].plot(range(len(prediction_apdex)), _dm.data_norm(med_data[2]), label='Apdex')
        axs[1, 0].plot(range(len(prediction_apdex)), prediction_apdex, label='Predicted', linestyle='-')
        axs[1, 0].plot(range(len(prediction_apdex)), _dm.data_norm(filt_data[2]), label='Filtered', linestyle='--')
        axs[1, 0].plot(range(len(prediction_apdex)), anomalies_ultra_apdex, label='Anomalies Ultra', linestyle='-')
        axs[1, 0].legend()

        #Веб запросы
        axs[1, 1].plot(range(len(prediction_error)), _dm.data_norm(med_data[3]), label='Error')
        axs[1, 1].plot(range(len(prediction_error)), prediction_error, label='Predicted', linestyle='-')
        axs[1, 1].plot(range(len(prediction_error)), _dm.data_norm(filt_data[3]), label='Filtered', linestyle='--')
        axs[1, 1].plot(range(len(prediction_error)), anomalies_error, label='Anomalies Ultra', linestyle='-')
        axs[1, 1].legend()

        plt.show()
    print("Responce ready")

    return ((ranges_request_, ranges_throughput_, ranges_apdex_, ranges_error_, _dm.data_linearization(filt_data[0],100) ,  _dm.data_linearization(filt_data[1],100),  _dm.data_linearization(filt_data[2],100),  _dm.data_linearization(filt_data[3],100)))

if __name__ == '__main__':
    print(main())