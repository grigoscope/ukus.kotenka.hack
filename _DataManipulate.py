import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import medfilt
from scipy.interpolate import interp1d

#Приведение набора данных к единому размеру для стаблизации работы нейросети
#При слишком большом разбросе размера входных данных, в ESN могут начаться автоколебания
def data_linearization(data, fit_to=5000):
    target_indices = np.linspace(0, len(data) - 1, fit_to)
    interpolation_function = interp1d(np.arange(len(data)), data, kind='linear')
    return interpolation_function(target_indices)

#Медианный фильтр для удаления одиночных выбросов данных
def data_filtering_med(data, kernel_size = 3):
    return medfilt(data, kernel_size=kernel_size)

#Фильтр Баттерворта для получения гладкой кривой из шумных входных данных
def butterworth_filter(data, order=3):
    #33.3 и 2.1 - магические числа подобранные для заданного чилса выборок (в нашем случае 5000)
    sampling_rate = 33.3
    cutoff_freq = 2.2

    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y_smoothed = filtfilt(b, a, data)
    return y_smoothed

#Фильтр бегущей экспоненциальной средней, для сглаживания углов в ряде вероятностей аномалии
def exponential_moving_average(data, alpha):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    return ema


#Нормализация данных 
def data_norm(data):
    return (data - data.min())  / (data.max() - data.min() + 0.0001)