# ConveyorBoxDetect
## Для установки пакетов pip:

pip3 install -r requirements.txt


## Запуск:

python3 pyqtGUI.py

Появится окно настройки, в котором можно обрезать кадр с помощью мыши, выделив область прямоугольником. Также отображается порт, к которому подключен дальномер. Если камера отсутствует, кадр будет залит синим цветом, до тех пор пока камера не будет подключена.
Нажав на кнопку Crop, появится окно, в котором отображаются два изображения: 
- Видеопоток с камеры
- Контуры объектов

Настройка четкости контуров осуществляется с помощью ползунка. Оптимальное значение 80-100, но можно варьировать и подстраивать по ситуации. 
Также в окне отображаются параметры, приходящие с дальномера: расстояние от камеры до конвейера, которое измеряется в начале работы программы и статично, и расстояние от камеры до коробки. При подъезде коробки под датчик, начинается процесс детектирования. Пока коробка едет под датчиком, все распознанные контуры добавляются в список. После того как коротка покидает область работы дальномера, начинается отбор подходящего контура: берется приблизительно середина списка, после чего исходный список обрезается до значений contourList[(middleIndices-n):(middleIndices+n)], где n - произвольное число дополнительных кадров, среди которых производится отбор. Использование начения middleIndices обосновано тем, что коробка будет нахождиться под прямым углом к камере, что позволит более точно определить ее габариты. Такой подход позволяет обеспечить безостановочную работу конвеера. После того как мы обрезали список contourList, отбираем контур с самой большой площадью и после отъезда коробки выводим значения на экран. 
Для эффективной работы нужно чтобы в кадре одновременно находилась только одна коробка. Если их будет несколько, определится самая большая. 

# TODO:
- Добавить опрос подключенных к COM-портам устройств (начато)

- Написан генератор QR-кода и перевод его в битовую маску (файл QR_generate.ipynb). Передача должна осуществляться через COM-порт. Прикрутить это все к конечному коду.

- Переписать протокол общения с дальномером

- Заменить текущий ультразвуковой дальномер на оптический

- Проработать случай когда в кадре несколько коробок

- Запуск и тестирование решения на Jetson Nano или другом компьютере. Для работы системы нужна графическая оболочка для отладки и настройки.

- Альтернативное решение: использовать tensorflow object detection api
