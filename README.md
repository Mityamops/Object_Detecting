#Object detecting and tracking using opencv
-------------------------------------------
Ссылка на скачивание используемых видео
<https://drive.google.com/file/d/1GW967zsZtRkeITy6ebbRRikX8CG8YfNx/view?usp=drive_link>


Название файла  | Содержание файла
----------------------|----------------------
tracker.py            | Файл, используемый для отслеживания bounding box 
main.py               | Первая версия детектора и трекера, работающая на статичных камерах без помех
no_open_cv.py         | Версия детектора без использования opencv
snow_nrt.py           | Доработанная версия детектора main.py , которая игнорирует помехи ,такие как снег, дождь и т.д. Так же способна детектировать обьекты , на видео , которые заранее были стабилизированы файлом  video_stab_nrt.py
snow_v2.py            | Другая версия snow_nrt.py, отличие состоит в том , что в этой версии используется стабилизация видео в реальном времени(работает хуже , чем стабилизация всего видео сразу), использует файл video_stabilization.py
video_stab_nrt.py     | Стабилизатор видео, взято отсюда <https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/> 
video_stabilization.py| Стабилизатор видео, библиотека : vidgear
