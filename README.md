# MetroGuard

Детектор по данным одного лидара для объектов, входящих в габарит поезда
2,1 × 3,0 м. Он читает
ROS 2 `PointCloud2`, восстанавливает положение пути, строит геометрические
кандидаты и отделяет препятствия от стационарной инфраструктуры компактным
ансамблем деревьев. Веса входят в репозиторий; обучение и сеть при проверке не
нужны.

Состояния выхода: `CLEAR`, `OBSTACLE`, `UNKNOWN`. Последнее используется, когда
наблюдаемости недостаточно для безопасного ответа.

## Демо

Веб-интерфейс принимает запись rosbag2 `.db3`, показывает прогресс, ход решений,
расстояние и время обработки. Решение можно оценить и без загрузки файла: в демо
встроены три готовых сценария — запись с препятствием, чистый тоннель и
официальная синтетика. Для каждого доступны чистая геометрия, линейная модель и
ансамбль деревьев. По умолчанию выбран ансамбль — победитель зафиксированного
сравнения EXP-067.

Визуализация показывает вид сверху и сбоку, границы габарита 2,1 × 3,0 м,
найденные компоненты, ход состояний `CLEAR` / `OBSTACLE` / `UNKNOWN` и причину
решения. Для записи с препятствием также доступно синхронное видео облака точек.

```bash
docker build -f demo_app/Dockerfile -t metroguard-demo .
docker run --rm -p 8000:8000 -v metroguard-jobs:/data/jobs metroguard-demo
```

Открыть `http://localhost:8000`. Ограничение загрузки задаётся переменной
`METROGUARD_MAX_UPLOAD_BYTES`, каталог задач — `METROGUARD_WORK_DIR`. Число
одновременно принятых задач ограничивает `METROGUARD_MAX_PENDING_JOBS`;
срок хранения управляется `METROGUARD_JOB_RETENTION_SECONDS` и
`METROGUARD_CLEANUP_INTERVAL_SECONDS`. Переполненная очередь возвращает HTTP
429, а незавершённые после рестарта задачи переводятся в `failed`.

Конфигурация боевого сервиса, nginx и TLS находится в `deploy/`. Она
рассчитана на установку сборки в `/opt/metroguard`; сертификат Let's Encrypt
не входит в репозиторий и должен быть выпущен на сервере заранее.

Локальный запуск без Docker:

```bash
python3 -m venv .venv
.venv/bin/pip install -r demo_app/requirements.txt
.venv/bin/uvicorn demo_app.app:app --host 0.0.0.0 --port 8000
```

## ROS 2

Отдельный контейнер запускает потоковый узел для ROS 2 Humble:

```bash
docker build -f lidar_geometry/Dockerfile.mvp -t metroguard-ros .
docker run --rm --network host metroguard-ros \
  python3 -m lidar_geometry.ros2_mvp_node --ros-args \
  -p input_topic:=/sensing/lidar/hesai128/pointcloud \
  -p algorithm:=tree_hybrid
```

Результат публикуется в `/metro_guard/detection`. Поддержаны обе выданные
организаторами раскладки: 16-байтная XYZI и 26-байтная XYZIRT. Значение
`tree_hybrid` используется по умолчанию; для воспроизведения сравнений доступны
`geometry` и `linear_hybrid`.

## Проверка

```bash
python3 -m venv .venv
.venv/bin/pip install -r lidar_geometry/requirements-experiments.txt
.venv/bin/python -m unittest discover -s lidar_geometry -p 'test_*.py'
.venv/bin/python -m unittest discover -s demo_app/tests -p 'test_*.py'
.venv/bin/python tools/experiments.py check
```

Проверка запуска на двух записях организаторов:

```bash
.venv/bin/python -m lidar_geometry.smoke_test \
  --xyzi-bag /data/cloud_with_fake_obj_0.db3 \
  --xyzirt-bag /data/roundT_doubleT_0.db3
```

Исходные записи не публикуются. Скрипты экспериментов принимают пути к ним через
аргументы командной строки; точные команды и результаты сохранены в
`experiments/registry.json`.

## Что смотреть

- `RESULTS.md` — сравнение комбинаций, их сильные стороны и ограничения.
- `experiments/dashboard.html` — автономная таблица всех экспериментов.
- `experiments/registry.json` — машиночитаемый журнал гипотез и критериев качества.
- `lidar_geometry/artifacts/` — JSON-результаты ключевых сравнений.
- `presentation/` и `demo/` — презентация и синхронная визуализация облака.

Метрики синтетики построены без смешивания соседних кадров одного фона между
обучающим и валидационным наборами. Официальная запись не содержит покадровых
масок, поэтому оценка на ней является пространственной псевдоразметкой, а не
оценкой на скрытой выборке.
