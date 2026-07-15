# План действий после обучения mixed-condition Graph-PPO

Статус документа: протокол должен быть зафиксирован **до просмотра финальных
test-результатов**. Это отделяет выбор checkpoint от итоговой оценки и уменьшает
selection bias.

## 1. Исходный результат обучения

Предполагается, что завершён следующий запуск:

```bash
PYTHONPATH=src python -u research/train_pv_graph_ppo.py \
  --road-length 1000 \
  --num-lanes 4 \
  --train-densities 0.10,0.20,0.30,0.40 \
  --train-av-fractions 0.9,0.95,1.00 \
  --front-distance 30 \
  --back-distance 10 \
  --sensor-distance 60 \
  --cooldown-steps 1 \
  --warmup-steps 1000 \
  --warmup-cache \
  --episode-steps 1000 \
  --backend optimized \
  --applied-penalty 0.001 \
  --rejected-penalty 0.002 \
  --hidden-dim 128 \
  --message-layers 3 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --clip-param 0.20 \
  --lr 0.0002 \
  --entropy-coeff 0.01 \
  --vf-loss-coeff 1.0 \
  --vf-clip-param 10.0 \
  --workers 32 \
  --envs-per-worker 1 \
  --num-gpus 1 \
  --rollout-fragment-length 256 \
  --train-batch-size 16384 \
  --minibatch-size 1024 \
  --epochs 6 \
  --iterations 2000 \
  --stop-timesteps 20000000 \
  --checkpoint-freq 25 \
  --seed 20260715 \
  --out-dir /workspace/artifacts/rl/pv_graph_ppo/mixed_big_h128_l3_20m
```

Основной каталог:

```bash
cd /workspace
export PYTHONPATH=src

RUN=/workspace/artifacts/rl/pv_graph_ppo/mixed_big_h128_l3_20m
CKPT_ROOT="$RUN/checkpoints"
EVAL_ROOT="$RUN/evaluation_post_training"

mkdir -p "$EVAL_ROOT"
```

Файл `run_config.json` является авторитетным описанием запуска, а
`train_metrics.jsonl` — журналом обучения. Checkpoint нельзя переносить без
`run_config.json`, если при evaluation не будет явно передан `--run-config`.

## 2. Этап A — проверить целостность результата обучения

### 2.1. Убедиться, что обязательные артефакты существуют

```bash
test -s "$RUN/run_config.json"
test -s "$RUN/train_metrics.jsonl"
test -d "$CKPT_ROOT"

find "$CKPT_ROOT" -maxdepth 1 -mindepth 1 -type d \
  -printf '%f\n' | sort -V | tail -n 15
```

Должен присутствовать каталог вида:

```text
final_iter_XXXXXX_steps_00002000XXXX
```

Найти его и сохранить путь в переменной:

```bash
FINAL="$(find "$CKPT_ROOT" -maxdepth 1 -mindepth 1 -type d \
  -name 'final_iter_*' | sort -V | tail -n 1)"

test -n "$FINAL"
test -d "$FINAL"
echo "FINAL=$FINAL"
```

Если `final_iter_*` отсутствует, обучение не завершилось штатно. В этом случае
нужно использовать `interrupted_iter_*` только как временный checkpoint либо
возобновить обучение через `--restore-checkpoint`.

### 2.2. Проверить последнюю запись обучения

```bash
tail -n 1 "$RUN/train_metrics.jsonl" | python -m json.tool
```

Проверить:

- `sampled_env_steps >= 20000000`;
- `pv_speed_mean`, `episode_return_mean` и `total_loss` конечны;
- нет `NaN`, `Infinity` и резкого взрыва loss;
- в `pv_speed_by_condition` присутствуют все 12 сочетаний плотности и доли AV;
- `elapsed_seconds` выглядит согласованно с фактическим временем запуска.

Записи с суффиксами `_min` и `_max` в verbose-журнале являются агрегатами
RLlib, а не отдельными условиями обучения.

## 3. Этап B — restore smoke финального checkpoint

`evaluate_pv_graph_ppo.py` восстанавливает RLlib policy на CPU, извлекает веса
GNN и проверяет, что действия RLlib совпадают с действиями автономной inference
модели. Флаг `--smoke` дополнительно выполняет один короткий paired replication.

```bash
PYTHONPATH=src python -u research/evaluate_pv_graph_ppo.py \
  --checkpoint "$FINAL" \
  --densities 0.30 \
  --av-fraction 1.00 \
  --cooldown-steps 1 \
  --yield-cooldown-steps 1 \
  --smoke \
  --out-dir "$EVAL_ROOT/restore_smoke" \
  --prefix restore_smoke
```

Успешный результат должен содержать:

```text
restore_smoke=passed
```

Также должны появиться CSV, metadata JSON и небольшой график. Ошибки формы
action vector, отсутствующие model keys и несовпадение deterministic actions
являются блокирующими: до полноценной evaluation продолжать нельзя.

Предупреждение RLlib о `durable policy name` само по себе не является ошибкой:
evaluator сначала создаёт policy того же класса и только затем вызывает
`restore()`. Тем не менее исходный код, `run_config.json` и версия Ray должны
храниться вместе с checkpoint.

## 4. Этап C — выбрать checkpoint на validation seeds

### 4.1. Зафиксированный validation-протокол

- кандидаты: итерации 825, 875, 900 и final;
- условие: `rho=0.30`, `AV=1.00`;
- 20 paired repetitions;
- validation seed: `310001`;
- policy deterministic: evaluator использует `argmax`, exploration отключён;
- основная метрика: средняя скорость PV;
- дополнительная метрика: парное преимущество над Baseline2;
- фоновые скорость и поток используются как диагностика, но не входят в
  критерий выбора.

Validation seed нельзя повторно использовать для финальной test-матрицы.

### 4.2. Проверить наличие кандидатов

```bash
C825="$CKPT_ROOT/periodic_iter_000825_steps_000013516800"
C875="$CKPT_ROOT/periodic_iter_000875_steps_000014336000"
C900="$CKPT_ROOT/periodic_iter_000900_steps_000014745600"

test -d "$C825"
test -d "$C875"
test -d "$C900"
test -d "$FINAL"
```

### 4.3. Запустить одинаковую evaluation для всех кандидатов

```bash
CANDIDATES=(
  "iter0825|$C825"
  "iter0875|$C875"
  "iter0900|$C900"
  "final|$FINAL"
)

SELECTION_ROOT="$EVAL_ROOT/checkpoint_selection"
mkdir -p "$SELECTION_ROOT"

for ITEM in "${CANDIDATES[@]}"; do
  LABEL="${ITEM%%|*}"
  CKPT="${ITEM#*|}"

  echo ">> evaluating $LABEL: $CKPT"

  PYTHONPATH=src python -u research/evaluate_pv_graph_ppo.py \
    --checkpoint "$CKPT" \
    --densities 0.30 \
    --av-fraction 1.00 \
    --runs 20 \
    --seed 310001 \
    --workers 20 \
    --chunksize 1 \
    --torch-threads 1 \
    --warmup-steps 1000 \
    --measure-steps 2000 \
    --backend optimized \
    --cooldown-steps 1 \
    --yield-distance 30 \
    --yield-cooldown-steps 1 \
    --out-dir "$SELECTION_ROOT/$LABEL" \
    --prefix selection
done
```

Запуски выполняются последовательно. Для одного условия имеется только 20
независимых tasks, поэтому `--workers 20` уже использует весь доступный
параллелизм; `--workers 32` здесь ускорения не даст.

### 4.4. Проверить каждый validation-запуск

В каждом каталоге кандидата должны быть:

```text
selection_runs.csv
selection_summary.csv
selection_paired.csv
selection_paired_summary.csv
selection_metadata.json
selection.png
```

Ожидаемые размеры:

- `selection_runs.csv`: 20 runs × 5 modes = 100 строк данных;
- `selection_summary.csv`: 1 density × 5 modes = 5 строк;
- `selection_paired.csv`: 20 runs × 4 comparisons = 80 строк;
- `selection_paired_summary.csv`: 4 строки.

Проверить строку `mode=graph_ppo` в `selection_summary.csv` и строку
`comparison_mode=baseline2_matched` в `selection_paired_summary.csv`.

Ключевые поля:

- `mean_speed_priority_mean`;
- `mean_speed_priority_std` и `mean_speed_priority_sem`;
- `pv_speed_delta_mean`;
- `pv_speed_delta_ci_low`, `pv_speed_delta_ci_high`;
- `pv_speed_win_rate`;
- `graph_ppo_speed_advantage`;
- `rule_applied_mean`, `rule_rejected_mean`;
- `policy_left_requests_mean`, `policy_right_requests_mean`,
  `policy_stay_actions_mean`.

Baseline1, Baseline2, noop и left-if-safe должны совпадать между checkpoint
запусками вплоть до численной воспроизводимости. Если они различаются, значит
были изменены seed, конфигурация, порядок плотностей или исходный код.

### 4.5. Правило выбора checkpoint

1. Исключить checkpoint, который не проходит restore или содержит неполные
   результаты.
2. Среди оставшихся выбрать максимальный `mean_speed_priority_mean` для
   `graph_ppo`.
3. Убедиться, что преимущество над Baseline2 не получено ценой аномально
   большого числа rejected actions.
4. Если разница между двумя лучшими checkpoints меньше `0.05` единицы скорости
   PV, считать их практически равными и выполнить дополнительный tie-break:
   50 runs, новый validation seed `320001`, только для этих двух checkpoints.
5. Если tie-break также не даёт устойчивого различия, выбрать более ранний
   checkpoint. Это более консервативный выбор и уменьшает риск позднего
   переобучения.

Не выбирать checkpoint по training return или единичному максимальному
эпизоду.

После выбора записать точный путь:

```bash
BEST=/workspace/artifacts/rl/pv_graph_ppo/mixed_big_h128_l3_20m/checkpoints/REPLACE_WITH_SELECTED_CHECKPOINT

test -d "$BEST"
printf '%s\n' "$BEST" > "$RUN/best_checkpoint.txt"
```

## 5. Этап D — финальная held-out test-матрица

### 5.1. Зафиксированный test-протокол

- checkpoint: только выбранный на этапе C;
- плотности: `0.10, 0.20, 0.30, 0.40`;
- доли AV: `0.90, 0.95, 1.00`;
- 20 paired repetitions на сочетание;
- test seed: `410001`, не использовавшийся при выборе checkpoint;
- пять режимов запускаются автоматически:
  `baseline1_hdv`, `baseline2_matched`, `graph_noop`, `left_if_safe`,
  `graph_ppo`;
- policy deterministic;
- primary outcome: скорость PV;
- secondary diagnostics: time loss PV, остановки PV, общий и фоновый поток,
  скорость потока, lane changes, applied/rejected actions.

### 5.2. Запустить три AV-среза последовательно

```bash
TEST_ROOT="$EVAL_ROOT/final_test"
mkdir -p "$TEST_ROOT"

for AV in 0.90 0.95 1.00; do
  TAG="${AV/./p}"

  echo ">> final test, AV=$AV"

  PYTHONPATH=src python -u research/evaluate_pv_graph_ppo.py \
    --checkpoint "$BEST" \
    --densities 0.10,0.20,0.30,0.40 \
    --av-fraction "$AV" \
    --runs 20 \
    --seed 410001 \
    --workers 32 \
    --chunksize 1 \
    --torch-threads 1 \
    --warmup-steps 1000 \
    --measure-steps 2000 \
    --backend optimized \
    --cooldown-steps 1 \
    --yield-distance 30 \
    --yield-cooldown-steps 1 \
    --out-dir "$TEST_ROOT/av_$TAG" \
    --prefix "test_av_$TAG"
done
```

Не запускать три процесса одновременно: каждый из них уже использует до 32
CPU workers. GPU для evaluation не требуется.

Ожидаемые размеры для каждого AV-среза:

- runs: 4 densities × 20 runs × 5 modes = 400 строк;
- summary: 4 densities × 5 modes = 20 строк;
- paired: 4 densities × 20 runs × 4 comparisons = 320 строк;
- paired summary: 4 densities × 4 comparisons = 16 строк.

Итого полная test-матрица содержит 1200 строк per-run результатов.

### 5.3. Primary baseline cooldown

В основном новом протоколе установлены:

```text
Graph/noop/left-if-safe cooldown = 1
Baseline2 cooldown              = 1
```

Это соответствует решению уменьшить прежний пятишаговый cooldown. Старые
baseline-графики были рассчитаны с cooldown 5 и поэтому не являются полностью
идентичным reference для нового paired test.

Для связи со старым baseline после основной матрицы можно выполнить отдельную
sensitivity evaluation с:

```bash
--yield-cooldown-steps 5
```

Её результаты должны иметь отдельный каталог и явную метку
`baseline2_legacy_cd5`; смешивать два варианта в одной таблице нельзя.

## 6. Этап E — проверки качества финальной матрицы

### 6.1. Техническая целостность

Для каждого из трёх `*_metadata.json` проверить:

- восстановлен один и тот же `BEST` checkpoint;
- `road_length=1000`, `num_lanes=4`;
- `front_distance=30`, `back_distance=10`, `sensor_distance=60`;
- `cooldown_steps=1`, `yield_cooldown_steps=1`;
- `warmup_steps=1000`, `measure_steps=2000`;
- `hidden_dim=128`, `message_layers=3`;
- backend действительно `optimized`;
- `runs=20`, seed равен `410001`.

Дополнительно проверить:

- нет пустых, `NaN` и бесконечных метрик;
- `effective_density_mean` близка к запрошенной плотности;
- все 20 run indices присутствуют для каждого режима;
- Baseline1 при одинаковых density и seed совпадает между AV-срезами;
- число `policy_left_requests + policy_right_requests + policy_stay_actions`
  согласовано с числом активных AV-actions;
- нет неожиданного взрыва `rule_rejected`.

### 6.2. Критерий эффективности

Для каждого сочетания `(rho, AV fraction)` сначала анализировать Graph-PPO
против `baseline2_matched`:

- `pv_speed_delta_mean > 0`;
- `pv_speed_delta_ci_low > 0`;
- `graph_ppo_speed_advantage = True`;
- win rate существенно выше 50%;
- `priority_time_loss_delta < 0`;
- рост скорости PV не сопровождается критическим падением фонового потока.

Затем сравнить Graph-PPO с `graph_noop` и `left_if_safe`. Если Graph-PPO не
превосходит эти абляции, результат нельзя интерпретировать как доказательство
обученной координационной стратегии: возможно, достаточно тривиального правила.

Baseline1 показывает естественный поток без уступания, а не является прямой
заменой matched Baseline2.

### 6.3. Интерпретация возможных исходов

- **Graph-PPO лучше Baseline2 и обеих абляций:** основной положительный
  результат.
- **Graph-PPO лучше Baseline2 только при AV=1.00:** свидетельство порога
  проникновения AV; отдельно анализировать разницу 0.95 → 1.00.
- **Graph-PPO примерно равен left-if-safe:** вероятна выученная тривиальная
  стратегия; требуется архитектурная/поведенческая абляция.
- **Graph-PPO примерно равен noop:** policy почти не использует управление.
- **Высокая скорость PV при большом падении background flow:** оптимизация
  достигла цели ценой потока; это нужно явно показать как ограничение reward.
- **Большой rejected rate:** проверить action masks, cooldown и локальную
  геометрию перестроений.
- **Training speed высока, deterministic test speed низка:** policy могла
  зависеть от stochastic exploration; основной результат всё равно должен
  основываться на deterministic test.

Если 95% bootstrap CI пересекает ноль, сначала увеличить `--runs` до 50 на
этом же test-протоколе и полностью пересчитать соответствующий AV-срез. Не
добавлять отдельные 30 строк к старому CSV вручную: новый запуск с `--runs 50`
воспроизводит первые 20 run indices и формирует цельный результат.

## 7. Этап F — зафиксировать и архивировать результат

После завершения evaluation сохранить:

```text
run_config.json
train_metrics.jsonl
best_checkpoint.txt
выбранный checkpoint
все selection metadata/CSV
все final-test metadata/CSV/PNG
текущий commit репозитория
версию Ray/RLlib и Python
```

Создать небольшой manifest:

```bash
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "best_checkpoint=$(cat "$RUN/best_checkpoint.txt")"
  python --version
  python -c 'import ray; print("ray=" + ray.__version__)'
} > "$RUN/evaluation_manifest.txt"
```

Не удалять остальные checkpoints до завершения анализа и подготовки статьи.
Training checkpoint с максимальным return не следует переименовывать в
`best`: термин `best` резервируется только за победителем validation-протокола.

## 8. Последовательность в одном списке

1. Проверить `run_config.json`, `train_metrics.jsonl` и наличие final
   checkpoint.
2. Выполнить restore smoke final checkpoint.
3. На validation seed `310001` оценить checkpoints 825, 875, 900 и final при
   `rho=0.30`, `AV=1.00`, 20 paired runs.
4. Выбрать checkpoint по скорости PV; при близком результате выполнить
   50-run tie-break на seed `320001`.
5. Зафиксировать путь в `best_checkpoint.txt`.
6. На новом test seed `410001` выполнить полную матрицу 4 densities × 3 AV
   fractions × 5 modes × 20 runs.
7. Проверить размерность CSV, конфигурацию metadata, воспроизводимость baseline
   и отсутствие некорректных метрик.
8. Проанализировать скорость PV, paired CI, win rate, time loss, фоновый поток
   и поведение actions.
9. Для неоднозначных условий расширить test до 50 runs.
10. Сохранить manifest, commit, конфигурацию, выбранный checkpoint и все
    evaluation-артефакты.

