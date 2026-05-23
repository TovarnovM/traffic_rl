import os
import time
import optuna
from optuna.storages import GrpcStorageProxy


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_int("y", -10, 10)
    z = trial.suggest_categorical("z", ["a", "b", "c"])
    # простая детерминированная функция
    penalty = 0.0 if z == "a" else 1.0
    return (x - 1.23) ** 2 + (y + 3) ** 2 + penalty


def main() -> None:

    study_name = os.getenv("OPTUNA_STUDY_NAME", "grpc_smoke_test")

    host = os.getenv("OPTUNA_GRPC_HOST", "optuna-lb")
    port = int(os.getenv("OPTUNA_GRPC_PORT", "13000"))
    storage = GrpcStorageProxy(host=host, port=port)

    # создаём/открываем study

    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except Exception:
        pass
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=50, n_jobs=4)  # n_jobs -> параллельно потоками
    dt = time.time() - t0

    trials = study.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state.is_finished()]

    print(f"Study: {study.study_name}")
    print(f"Total trials: {len(trials)}")
    print(f"Finished trials: {len(complete)}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Time: {dt:.2f}s")

    # простая проверка, что реально что-то посчиталось
    assert len(complete) >= 40, "Too few finished trials, something is wrong"
    assert study.best_value < 5.0, "Best value is unexpectedly bad (proxy/LB/db issue?)"

    print("OK: gRPC proxy + LB + DB stack seems working.")


if __name__ == "__main__":
    main()


