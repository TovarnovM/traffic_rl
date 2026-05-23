import os
import signal
import subprocess
import sys
import time
from optuna.storages import get_storage
import sqlalchemy.exc



def main() -> None:
    db_url = os.environ["OPTUNA_DB_URL"]
    port_min = int(os.environ["OPTUNA_PROXY_PORT_MIN"])
    port_max = int(os.environ["OPTUNA_PROXY_PORT_MAX"])
    thread_pool_size = int(os.environ.get("OPTUNA_PROXY_THREAD_POOL_SIZE", "32"))

    for attempt in range(60):
        try:
            _ = get_storage(db_url)
            print("Optuna RDB schema is ready.")
            break
        except sqlalchemy.exc.IntegrityError as e:
            # типичная гонка CREATE TYPE/CREATE TABLE — подождать и повторить
            print(f"Schema init race, retrying... ({attempt+1}/60) {e}")
            time.sleep(1.0)
        except Exception as e:
            print(f"DB not ready yet, retrying... ({attempt+1}/60) {e}")
            time.sleep(1.0)
    else:
        raise RuntimeError("Failed to initialize Optuna storage schema.")


    procs: list[subprocess.Popen] = []


    def shutdown(signum, frame):
        print("Stopping gRPC proxies...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)


    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)


    print(f"Starting gRPC proxies on ports {port_min}..{port_max} with thread_pool_size={thread_pool_size}")


    for port in range(port_min, port_max + 1):
        code = f"""
import os
from concurrent.futures import ThreadPoolExecutor
from optuna.storages import get_storage, run_grpc_proxy_server


db_url = os.environ['OPTUNA_DB_URL']
thread_pool_size = int(os.environ.get('OPTUNA_PROXY_THREAD_POOL_SIZE', '32'))


storage = get_storage(db_url)
print('Starting gRPC proxy on 0.0.0.0:{port} (thread_pool_size=%d)' % thread_pool_size)


thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
run_grpc_proxy_server(storage, host='0.0.0.0', port={port}, thread_pool=thread_pool)
"""
        p = subprocess.Popen([sys.executable, "-c", code])
        procs.append(p)


    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        shutdown(None, None)




if __name__ == "__main__":
    main()