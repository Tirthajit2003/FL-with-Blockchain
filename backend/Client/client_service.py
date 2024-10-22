import asyncio
import logging
import pickle
from concurrent.futures.process import ProcessPoolExecutor
import websockets
from FedHetero.ext_output import ext_output
from FedHetero.get_results import get_results
from FedHetero.optimize_extractor import optimize_extractor
from FedHetero.shuffle_dataset import shuffle_dataset
from client_process import process
from FedHetero.initializer import initialize
from FedHetero.rep_output import rep_output
from check_model import check_model
from data_config import get_data_config
import json
import argparse

task_executor = ProcessPoolExecutor(max_workers=3)


async def producer(websocket, message):
    log = logging.getLogger('producer')
    log.info('Received processed message')
    serialized_message = json.dumps(message)
    logging.debug('serial ' + str(serialized_message))
    try:
        await websocket.send(serialized_message)
    except Exception as e:
        logging.debug('producer exception catch ' + str(e))


async def listener(websocket, path, client_number):
    log = logging.getLogger(f'listener_{client_number}')
    log.info(f'Client {client_number} connected on path {path}')

    if path == '/process':
        async for message in websocket:
            print(f'Client {client_number} received message on /process')
            job_data = pickle.loads(message)
            loop = asyncio.get_running_loop()
            await process(job_data, websocket,client_number)
            print(f'Client {client_number} task done, closing connection')
            await websocket.close()

    if path == '/initialize_hetero':
        async for message in websocket:
            print(f'Client {client_number} received message on /initialize_hetero')
            job_data = pickle.loads(message)
            loop = asyncio.get_running_loop()
            await initialize(job_data, websocket)
            print(f'Client {client_number} task done, closing connection')
            await websocket.close()

    # The other paths follow the same structure
    if path == '/get_rep_output':
        async for message in websocket:
            print(f'Client {client_number} received message on /get_rep_output')
            job_data = pickle.loads(message)
            loop = asyncio.get_running_loop()
            await rep_output(job_data, websocket)
            print(f'Client {client_number} task done, closing connection')
            await websocket.close()

    if path == '/optimize_model':
        async for message in websocket:
            print(f'Client {client_number} received message on /optimize_model')
            job_data = pickle.loads(message)
            loop = asyncio.get_running_loop()
            await optimize_extractor(job_data, websocket)
            print(f'Client {client_number} task done, closing connection')
            await websocket.close()

    if path == '/get_model_weights':
        async for message in websocket:
            print(f'Client {client_number} received message on /get_model_weights')
            job_data = pickle.loads(message)
            loop = asyncio.get_running_loop()
            await ext_output(job_data, websocket)
            print(f'Client {client_number} task done, closing connection')
            await websocket.close()

    if path == '/get_results':
        async for message in websocket:
            print(f'Client {client_number} received message on /get_results')
            job_data = pickle.loads(message)
            loop = asyncio.get_running_loop()
            await get_results(job_data, websocket)
            print(f'Client {client_number} task done, closing connection')
            await websocket.close()

    if path == '/shuffle':
        async for message in websocket:
            print(f'Client {client_number} received message on /shuffle')
            job_data = pickle.loads(message)
            loop = asyncio.get_running_loop()
            await shuffle_dataset(job_data, websocket)
            print(f'Client {client_number} task done, closing connection')
            await websocket.close()

    if path == '/check_model':
        async for message in websocket:
            print(f'Client {client_number} received message on /check_model')
            job_data = pickle.loads(message)
            loop = asyncio.get_running_loop()
            await check_model(job_data, websocket,client_number)
            print(f'Client {client_number} task done, closing connection')
            await websocket.close()

    if path == '/get_data_config':
        async for message in websocket:
            print(f'Client {client_number} received message on /get_data_config')
            job_data = pickle.loads(message)
            loop = asyncio.get_running_loop()
            await get_data_config(job_data, websocket)
            print(f'Client {client_number} task done, closing connection')
            await websocket.close()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser("client")
        parser.add_argument("port", help="Define a valid port for the client to run on", type=int)
        parser.add_argument("client_number", help="Specify the client number", type=int)
        args = parser.parse_args()

        print(f'Client {args.client_number} running on port {args.port}')
        start_server = websockets.serve(
            lambda ws, path: listener(ws, path, args.client_number),
            "0.0.0.0", args.port, ping_timeout=None, max_size=None
        )
        loop = asyncio.get_event_loop()

        loop.run_until_complete(start_server)
        loop.run_forever()
    except Exception as e:
        print(f'Caught exception: {e}')
    finally:
        loop.close()
