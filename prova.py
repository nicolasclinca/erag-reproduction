import sys
import os
import time
import datetime

instant = datetime.datetime.now().strftime("%d:%m_%H:%M:%S")
log_path = f"../logs/prova_{instant}.log"  # log file creation
os.makedirs(os.path.dirname(log_path), exist_ok=True)  # directory creation, if not existing
log_file = open(log_path, "a")  # log file creation
sys.stdout = log_file  # set the log file as standard output for print functions
sys.stderr = log_file


def stampa(args):
    nome = args.nome
    messaggio = args.messaggio

    print(f'{messaggio}, {nome}, usa pure il terminale', file=sys.__stdout__)

    print(f"{nome} → Apertura file di log")

    time.sleep(3)

    print(f"{nome} → Chiusura file di log")

    print(f"Operazione conclusa per {nome}", file=sys.__stdout__)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Programma di prova')

    parser.add_argument('-n', '--nome', type=str,
                        default='Utente',
                        help="Specificare il nome dell'utente")
    parser.add_argument('-m', '--messaggio', type=str,
                        default='Salve',
                        help="Specificare il messaggio di apertura")

    args = parser.parse_args()

    stampa(args)  # args.nome, args.apertura

    log_file.close()
