import sys
import time


def stampa(args):
    nome = args.nome
    messaggio = args.messaggio

    print(f'{messaggio}, {nome}, usa pure il terminale', file=sys.__stdout__)

    print(f"{nome} → Apertura file di log")

    for i in range(150):
        print(i+1)
        time.sleep(0.1)

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

