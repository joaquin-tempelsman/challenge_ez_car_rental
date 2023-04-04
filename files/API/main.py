import sys
sys.path.insert(1,'src')

import logging
import click
from api import train, predict



@click.group()
def cli():
    pass


@click.command('train')
@click.option('-t', '--trials',
              required=False,
              help=('optimization trials for optimization'))

def run_train_cli(trials= 5):
    train(trials)
        


##-- match tables functionality --##
    # ['San Francisco', 210.2, '2017-10-01', '20:20']


@click.command('predict')
@click.option('-c', '--city',
              required=True)

@click.option('-d', '--duration',
              required=True)              

@click.option('-t', '--time',
              required=True)              

@click.option('-dt', '--date',
              required=True)
def run_predict_cli(city, duration, date, time):
    print(city,duration,date,time)
    predict(city, duration, date, time)
 

cli.add_command(run_train_cli)
cli.add_command(run_predict_cli)


if __name__ == '__main__':
    cli()
