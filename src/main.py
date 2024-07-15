'''Main module for running application'''

from modules.GUI import GUI
from modules.database import Database


def main():
    '''Main function'''
    GUI() #start app
    db = Database() #create database

if __name__ == '__main__':
    main()
