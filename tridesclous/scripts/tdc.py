"""
Very basic comand line for tridesclous


"""
import sys
import os
import argparse


import pyqtgraph as pg
import tridesclous as tdc

comand_list =[
    'mainwin',
    'makecatalogue',
    'runpeeler',
    'cataloguewin',
    'peelerwin',
    'init',
]
txt_command_list = ', '.join(comand_list)


def open_mainwindow():
        app = pg.mkQApp()
        win = tdc.MainWindow()
        win.show()
        app.exec_()            
    
def main():
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='tridesclous')
    parser.add_argument('command', help='command in [{}]'.format(txt_command_list), default='mainwin', nargs='?')
    
    parser.add_argument('-d', '--dirname', help='working directory', default=None)
    parser.add_argument('-c', '--chan_grp', type=int, help='channel group index', default=0)
    parser.add_argument('-p', '--parameters', help='JSON parameter file', default=None)
    
    
    args = parser.parse_args(argv)
    #~ print(sys.argv)
    #~ print(args)
    #~ print(args.command)
    
    command = args.command
    if not command in comand_list:
        print('command should be in [{}]'.format(txt_command_list))
        exit()
    
    dirname = args.dirname
    if dirname is None:
        dirname = os.getcwd()
    
    #~ print(command)
    
    if command in ['cataloguewin', 'peelerwin']:
        if not tdc.DataIO.check_initialized(dirname):
            print('{} is not initialized'.format(dirname))
            exit()
        dataio = tdc.DataIO(dirname=dirname)
        print(dataio)
    
    if command=='mainwin':
        open_mainwindow()
    
    elif command=='makecatalogue':
        pass
    
    elif command=='runpeeler':
        pass
        
    elif command=='cataloguewin':
        catalogueconstructor = tdc.CatalogueConstructor(dataio=dataio, chan_grp=args.chan_grp)
        app = pg.mkQApp()
        win = tdc.CatalogueWindow(catalogueconstructor)
        win.show()
        app.exec_()    
        
    elif command=='peelerwin':
        initial_catalogue = dataio.load_catalogue(chan_grp=args.chan_grp)
        app = pg.mkQApp()
        win = tdc.PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
        win.show()
        app.exec_()
    
    elif command=='init':
        app = pg.mkQApp()
        win = tdc.InitializeDatasetWindow()
        win.show()
        app.exec_()    
    
    

if __name__ =='__main__':
    open_mainwindow()

