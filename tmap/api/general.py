import pandas as pd
import csv,os
import pickle
import random
import string

def logger(*args,verbose=0):
    if verbose !=0:
        print(' '.join([str(_) for _ in args]))
    else:
        pass

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(stringLength))

def data_parser(path, ft='csv',verbose=1,**kwargs):
    if type(path) != str:
        df = path.copy()
        if type(df) != pd.DataFrame:
            df = pd.DataFrame(df)
        logger('Input data is a matrix: ', verbose=verbose)
    else:
        if ft == 'csv':
            sniffer = csv.Sniffer()
            sniffer.preferred = [',', '\t', '|']
            dialect = sniffer.sniff(open(path,'r').readline().strip('\n'))
            df = pd.read_csv(path, sep=dialect.delimiter,index_col=0, header=0, **kwargs)
        else:
            df = pd.read_excel(path, index_col=0, header=0, **kwargs)
        logger('Input data path: ', path,verbose=verbose)
    logger('Shape of Input data: ', df.shape,verbose=verbose)
    logger("Focus, this data means %s samples, and %s features. " % (str(df.shape[0]),
                                                                            str(df.shape[1])),verbose=verbose)
    return df

def preprocess_metadata_beta():

    # divide numeral cols and categorical cols

    # fill nan numeral cols

    # one hot / get dummy categorical cols

    # output
    pass

def write_data(data,prefix,suffix='',mode='df',verbose=1, **kwargs):
    if mode =='df':
        data.to_csv('_'.join([prefix,suffix])+'.csv',sep=',',index=1)
        logger("Data with prefix %s has been output." % prefix, verbose=verbose)
    elif mode == 'multidf':
        cols = kwargs['df2cols']
        logger("There are multiple data matrixs need to output. Inclduing", '\n'.join([os.path.basename(_) for _ in cols.keys()]),verbose=verbose)
        for name,col in cols.items():
            subdata = data.loc[:,col]
            subdata.to_csv(prefix +'_%s.csv' % '_'.join([name,suffix]) ,sep=',',index=1)
        logger("Data with prefix %s has been output." % prefix, verbose=verbose)
    elif mode == 'html':
        pass
