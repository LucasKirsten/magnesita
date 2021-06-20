import os
import argparse
from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_squared_error
from src import DataLoader, Model

def main(args):
    
    # handle data
    loader = DataLoader('test', args.path_save, args.path_raw_data, args.path_target_data)
    X,y = loader.get_data()
    
    # load model
    model = Model(X.shape[-1], args.path_save)
    model.load_models(args.path_save)
    
    # test model
    pred = model.predict(X)
    
    # save predictions in csv file
    loader.df['pred'] = loader.pred_denorm(pred)
    loader.df.to_csv(os.path.join(args.path_save, 'predicted.csv'), index=False)
    
    # if target is provided
    if y is not None:
        print('\nR2 score: ', r2_score(y, pred))
        print('Max error: ', max_error(y, pred))
        print('Mean absolute error: ', mean_absolute_error(y, pred))
        print('Mean squared error: ', mean_squared_error(y, pred))

if __name__=='__main__':
    
    # arguments
    parser = argparse.ArgumentParser(description='Train the model(s).')
    parser.add_argument('-pr', '--path_raw_data', type=str, required=True,
                        help="Path to the csv file containing the raw data.")
    parser.add_argument('-pt', '--path_target_data', type=str, default=None,
                        help="Path to the csv file containing the target data.")
    parser.add_argument('-ps', '--path_save', type=str, default='./saves',
                        help="Path to load created data during training.")
    
    args = parser.parse_args()
    print('\nArguments: ', str(args), end='\n')
    
    main(args)