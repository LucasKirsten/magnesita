import os
import argparse
from src import DataLoader, Model

def main(args):
    
    # create path to save data
    os.makedirs(args.path_save, exist_ok=True)
    
    # handle data
    loader = DataLoader('train', args.path_save, args.path_raw_data, args.path_target_data, args.split_train_test)
    X,y = loader.get_data()
    
    # train model
    model = Model(input_shape=(X.shape[-1],), path_save=args.path_save)
    model.train(X, y, args.batch_size, args.epochs, args.verbose, args.n_splits)

if __name__=='__main__':
    
    # arguments
    parser = argparse.ArgumentParser(description='Train the model(s).')
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                       help="Batch size to be used during training.")
    parser.add_argument('-e', '--epochs', type=int, default=100,
                       help="Number of epochs that each model will be trained.")
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="Verbose level for the training.")
    parser.add_argument('-s', '--n_splits', type=int, default=10,
                       help="Number of splits in the KFold algorithm. The number of models to be trained is related to this value.")
    parser.add_argument('-pr', '--path_raw_data', type=str, required=True,
                        help="Path to the csv file containing the raw data.")
    parser.add_argument('-pt', '--path_target_data', type=str, required=True,
                        help="Path to the csv file containing the target data.")
    parser.add_argument('-ps', '--path_save', type=str, default='./saves',
                        help="Path to save created data during training.")
    parser.add_argument('-st', '--split_train_test', type=float, default=0.1,
                       help="Ratio to split raw data into training and testing. The split will be saved in the path_save.")
    
    args = parser.parse_args()
    print('\nArguments: ', str(args), end='\n')
    
    main(args)