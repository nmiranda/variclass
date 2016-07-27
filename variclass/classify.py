import argparse
from features import FeatureData

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training', required=True)
    parser.add_argument('-s', '--test', required=True)
    args = parser.parse_args()

    training_data_store = FeatureData(args.training)
    training_data_features = this_store.get_features()

    test_data_store = FeatureData(args.test)
    test_data_features = test_data_store.get_features()

    import ipdb;ipdb.set_trace() 
    

if __name__=="__main__":
    main()
