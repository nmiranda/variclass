import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('curve')
    args = parser.parse_args()

    this_curve = pd.read_csv(args.curve)['0']
    
    plt.plot(this_curve.index, this_curve, 'b*')
    plt.xlabel('days')
    plt.ylabel('mag')
    plt.show()
    #plt.savefig('lel.png')
    
if __name__ == '__main__':
    main()
