# -*- coding: utf-8 -*-

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('FEATURES')
    args = parser.parse_args()

    features = pd.read_csv(args.FEATURES)
    n_features = 0
    
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores = list()
    fa_scores = list()

    n_components = range(0, n_features)
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    print("Best number of components by PCA CV = %d" % n_components_pca)
    print("Best number of components by FactorAnalysis CV = %d" % n_components_fa)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(n_components_pca, color='b', label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r', label='FactorAnalysis CV: %d' % n_components_fa, linestyle='--')
    plt.xlabel('Number of components')
    plt.ylabel('CV score')
    plt.legend(loc='best')
    plt.title('Component selection results')
    plt.show()
    
    

if __name__=="__main__":
    main()

    
