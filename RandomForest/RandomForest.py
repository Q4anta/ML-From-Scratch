from DecisionTree import DecisionTree as DT
from common_math import gini, entropy, std_devn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics

class RandomForest:
    def __init__(self, **kwargs):
        self.number_trees = kwargs.get('number_trees',100)
        self.bootstrap_frac = kwargs.get('bootstrap_frac', 1.0)
        self.forest = []
        self.kwargs = kwargs

    def fit(self):
        # Fit a forest one tree at a time.
        self.forest = []
        params = self.kwargs.copy()
        orig_features = params.get('features')
        orig_output = params.get('output')
        params.pop('features')
        params.pop('output')

        for tree_index in range(self.number_trees):
            # Bootstrap
            features = orig_features.sample(frac=self.bootstrap_frac, replace=True)
            output = orig_output.loc[features.index]

            # Construct the tree.
            tree_n = DT(features=features, output=output, **params)
            tree_n.fit()
            self.forest.append(tree_n)
            if(tree_index % 10 == 0): print('='*int(tree_index/10))

    def predict(self, features, tree_method='mode', forest_agg = 'mode'):
        predictions = []
        index = 0
        for tree in self.forest:
            index = index + 1
            pred = tree.predict(features, method=tree_method)
            pred.name = pred.name + '_' + str(index)
            predictions.append(pred)
        self.aggregate_pred = pd.concat(predictions, axis=1)
        final_pred = self.aggregate_pred.mean(axis=1)
        
        if(forest_agg == 'mode'):
            final_pred = self.aggregate_pred.mode(axis=1)
        elif(forest_agg == 'average'):
            final_pred = self.aggregate_pred.mean(axis=1)
        elif(forest_agg == 'median'):
            final_pred = self.aggregate_pred.median(axis=1)
        
        final_pred.name = '_'.join(pred.name.split('_')[0:-1])
        return final_pred


if __name__ == '__main__':
    sim_case = 'iris'
    sim_type = 'classifier'
    # Titanic example.

    if(sim_case == 'test_dummy'):
        samples = 2000
        train = pd.DataFrame(columns = ['Gender', 'Age', 'Employed'])
        train['Age'] = np.random.randint(18, 66, size=samples)
        train['Gender'] = np.random.randint(2, size=samples)
        
        # Option 1: Likelihood of employment increases with age, higher for men.
        train['Employed'] = np.random.randint(90, size=samples)
        train['Employed'] = -train['Employed'] + train['Age'] + train['Gender'] * 30
        train['Employed'] = train['Employed'].apply(lambda x: 1 if x > 0 else 0)
        
        # Option 2: Clean data - Men > 40 employed, W < 35 employed.
        train['Employed'] = 0
        train.loc[(train.Gender>0.5)&(train.Age>40),'Employed'] = 1
        train.loc[(train.Gender<0.5)&(train.Age<35),'Employed'] = 1
        
        # Convert gender to categorical
        train['Gender'] = train['Gender'].apply(lambda x: 'F' if x < 0.5 else 'M')
        features = train[['Gender','Age']]
        output = train['Employed']
    elif(sim_case == 'iris'):
        data = load_iris()
        features = data.data
        output = data.target
        features, features_test, output, output_test = train_test_split(data.data,data.target, random_state = 50, test_size = 0.25)
        features = pd.DataFrame(data = features, columns = ['sl','sw','pl','pw'])
        output = pd.Series(data=output)
        output.name = 'Target'

    # Build the decision tree
    if(sim_type == 'classifier'):
        max_depth = 10
        number_trees = 10
        RF = RandomForest(features=features, output=output, measure=gini, 
                          min_split_improvement = 0.0, max_depth=max_depth, number_trees=number_trees,
                          min_leaf_size=1, bootstrap_frac=0.5)
        RF.fit()
        in_sample_pred = RF.predict(features)
        
        confusion_matrix = metrics.confusion_matrix(output, in_sample_predict)
        print(confusion_matrix)
