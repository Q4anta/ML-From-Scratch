from common_math import gini, entropy, std_devn
import pandas as pd
import numpy as np
from sklearn import metrics
import numbers
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

eq_op = lambda x, y: x.eq(y)
le_op = lambda x, y: x.le(y)
class DecisionTree:
    def __init__(self, features, output, **kwargs):
        self.measure = kwargs.get('measure', gini)
        self.node_purity = self.measure(output)
        self.features = features
        self.output = output
        self.min_split_improvement = kwargs.get('min_split_improvement', 0)
        self.min_leaf_size = kwargs.get('min_leaf_size',100)
        self.max_depth = kwargs.get('max_depth', np.inf)
        self.kwargs = kwargs

    def fit(self, **kwargs):
        # Depth decreases down to zero with root being at highest depth.
        # Counter to standard notation!
        self.left_branch = None
        self.right_branch = None
        df_split = pd.DataFrame(columns = ['feature', 'split_point', 'split_purity','operator'])
        self.split_analysis = df_split
        self.best_split = None
        feature_frac = kwargs.get('feature_frac',1.0)
        # Make sure depth has a meaningful value.
        depth = kwargs.get('depth', self.max_depth)

        if(depth <= 0 or len(self.output) < 2*self.min_leaf_size): # No further splits.
            return None

        # Try splitting by each feature, and getting weighted metric of children to compare.
        for feature in self.features.sample(frac=1.0, axis=1).columns:
            unique_vals = self.features[feature].unique()
            all_numeric = all(isinstance(x, numbers.Real) for x in unique_vals)
            cmp_operator = le_op if all_numeric else eq_op

            for split_point in unique_vals: 
                left_split_index = cmp_operator(self.features[feature], split_point)

                left_split = self.output[left_split_index]
                right_split = self.output[~left_split_index]

                if(len(left_split) < self.min_leaf_size or len(right_split) < self.min_leaf_size):
                    continue
                weight_left = len(left_split) / (len(left_split) + len(right_split))
                left_purity = self.measure(left_split)
                right_purity = self.measure(right_split)
                split_purity = weight_left*left_purity + (1-weight_left)*right_purity

                df_split = df_split.append({'feature' : feature, 'split_point' : split_point, 'split_purity' : split_purity, 'operator' : cmp_operator}, ignore_index=True)
            # Succesfully got split impact for target fraction of features
            if(df_split['feature'].nunique() >= len(self.features.columns) * feature_frac): 
                break

        self.split_analysis = df_split
        # See if any split is better, and if so make the split and recurse.
        df_split.sort_values(by='split_purity', ascending=True, inplace=True)
        if(len(df_split) == 0):
            return None
        best_split = df_split.iloc[0,:]
        if(best_split['split_purity'] < self.node_purity - self.min_split_improvement): # split improves current purity
            self.best_split = best_split
            split_feature = best_split['feature']
            split_op = best_split['operator']
            split_point = best_split['split_point']
            left_split_index = split_op(self.features[split_feature], split_point)
        
            # Create left and right leaves.
            self.left_branch = DecisionTree(
                features=self.features[left_split_index], 
                output=self.output[left_split_index],
                **self.kwargs
                )
            self.right_branch = DecisionTree(
                features=self.features[~left_split_index],
                output=self.output[~left_split_index],
                **self.kwargs
            )

            # Fit left and right branches with a lower depth.
            self.left_branch.fit(depth=depth-1, feature_frac=feature_frac)
            self.right_branch.fit(depth=depth-1, feature_frac=feature_frac)

    def printTree(self, depth=0):
        # This uses the standard depth notation.
        print("=="*depth)
        print("    "*depth + "Purity: {}".format(self.node_purity))
        if(self.best_split is not None):
            print("    "*depth + "Split on: " + self.best_split['feature'] + " @ {}".format(self.best_split['split_point']) + "; # {}".format(len(self.features)))
            self.left_branch.printTree(depth + 1)
            self.right_branch.printTree(depth + 1)

    def predict(self, features, method='mode'):
        # If there is no further split, return the prediction for this node. Otherwise, continue going down the tree.
        if(self.best_split is None):
            if(method == 'mode'):
                predicted_value = self.output.dropna().mode().iloc[0]
            elif(method == 'average'):
                predicted_value = self.output.dropna().mean().iloc[0]
            elif(method == 'median'):
                predicted_value = self.output.dropna().median().iloc[0]
            
            output = pd.DataFrame(index = features.index)
            output[self.output.name] = predicted_value
            output = output.iloc[:,0]
            output.name = self.output.name
            return output
        else:
            left_index=self.best_split['operator'](features[self.best_split['feature']], self.best_split['split_point'])
            left_predict = self.left_branch.predict(features[left_index])
            right_predict = self.right_branch.predict(features[~left_index])
            return pd.concat([left_predict,right_predict])


            
if __name__ == '__main__':
    sim_case = 'iris'
    sim_type = 'classifier'
    # Titanic example.
    if(sim_case == 'titanic'):
        train = pd.read_csv('~/Projects/FromScratch/Datasets/titanic/train.csv', sep=',')
        features = train.drop(['Survived', 'PassengerId','Name'], axis = 1)
        output = train['Survived']
    elif(sim_case == 'test_dummy'):
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
        
    # Build the decision tree
    if(sim_type == 'classifier'):
        max_depth = 10
        DT = DecisionTree(features=features, output=output, measure=gini, min_split_improvement = 0.0, max_depth=max_depth, min_leaf_size=2)
        DT.fit()
        DT.printTree()
        
        skTree = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
        skTree.fit(features, output)
        tree.plot_tree(skTree)
        
    in_sample_predict = DT.predict(features=features)
    confusion_matrix = metrics.confusion_matrix(output, in_sample_predict)
    print(confusion_matrix)
    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    #cm_display.plot()
    #plt.show() 