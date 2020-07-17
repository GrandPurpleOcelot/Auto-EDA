# Core libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import datetime

# Missing handling
import missingno as msno
from sklearn.impute import SimpleImputer

# logging
import logging
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# CA
import prince

# modeling
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn import metrics
from matplotlib import colors

# set seaborn theme
sns.set()

#-----------------HELPER FUNCTIONS---------------
def datetime_validate(text):
    matched_format = 0
    for date_format in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%m-%d-%Y','%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d %H:%M:%S.%f', '%H:%M:%S', '%H:%M', '%Y/%m/%d %H:%M:%S',
                        '%Y/%m/%d %H:%M:%S.%f', '%m-%d-%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
                        '%m-%d-%Y %H:%M:%S.%f', '%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S']:
        try:
            datetime.datetime.strptime(text, date_format)
            matched_format += 1
        except ValueError:
            pass
    return matched_format > 0

def numeric_validate(text):
    return text.replace('.','',1).replace('-','',1).isdigit()

def corrfunc(x,y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    # Unicode for lowercase rho (ρ)
    rho = '\u03C1'
    ax.annotate(f'{rho} = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
    

def visualize_components(score, coeff, labels=None):
    # code modified from: https://github.com/ostwalprasad/ostwalprasad.github.io/blob/master/jupyterbooks/2019-01-20-PCA%20using%20python.ipynb
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    
    plt.figure(figsize = (10, 10))
    pca_plot = sns.scatterplot(x = xs * scalex, y = ys * scaley, s=20)
    pca_plot.axes.set_title("PCA Biplot", fontsize=20)

    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5, shape = 'full', width = 0.003)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    
    txt = '''Hint: Biplot contains two plots. 
    Scatter plot (blue points) which shows first two principal components.
    Loading vectors (red arrows) explains how much weight they have on that component, 
    angles between individual vectors tells about correlation between them.'''
    plt.figtext(0.5, 0.01, txt, wrap=True, fontsize=12, horizontalalignment='center')
    
    plt.show()
    
def feature_importance_plot(features = None, importances = None):
    features = features
    importances = importances
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

NUMERICS_TYPES = ['int', 'float', 'int32', 'float32', 'int64', 'float64']
CATEGORICAL_TYPES = [np.object]

def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]

class auto_eda():
    def __init__(self, df, target_variable = None):
        self.df = df.copy()
        self.target_variable = target_variable
        self.num_variables = self.df.shape[0]
        self.num_obs = self.df.shape[1]
        self.memory_usage = str(self.df.memory_usage(deep=True).sum() / 1000000) + ' Mb' # memory in mb
        self.numeric_cols = self.df.select_dtypes(include = NUMERICS_TYPES).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include = CATEGORICAL_TYPES).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include = np.datetime64).columns.tolist()
        self.high_cardinality_cols = None
        self.other_cols = self.df.select_dtypes(exclude = NUMERICS_TYPES + CATEGORICAL_TYPES).columns.tolist()
        self.encoder = None
        
    def get_samples(self, n = 3):
        if len(self.df) < n*3:
            return(self.df)
        head = self.df.head(n = n)
        tail = self.df.tail(n = n)
        random = self.df.sample(n = n, random_state = 42)
        samples = pd.concat([head, random, tail])
        return samples

    def get_overview(self):
        overview = {}
        overview['Number of Variables:'] = self.num_variables
        overview['Number of Observations:'] = self.num_obs
        overview['Memory Usage:'] = self.memory_usage
        for k,v in overview.items():
            print(k,v)
    
    def get_missings(self, missing_tag = None):
        '''
        Sometimes missing values are denoted with a number or string, 
        enter the missing tag to replace them with NAs
        '''
        if missing_tag is not None:
            self.df.replace(missing_tag, np.nan, inplace = True)
        
        # check if there are any null values
        if self.df.isnull().sum().sum() == 0:
            print('''There is no missing value, please check if the missings have been encoded with non-NAN value.
Use argument missing_tag for encoded missing values''')
        else:
            # missing heatmap display the missing values position in the dataset
            missing_heatmap = plt.figure(1)
            msno.matrix(self.df)
            plt.title('Missing Values shown in White',fontsize=25)

            # correlation plot: how strongly the presence or absence of one variable affects the presence of another
            correlation_plot = plt.figure(2)
            msno.heatmap(self.df,cbar= False)
            plt.title('Missing Values Correlation',fontsize=25)

            # The dendrogram uses a hierarchical clustering algorithm 
            # to bin variables against one another by their missing values correlation 
            missing_dendogram = plt.figure(3)
            msno.dendrogram(self.df)
            plt.title('Missing Values Dendrogram',fontsize=25)
        
    def handle_missings(self, strategy = None, drop_threshold = 0.7):
        '''
        PLEASE RUN get_missings() FIRST TO IDENTIFY MISSINGS.
        
        3 Strategies:
        
        'deletion': drop variables with > 70% missing (or a different threshold using argument 'drop_threshold') and remove observations that contain at least 1 missing value.
        
        'encode'(Encoding imputation): for numerical variable, encoding missing entries as -999. For categorical variable, encoding missing entries as string "unknown"
        
        'mean_mode'(Mean/mode imputation): for numerial variable, impute the missing entries with the mean. For categorical variable, impute the missing entries with the mode
        
        '''
        strategies = ['deletion', 'encode', 'mean_mode']
        if self.df.isnull().sum().sum() == 0:
            print('There is no missing value in the dataset')
        elif strategy not in strategies:
            print('No strategy selected, please specify one of the following deletion, encode, or mean_mode')
        else:
            if strategy == 'deletion':
                # drop column if missing more than threshold (default: > 70% missing)
                fraction_missing = self.df.isnull().sum() / len(self.df)
                drop_list = fraction_missing[fraction_missing > drop_threshold].index.tolist()
                self.df.drop(drop_list, axis = 1, inplace = True) 
                
                self.numeric_cols = self.df.select_dtypes(include = NUMERICS_TYPES).columns.tolist()
                self.cat_cols = self.df.select_dtypes(include = CATEGORICAL_TYPES).columns.tolist()
                
                # drop row contains 1 or more missing values
                drop_row_count = self.df.shape[0] - self.df.dropna().shape[0]
                rows_percentage = round(drop_row_count / self.df.shape[0] * 100, 1)
                self.df.dropna(inplace=True)
                
                print('Dropped columns: {}\nNumber of dropped rows: {} --> {}% of rows removed'.format(drop_list, drop_row_count, rows_percentage))
            
            elif strategy == 'encode':
                # encoding missing numerics as -999, categories as 'unknown'
                numerics_replaced = (len(self.df[self.numeric_cols]) - self.df[self.numeric_cols].count()).sum()
                self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(-999)
                
                cats_replaced = (len(self.df[self.cat_cols]) - self.df[self.cat_cols].count()).sum()
                self.df[self.cat_cols] = self.df[self.cat_cols].fillna('unknown')
                
                print('Count of encoded numerical values: {}\nCount of encoded categorical values: {}'.format(numerics_replaced, cats_replaced))
                
            elif strategy == 'mean_mode':
                # impute missing numerics with mean value
                numerics_replaced = (len(self.df[self.numeric_cols]) - self.df[self.numeric_cols].count()).sum()
                self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(self.df[self.numeric_cols].mean())
                
                # impute missing categories with mode value
                cats_replaced = (len(self.df[self.cat_cols]) - self.df[self.cat_cols].count()).sum()
                self.df[self.cat_cols] = self.df[self.cat_cols].fillna(self.df[self.cat_cols].mode().iloc[0])
                
                print('Count of imputed numerical values: {}\nCount of imputed categorical values: {}'.format(numerics_replaced, cats_replaced))
    
    def check_data_type(self):
        column_series = self.df.dtypes.index
        type_series = self.df.dtypes.values
        cardinality = self.df.apply(pd.Series.nunique)
        
        frame = {'Column': column_series, 'Type': type_series, 'Cardinality': cardinality }
        data_types = pd.DataFrame(frame).reset_index(drop=True)
                
        high_cardinality_condition = (data_types['Cardinality'] > len(self.df)//2)
        low_cardinality_condition = (data_types['Cardinality'] < 13)
        no_cardinality_condition = (data_types['Cardinality'] == 1)
        
        # check cardinality
        data_types['Warning'] = np.where(no_cardinality_condition, 'no_cardinality',
                                         (np.where(low_cardinality_condition, 'low_cardinality', 
                                          np.where(high_cardinality_condition, 'high_cardinality', 'None'))))
        
        # check for date string
        data_types['Is_datetime'] = np.where(self.df.iloc[0].apply(str).apply(datetime_validate), 'yes', 'no')
        
        # check for string numbers
        data_types['String_number'] = np.where(self.df.iloc[0].apply(str).apply(numeric_validate), 'yes', 'no')
        
        # suggest better data type
        data_types['Suggest'] = np.where((data_types['Is_datetime'] == 'yes') & (data_types['Type'] == 'object'), 'converts to datetime',
                                         (np.where((data_types['Warning'] == 'low_cardinality') & (data_types['Type'].apply(is_numeric_dtype)), 'converts to object',
                                          np.where((data_types['String_number'] == 'yes') & (data_types['Warning'] == 'None') & (data_types['Type'] == 'object'), 'converts to numeric', 'None'))))
        
        return data_types
                
    def change_data_type(self, alter_columns = 'all'):
        '''
        Change the data type according to the sugesstions in check_data_type()
        '''
        type_table = self.check_data_type()
        conversion_commands = {'converts to datetime': (lambda x: pd.to_datetime(x)),
                               'converts to numeric': (lambda x: pd.to_numeric(x)),
                              'converts to object': (lambda x: str(x))}
        if alter_columns == 'all':
            columns_to_change = type_table[type_table['Suggest'] != 'None'][['Column', 'Suggest']]
            for index, row in columns_to_change.iterrows():
                try:
                    self.df[row['Column']] = self.df[row['Column']].apply(conversion_commands[row['Suggest']])
                    print('Column {} {}'.format(row['Column'],row['Suggest']))
                except ValueError:
                    print('Column {} failed to {}. There is more than 1 type of data in this column'.format(row['Column'],row['Suggest']))

            self.high_cardinality_cols = type_table[(type_table['Cardinality'] > 30) & (type_table['Type'] == 'object')]['Column'].tolist()   
            self.numeric_cols = self.df.select_dtypes(include = NUMERICS_TYPES).columns.tolist()
            # remove high_cardinality_cols from cat_cols
            self.cat_cols = self.df.select_dtypes(include = CATEGORICAL_TYPES).columns.tolist()
            self.cat_cols = [i for i in self.cat_cols if i not in self.high_cardinality_cols]
            self.datetime_cols = self.df.select_dtypes(include = np.datetime64).columns.tolist()
            
        else:
            print('No column type changed')
            return None
        
    def histogram(self, kde = False):
        num_plots = len(self.numeric_cols)
        total_cols = 2
        total_rows = int(np.ceil(num_plots/total_cols)) 
        fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                                figsize=(7*total_cols, 7*total_rows), constrained_layout=True, squeeze=False)
        fig.suptitle('Histograms of Numerical Variables', fontsize = 20)
        for i, col in enumerate(self.numeric_cols):
            row = i//total_cols
            pos = i % total_cols
            p = sns.distplot(self.df[col],ax=axs[row][pos], kde = kde)
            p.set(title = col)
            p.set(xlabel=None)
            
    def count_plots(self):
        num_plots = len(self.cat_cols)
        if num_plots == 1:
            cplot = sns.countplot(self.df[self.cat_cols[0]])
            for p in cplot.patches:
                    cplot.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        else:
            total_cols = 2
            total_rows = int(np.ceil(num_plots/total_cols))
            fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                                    figsize=(7*total_cols, 7*total_rows), constrained_layout=True, squeeze=False)
            fig.suptitle('Frequency Plot of Categorical Variables', fontsize = 20)
            for i, col in enumerate(self.cat_cols):
                row = i//total_cols
                pos = i % total_cols
                cplot = sns.countplot(self.df[col],ax=axs[row][pos]) # cplot = countplot
                cplot.set(title = col)
                cplot.set(xlabel=None)
                cplot.set(yticklabels=[])
                plt.setp(axs[row][pos].get_xticklabels(), rotation=30)
                
                for p in cplot.patches:
                    cplot.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

            if len(self.high_cardinality_cols) > 0:
                print('Some categorical columns have high cardinality: {}'.format(self.high_cardinality_cols)) 
                print('Consider a different visualization method for these columns.')
                
    def word_cloud(self):
        pass
    
    def correlation(self, target= None, show_all = True):
        '''
        Arguments:
        
        target: specify a categorcal column for color grouping
        
        show_all: display all plots
        
        Generate plots for correlation analysis. Only works with numerical columns.
        If number of columns is small (less than 10), pairplot is selected. Pearson correlation notated on the upper-left corner.
        Else if number of columns is > 10, Pearson Corr Heatmap is selected.
        '''
        if target is None:
            target = self.target_variable
        
        num_cols = len(self.numeric_cols)
        if num_cols < 2:
                print("Correlation plots requires at least 2 numerical variables")
        else:
            if num_cols > 10 or show_all == True:
                corr = self.df[self.numeric_cols].corr()
                cmap=sns.diverging_palette(5, 250, as_cmap=True)

                # style_table = corr.style.background_gradient(cmap = cmap, axis=1)\
                #     .set_properties(**{'max-width': '60px', 'font-size': '10pt'})\
                #     .set_precision(2)

                even_range = np.max([np.abs(corr.values.min()), np.abs(corr.values.max())])
                style_table = corr.style.apply(background_gradient,
                                cmap=cmap,
                                m=-even_range,
                                M=even_range).set_precision(2)
                
                display(style_table)
                
                cluster_plot = sns.clustermap(corr, cmap = cmap, vmin=-1, vmax=1)
                cluster_plot.fig.suptitle("Hierarchical Structure in Correlation Matrix", y=1.05, fontsize=20)
                txt = '''
                
                Hint: Similarly correlated variables are grouped together (increase/decrease together).'''
                plt.figtext(0.5, 0.01, txt, wrap=True, fontsize=12, horizontalalignment='center')
                
            if num_cols <= 10 or show_all == True:    
                pplot = sns.pairplot(self.df[self.numeric_cols], corner=True, diag_kind = 'kde') # pplot = pairplot
                pplot.map_lower(corrfunc)
                pplot.fig.suptitle("Pearson Correlation Matrix", y=1.05, fontsize=20)

                if target != None and self.df[target].dtype in CATEGORICAL_TYPES:
                    num_cols_with_target = self.numeric_cols + [target]
                    pplot2 = sns.pairplot(self.df[num_cols_with_target], corner=True, hue = target)
                    pplot2.fig.suptitle("Grouped by: " + target, y=1.05, fontsize=20)
    
    def pca(self):
        '''
        input: dataframe of numerical columns
        output: 2 PCA components (PC1 and PC2)
        
        Principal component analysis (PCA) extracts 2 dimensional set of features from a high dimensional data set.
        Angles between individual vectors tells about correlation between them
        '''
        numeric_cols = self.numeric_cols
        if len(numeric_cols) < 3:
            print('PCA requires at least 3 numerical variables')
        else:
            scaled_df = StandardScaler().fit_transform(self.df[numeric_cols])
            scaled_df = pd.DataFrame(scaled_df, columns=numeric_cols)

            pcamodel = PCA(n_components=5)
            pca = pcamodel.fit_transform(scaled_df)

            # create PCA scatter plot of the first two components
            # draw loading vectors 
            visualize_components(pca[:,0:2], coeff = np.transpose(pcamodel.components_[0:2, :]), labels = numeric_cols)
    
    
    def boxplots(self, target = None):
        if target is None:
            target = self.target_variable
        if target is None:
            print("Please specify a categorical column as x-axis using 'target' argument.")
            return
        if self.df[target].dtype not in CATEGORICAL_TYPES:
            print("Target must be a categorical column.")
        else:
            num_plots = len(self.numeric_cols)
            total_cols = 2
            total_rows = int(np.ceil(num_plots/total_cols))
            fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                                    figsize=(7*total_cols, 5*total_rows), constrained_layout=True, squeeze=False)
            fig.suptitle('Boxplots', fontsize = 20)
            for i, col in enumerate(self.numeric_cols):
                row = i//total_cols
                pos = i % total_cols
                bplot = sns.boxplot(x = self.df[target], y = self.df[col], ax=axs[row][pos])
                bplot.set(title = col + ' vs. ' + target)
                bplot.set_xlabel('')
                
    def cat_plots(self, target = None):
        if target is None:
            target = self.target_variable
        if target == None:
            print("Please specify a categorical column as x-axis using 'target' argument")
            return
        if self.df[target].dtype not in CATEGORICAL_TYPES:
            print("Target must be a categorical column.")
        else:
            num_plots = len(self.cat_cols) - 1
            if num_plots == 0:
                print("Categorical plot requires at least 2 numerical variables")
            else:
                total_cols = 2
                total_rows = int(np.ceil(num_plots/total_cols))
                fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                                        figsize=(7*total_cols, 5*total_rows), constrained_layout=True, squeeze=False)
                fig.suptitle('Categorical Frequency Plots of: ' + target, fontsize = 20)
                # exclude target variable
                non_target_list = [i for i in self.cat_cols if i != target]
                for i, col in enumerate(non_target_list):
                    row = i//total_cols
                    pos = i % total_cols

                    x = target
                    y = col
                    temp_df = self.df[self.cat_cols].groupby(x)[y].value_counts(normalize=True)
                    temp_df = temp_df.mul(100)
                    temp_df = temp_df.rename('percent').reset_index()

                    catplot = sns.barplot(x = x, y='percent', hue=y, data=temp_df, ax=axs[row][pos])
                    catplot.set(title = col + ' vs. ' + target)
                    catplot.set_ylim(0,100)
                    catplot.set_ylabel('')
                    catplot.set(yticklabels=[])
                    catplot.set(xlabel=None)

                    for p in catplot.patches:
                        txt = str(p.get_height().round(1)) + '%'
                        txt_x = p.get_x() 
                        txt_y = p.get_height()

                        # disable logging warning where percentage is na 
                        logging.disable(logging.WARNING)
                        catplot.text(txt_x, txt_y, txt)
                    
    def correspondence_analysis(self, target = None):
        if target is None:
            target = self.target_variable
        if target == None:
            print("Please specify a categorical column as x-axis using 'target' argument")
            return
        if self.df[target].dtype not in CATEGORICAL_TYPES:
            print("Target must be a categorical column.")
        else:
            num_plots = len(self.cat_cols) - 1
            if num_plots == 0:
                print("Correspondence Analysis requires at least 2 numerical variables")
            else:
                total_cols = 2
                total_rows = int(np.ceil(num_plots/total_cols))
                fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                                        figsize=(7*total_cols, 7*total_rows), constrained_layout=True, squeeze=False)
                fig.suptitle('Correspondence Analysis for column: ' + target, fontsize = 20)
                # exclude target variable
                non_target_list = [i for i in self.cat_cols if i != target]           
                for i, col in enumerate(non_target_list):
                    row = i//total_cols
                    pos = i % total_cols
                    X = self.df.copy()
                    X = pd.crosstab(X[col], X[target])

                    ca = prince.CA(
                         n_components=2,
                         n_iter=3,
                         copy=True,
                         check_input=True,
                         engine='auto',
                         random_state=42)


                    ca = ca.fit(X)

                    ax = ca.plot_coordinates(
                         X=X,
                         ax=axs[row][pos],
                         x_component=0,
                         y_component=1,
                         show_row_labels=True,
                         show_col_labels=True)
                    
    def timeseries_plots(self, grouper = 'W'):
        if len(self.datetime_cols) == 0:
            print('No datetime column detected. Make sure datetime64 type exist.')
        else:
            for i in range(len(self.datetime_cols)):
                grouped_timeseries = self.df.resample(grouper, on = self.datetime_cols[i]).mean().dropna()

                fig = go.Figure()
                # add trend plot for each numerical column
                for col in self.numeric_cols:
                    fig.add_trace(go.Scatter(x = grouped_timeseries.index, y = grouped_timeseries[col],
                                        mode = 'lines+markers',
                                        name = col))

                # Edit the layout
                fig.update_layout(title='Time Series plot of Numerical Variables')

                fig.show()

    def tree_model(self, target = None, max_depth = 3, target_class_names = None):
        if target is None:
            target = self.target_variable
        if target == None:
            print("Please specify a categorical column as x-axis using 'target' argument")
        else:
            encoded_df = self.df.copy()
            encoder_dict = defaultdict(LabelEncoder)
            labeled_housing = encoded_df.apply(lambda x: encoder_dict[x.name].fit_transform(x))
            self.encoder = encoder_dict

            objList = self.cat_cols + self.high_cardinality_cols

            le = LabelEncoder()

            for col in objList:
                encoded_df[col] = le.fit_transform(encoded_df[col])
            
            target_name = target
            feature_names = [i for i in self.df.columns.tolist() if i != target]
            
            # Step 1 subset data into train and test set (75/25 split)
            X_train, X_test, y_train, y_test = train_test_split(encoded_df[feature_names], encoded_df[target_name], random_state=42)
            
            if self.df[target_name].dtype in CATEGORICAL_TYPES:
                # CATEGORICAL CLASSIFICATION:
                target_class_names = encoder_dict[target].classes_.tolist()
                
                # Step 2: Make an instance of the Model
                clf = DecisionTreeClassifier(random_state = 42, max_depth = max_depth)

                # Step 3: Train the model on the data
                clf.fit(X_train, y_train)

                # Step 4: Predict labels of unseen (test) data
                prediction = clf.predict(X_test)
                
                print('Classification Report on 25% of Testing Data:')
                print(classification_report(y_test, prediction, target_names=target_class_names))
                                
                #plot feature importances
                feature_importance_plot(features = feature_names, importances = clf.feature_importances_)

                # visualize tree
                import warnings
                warnings.simplefilter(action='ignore', category=FutureWarning)

                viz = dtreeviz(clf, 
                               X_train, 
                               y_train,
                               target_name = target,
                               feature_names = feature_names,
                               class_names = target_class_names)  

                return viz


            else:
                # NUMERICAL REGRESSION:
                regr = tree.DecisionTreeRegressor(max_depth = max_depth)

                # Step 3: Train the model on the data
                regr.fit(X_train, y_train)

                # Step 4: Predict labels of unseen (test) data
                prediction = regr.predict(X_test)
                
                print('Classification Report on 25% of Testing Data:')
                print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, prediction))
                print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, prediction))
                print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
                mape = np.mean(np.abs((y_test - prediction) / np.abs(y_test)))
                print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
                print('Accuracy:', round(100*(1 - mape), 2))
                
                #plot feature importances
                feature_importance_plot(features = feature_names, importances = regr.feature_importances_)

                # visualize tree

                import warnings
                warnings.simplefilter(action='ignore', category=FutureWarning)

                viz = dtreeviz(regr, 
                               X_train, 
                               y_train,
                               target_name = target,
                               feature_names = feature_names,
                               fancy=False)  
                viz.view()
                # return viz
                        
            

            
