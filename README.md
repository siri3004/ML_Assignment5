# ML_Assignment5

Video Link: https://drive.google.com/file/d/1JlqiUxDIHN6W1kpyUacSIuKq9Y3yEvbx/view

1. Principal Component Analysis

a. Apply PCA on CC dataset. 

b. Apply k-means algorithm on the PCA result and report your observation if the silhouette score has improved or not?

c. Perform Scaling+PCA+K-Means and report performance

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import warnings
        warnings.filterwarnings('ignore')
        import seaborn as sns
        sns.set(style="white", color_codes=True)
        
        df= pd.read_csv("CC GENERAL.csv")
        df.head() 
        
        df.isnull().any()
        
        df.fillna(df.mean(), inplace = True)
        df.isnull().any()
         
        x = df.iloc[:,[1,2,3,4]] 
        y = df.iloc[:,-1] 
        print(x.shape, y.shape)    
        
a. Apply PCA on CC GENERAL dataset.

        pca = PCA(3)  
        x_pca = pca.fit_transform(x) 
        df2 = pd.DataFrame(data = x_pca, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
        df3 = pd.concat([df2, df.iloc[:,-1]], axis = 1)
        df3.head()    
        
 b. Apply k-means algorithm on the original dataset

        x = df3.iloc[:,0:-1]
        y = df3.iloc[:,-1]
        print(x.shape, y.shape) 
        
        from sklearn.cluster import KMeans
        # setting the number of clusters to 3
        nclusters = 3 # this is k = 3 in kmeans

        k_means = KMeans(n_clusters=nclusters)
        k_means.fit(x)
        y_cluster_kmeans = k_means.predict(x)

        from sklearn import metrics
        score = metrics.silhouette_score(x, y_cluster_kmeans)
        print("Sihouette Score without PCA: ",score)

c. Perform Scaling+PCA+K-Means and report performance

        scaler = StandardScaler()
        scaler.fit(x) 
        x_scale = scaler.transform(x)

        pca2 = PCA(3)
        x_pca2 = pca.fit_transform(x_scale)

        df4 = pd.DataFrame(data = x_pca2, columns = ['principal component 1',
                                          'principal component 2', 'principal component 3'])

        final_df = pd.concat([df4, df[['TENURE']]], axis = 1)
        final_df.head()

        from sklearn.cluster import KMeans
        nclusters = 3 # this is the k in kmeans
        km = KMeans(n_clusters=nclusters)
        km.fit(x_scale)

        y_cluster_kmeans = km.predict(x_scale)
        from sklearn import metrics
        score = metrics.silhouette_score(x_scale, y_cluster_kmeans)
        print(score)
        
Explanation:

•	Load the "CC_GENERAL.csv" dataset, drop the irrelevant columns, and handle the missing values.

•	Apply k-means algorithm on the original data and on the PCA-transformed data and calculate the silhouette scores for each.

•	Compare the silhouette score obtained from applying k-means on PCA data with the silhouette score obtained from the original data to see if there is any improvement.

•	Compare the silhouette score obtained from applying Scaling + PCA + K-Means with the silhouette score obtained from k-means + PCA data to see if there is any improvement.       
        
2. Use pd_speech_features.csv

 a. Perform Scaling
 
 b. Apply PCA (k=3) 
 
 c. Use SVM to report performance
        
        
        df_pd = pd.read_csv(r"pd_speech_features.csv")
        
        df_pd.head()
        
        df_pd.isnull().any()
        
a. to perform Scaling

        scaler = StandardScaler()
        x_scale = scaler.fit_transform(x)
        
b. Applying PCA for k=3
        pca = PCA(3)
        x_pca = pca.fit_transform(x_scale)

        principalDf = pd.DataFrame(data = x_pca, columns = ['principal component 1', 'principal component 2','Principal Component 3'])

        finalDf = pd.concat([principalDf, df_pd[['class']]], axis = 1)
        finalDf.head()

c. to use SVM and report performace

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)
        from sklearn.svm import SVC
        svm_classifier = SVC()
        svm_classifier.fit(x_train, y_train)
        y_pred = svm_classifier.predict(x_test)

        print(classification_report(y_test, y_pred, zero_division=1))
        print(confusion_matrix(y_test, y_pred))
        acc_svc = accuracy_score(y_pred,y_test)
        print('accuracy is',acc_svc)

        #Calculate sihouette Score
        score = metrics.silhouette_score(x_test, y_pred)
        print("Sihouette Score: ",score) 
        
Explanation:

•	Load the "pd_speech_features.csv" dataset using Pandas and split it into features (stored in "X") and target variable (stored in "y").

•	Apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset to 3 principal components using the "PCA()" function from scikit-learn.

•	Train a Support Vector Machine (SVM) classifier on the training set using the "SVC()" function from scikit-learn.

•	Evaluate the performance of the classifier on the testing set using the "accuracy_score()" function from scikit-learn and print the accuracy.
        

3. Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data to k=2

        import math
        import numpy as np
        df_iris = pd.read_csv(r"Iris.csv")
        df_iris.head()
        
        df_iris.isnull().any()
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        from sklearn.preprocessing import StandardScaler
        le = LabelEncoder()
        y = le.fit_transform(y)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        le = LabelEncoder()
        y = le.fit_transform(y)
        x_train_std = scaler.fit_transform(df.iloc[:,1:-1].values)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_le = le.fit_transform(df.iloc[:,-1].values)
        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components=2)
        x_train = lda.fit_transform(x_train, y_train)
        x_test = lda.transform(x_test)
        print(x_train.shape,x_test.shape)
        
 Explanation:

•	Load the Iris.csv dataset: The code loads the Iris.csv dataset using Pandas and separates it into features (stored in "X") and target variable (stored in "y").

•	Standardize the data: The code uses the "StandardScaler()" function from scikit-learn to standardize the features by subtracting the mean and scaling to unit variance.

•	Apply Linear Discriminant Analysis (LDA): The code uses the "LinearDiscriminantAnalysis()" function from scikit-learn with the argument "n_components=2" to reduce the dimensionality of the dataset to 2 components.

•	Save the LDA-transformed data to a CSV file: The code saves the LDA-transformed data to a new CSV file named "Iris_LDA.csv".

•	Display the LDA-transformed data: The code displays the LDA-transformed data in a new Pandas dataframe and prints it to the console.

4. Briefly identify the difference between PCA and LDA

•	PCA (Principal Component Analysis) - Linear dimensionality reduction using Singular Value.

•	Decomposition of the data to project it to a lower dimensional space. The input data is centered but not scaled for each feature before applying the SVD.

•	It uses the LAPACK implementation of the full SVD or a randomized truncated SVD.

•	LDA (Linear Discriminant Analysis) - A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a Gaussian density to each class if all classes share the same covariance matrix.

•	Both LDA and PCA are linear transformation techniques: 

LDA is supervised whereas PCA is unsupervised - PCA ignores class labels. In contrast to PCA, LDA attempts to find a feature subspace that maximizes class separability.
     

        
