import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

### GLOBAL VARIABLES


h = .02  # step size in the mesh

# You can add classifiers to this dictionary, don't forget to import them!
classifiers = {"Logistic Regression": LogisticRegression(),
            "Naive Bayes": GaussianNB(),
            "SVM": svm.SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()}

datasets = [pd.read_csv("p1_blobs.csv"),
            pd.read_csv("p1_circles.csv"),
            pd.read_csv("p1_moons.csv"),
            ]

datasets_names = ["Data A", "Data B", "Data C"]


def plot_Scatter_2D(ax, X, y, Alpha):
    color = ['#FF0000' if l == "A" else '#0000FF' for l in y]
    ax.scatter(X[:, 0], X[:, 1], cmap=cm_bright, c=color, alpha= Alpha)

### Code Begin

print(len(datasets), len(datasets_names))
figure = plt.figure(figsize=(27, 9))

i = 1

for di in range(0, len(datasets), 1):
    print(di, datasets_names[di]);
    ds = datasets[di];
    # preprocess dataset, split into training and test part
    X, y = ds[["d1", "d2"]], ds["Label"]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

    #Plot Train and Test
    plot_Scatter_2D(ax, X_train, y_train, 1)
    plot_Scatter_2D(ax, X_test, y_test, 0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(datasets_names[di], fontsize=20)
    i += 1

    # iterate over classifiers
    for name, clf in zip(classifiers.keys(), classifiers.values()):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        #ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        #ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,alpha=0.6)

        # Plot Train and Test
        y_tr_pred = clf.predict(X_train)
        y_te_pred = clf.predict(X_test)
        plot_Scatter_2D(ax, X_train, y_tr_pred, 1)
        plot_Scatter_2D(ax, X_test, y_te_pred, 0.6)


        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name, fontsize=20)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right', fontsize=20)
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()
