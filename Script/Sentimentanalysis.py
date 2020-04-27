{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "toc": true
   },
   "source": [
    "<h1>Table of Contents<span class=\"tocSkip\"></span></h1>\n",
    "<div class=\"toc\"><ul class=\"toc-item\"><li><span><a href=\"#Introduction\" data-toc-modified-id=\"Introduction-1\"><span class=\"toc-item-num\">1&nbsp;&nbsp;</span>Introduction</a></span><ul class=\"toc-item\"><li><span><a href=\"#Library\" data-toc-modified-id=\"Library-1.1\"><span class=\"toc-item-num\">1.1&nbsp;&nbsp;</span>Library</a></span></li><li><span><a href=\"#Loading-data-set´s\" data-toc-modified-id=\"Loading-data-set´s-1.2\"><span class=\"toc-item-num\">1.2&nbsp;&nbsp;</span>Loading data set´s</a></span></li></ul></li><li><span><a href=\"#Data-Exploration\" data-toc-modified-id=\"Data-Exploration-2\"><span class=\"toc-item-num\">2&nbsp;&nbsp;</span>Data Exploration</a></span><ul class=\"toc-item\"><li><span><a href=\"#Na´s-Analysis\" data-toc-modified-id=\"Na´s-Analysis-2.1\"><span class=\"toc-item-num\">2.1&nbsp;&nbsp;</span>Na´s Analysis</a></span></li><li><span><a href=\"#Distribution-(STD)\" data-toc-modified-id=\"Distribution-(STD)-2.2\"><span class=\"toc-item-num\">2.2&nbsp;&nbsp;</span>Distribution (STD)</a></span></li></ul></li><li><span><a href=\"#Pre-Sentiment-Analys\" data-toc-modified-id=\"Pre-Sentiment-Analys-3\"><span class=\"toc-item-num\">3&nbsp;&nbsp;</span>Pre Sentiment-Analys</a></span><ul class=\"toc-item\"><li><span><a href=\"#Histograms\" data-toc-modified-id=\"Histograms-3.1\"><span class=\"toc-item-num\">3.1&nbsp;&nbsp;</span>Histograms</a></span></li><li><span><a href=\"#Examine-Correlation\" data-toc-modified-id=\"Examine-Correlation-3.2\"><span class=\"toc-item-num\">3.2&nbsp;&nbsp;</span>Examine Correlation</a></span></li></ul></li><li><span><a href=\"#Feature-Selection\" data-toc-modified-id=\"Feature-Selection-4\"><span class=\"toc-item-num\">4&nbsp;&nbsp;</span>Feature Selection</a></span><ul class=\"toc-item\"><li><span><a href=\"#SelectKBest\" data-toc-modified-id=\"SelectKBest-4.1\"><span class=\"toc-item-num\">4.1&nbsp;&nbsp;</span>SelectKBest</a></span></li></ul></li><li><span><a href=\"#VarianceThreshold\" data-toc-modified-id=\"VarianceThreshold-5\"><span class=\"toc-item-num\">5&nbsp;&nbsp;</span>VarianceThreshold</a></span></li><li><span><a href=\"#Selection-based-on-mutual-information-(MI)\" data-toc-modified-id=\"Selection-based-on-mutual-information-(MI)-6\"><span class=\"toc-item-num\">6&nbsp;&nbsp;</span>Selection based on mutual information (MI)</a></span><ul class=\"toc-item\"><li><span><a href=\"#IPHONE-ANALYSIS-MUTUAL-INFORMATION\" data-toc-modified-id=\"IPHONE-ANALYSIS-MUTUAL-INFORMATION-6.1\"><span class=\"toc-item-num\">6.1&nbsp;&nbsp;</span>IPHONE ANALYSIS-MUTUAL INFORMATION</a></span></li></ul></li><li><span><a href=\"#GALAXY-ANALYSIS-MUTUAL-INFORMATION\" data-toc-modified-id=\"GALAXY-ANALYSIS-MUTUAL-INFORMATION-7\"><span class=\"toc-item-num\">7&nbsp;&nbsp;</span>GALAXY ANALYSIS-MUTUAL INFORMATION</a></span></li><li><span><a href=\"#RFE\" data-toc-modified-id=\"RFE-8\"><span class=\"toc-item-num\">8&nbsp;&nbsp;</span>RFE</a></span></li><li><span><a href=\"#Recursive-Feature-Elimination\" data-toc-modified-id=\"Recursive-Feature-Elimination-9\"><span class=\"toc-item-num\">9&nbsp;&nbsp;</span>Recursive Feature Elimination</a></span></li><li><span><a href=\"#CROSS-VALIDATION-TUNE-AND-MODEL-FOR-VARIANCE-ANALYSYS\" data-toc-modified-id=\"CROSS-VALIDATION-TUNE-AND-MODEL-FOR-VARIANCE-ANALYSYS-10\"><span class=\"toc-item-num\">10&nbsp;&nbsp;</span>CROSS VALIDATION-TUNE AND MODEL FOR VARIANCE ANALYSYS</a></span></li><li><span><a href=\"#Cross-Validation-Tune-and-Modeling-for-M.I.\" data-toc-modified-id=\"Cross-Validation-Tune-and-Modeling-for-M.I.-11\"><span class=\"toc-item-num\">11&nbsp;&nbsp;</span>Cross Validation-Tune and Modeling for M.I.</a></span></li><li><span><a href=\"#Final-Predicitons-Iphone-Large-Matrix\" data-toc-modified-id=\"Final-Predicitons-Iphone-Large-Matrix-12\"><span class=\"toc-item-num\">12&nbsp;&nbsp;</span>Final Predicitons Iphone-Large Matrix</a></span></li><li><span><a href=\"#Final-Predictions-Galaxy---Large-Matrix\" data-toc-modified-id=\"Final-Predictions-Galaxy---Large-Matrix-13\"><span class=\"toc-item-num\">13&nbsp;&nbsp;</span>Final Predictions Galaxy - Large Matrix</a></span></li><li><span><a href=\"#CONCLUSIONS\" data-toc-modified-id=\"CONCLUSIONS-14\"><span class=\"toc-item-num\">14&nbsp;&nbsp;</span>CONCLUSIONS</a></span></li></ul></div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Introduction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The aim of the project for Helio is to make an sentimental analysis regarding smart phones focused on Iphone and Galaxy.\n",
    "For wich we have been working with 3 data sets; galaxy data set(12793,59), iphone data set (12973,59), large matrix data set(114553,58), Galaxy and Iphone data sets has been using for cv analysis and also fot tuning and modeling and large matrix has been used as validation set, sentiment is categorized from 0 (very negative) to 5 (very positive).\n",
    "For this task purpose we have been working with Common Crawl with AWS to get better experience in data analysis, also many new concepts has been acquired for this sesion.\n",
    "\n",
    "The focus of the analysis is on the pre-processeing wich is why i have take in consideration many assumptions and take the one that fit the best for my final predictions, regarding on this some approaches have been considered:\n",
    "\n",
    "1.Data exploration and sentiment pre-analysis\n",
    "\n",
    "2.Variance Threshold analysis\n",
    "\n",
    "3.Selection based on mutual information analysis\n",
    "\n",
    "4.Best k analysis\n",
    "\n",
    "5.PCA analysis performed for some of the predictions that perform better\n",
    "\n",
    "6.Recursive Feature elimination \n",
    "\n",
    "7.CV-TUNE-MODELING\n",
    "\n",
    "The idea is try to get the best way to represent the data and based on that, model and decide wich model fit the best.\n",
    "The metrics for evaluate the performance fot this project : accuracy "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:17:36.519690Z",
     "start_time": "2020-03-03T19:17:36.513653Z"
    }
   },
   "source": [
    "## Library "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:32:03.273743Z",
     "start_time": "2020-03-03T17:32:03.182991Z"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import scipy as sp\n",
    "import os\n",
    "import sklearn\n",
    "%matplotlib inline\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.feature_selection import chi2\n",
    "mpl.rc('axes', labelsize=14)\n",
    "mpl.rc('xtick', labelsize=12)\n",
    "mpl.rc('ytick', labelsize=12)\n",
    "np.random.seed(42)\n",
    "from sklearn.model_selection import train_test_split\n",
    "from math import sqrt\n",
    "import random\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.model_selection import learning_curve, validation_curve\n",
    "from sklearn.linear_model import LinearRegression, Lasso, Ridge\n",
    "from sklearn.kernel_ridge import KernelRidge\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.tree import ExtraTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor\n",
    "from sklearn.multioutput import MultiOutputRegressor\n",
    "from sklearn.svm import SVR\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.neural_network import MLPRegressor\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.model_selection import learning_curve, validation_curve\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.feature_selection import VarianceThreshold\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVR\n",
    "from sklearn import linear_model\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.preprocessing import scale\n",
    "#model metrics\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.metrics import cohen_kappa_score\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.svm import SVC"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Loading data set´s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 284,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:44:05.092169Z",
     "start_time": "2020-03-03T19:44:04.926455Z"
    }
   },
   "outputs": [],
   "source": [
    "# %% loading Data Set´s \n",
    "galaxyPath= 'C:/Respaldo FR/UBIQUM/LAST/'\n",
    "import pandas as pd\n",
    "galaxyData = pd.read_csv(galaxyPath+ 'galaxy_smallmatrix_labeled_8d.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 285,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:44:06.030837Z",
     "start_time": "2020-03-03T19:44:05.906264Z"
    }
   },
   "outputs": [],
   "source": [
    "iphonePath= 'C:/Respaldo FR/UBIQUM/LAST/'\n",
    "import pandas as pd\n",
    "iphoneData = pd.read_csv(iphonePath+ 'iphone_smallmatrix_labeled_8d.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 286,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:44:07.804086Z",
     "start_time": "2020-03-03T19:44:06.979641Z"
    }
   },
   "outputs": [],
   "source": [
    "matrixPath='C:/Respaldo FR/UBIQUM/LAST/'\n",
    "import pandas as pd\n",
    "MAtrix_forLazayPeople_as_me = pd.read_csv(matrixPath+ \"MAtrix_forLazayPeople.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-02-29T18:08:06.821139Z",
     "start_time": "2020-02-29T18:08:06.817129Z"
    }
   },
   "source": [
    "# Data Exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-02T10:38:22.846786Z",
     "start_time": "2020-03-02T10:38:22.840062Z"
    }
   },
   "outputs": [],
   "source": [
    "#display(galaxyData.shape)\n",
    "#display(iphoneData.shape)\n",
    "#display(galaxyData.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-02T10:32:19.286713Z",
     "start_time": "2020-03-02T10:32:19.277024Z"
    }
   },
   "outputs": [],
   "source": [
    "#display(galaxyData.describe())\n",
    "#display(iphoneData.describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Na´s Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#missing_values_count = galaxyData.isnull().sum()\n",
    "#print(missing_values_count)\n",
    "#missing_values_count = iphoneData.isnull().sum()\n",
    "#print(missing_values_count)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##  Distribution (STD) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 383,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T10:20:53.641354Z",
     "start_time": "2020-03-03T10:20:52.915935Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x19ecbbd2f08>"
      ]
     },
     "execution_count": 383,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "metrics=(iphoneData.describe())\n",
    "metrics=metrics.transpose()\n",
    "metrics=metrics[:-1]\n",
    "metrics[\"std\"].hist(bins=40)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We will use this information for the threshold on our further analysis.\n",
    "A balance of the data set will be a good option to perform better any model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-02-29T18:16:26.526509Z",
     "start_time": "2020-02-29T18:16:26.521523Z"
    }
   },
   "source": [
    "# Pre Sentiment-Analys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-02-29T18:14:22.881566Z",
     "start_time": "2020-02-29T18:14:22.867575Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    12973.000000\n",
       "mean         3.824327\n",
       "std          1.781302\n",
       "min          0.000000\n",
       "25%          3.000000\n",
       "50%          5.000000\n",
       "75%          5.000000\n",
       "max          5.000000\n",
       "Name: galaxysentiment, dtype: float64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sentimentgalaxy=galaxyData.galaxysentiment\n",
    "sentimentgalaxy.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-02-29T18:18:51.861062Z",
     "start_time": "2020-02-29T18:18:51.843110Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    12973.000000\n",
       "mean         3.724505\n",
       "std          1.851348\n",
       "min          0.000000\n",
       "25%          3.000000\n",
       "50%          5.000000\n",
       "75%          5.000000\n",
       "max          5.000000\n",
       "Name: iphonesentiment, dtype: float64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sentimentiphone=iphoneData.iphonesentiment\n",
    "sentimentiphone.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 290,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:56:10.707140Z",
     "start_time": "2020-03-03T19:56:10.603384Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "galaxysentiment\n",
      "0    0.131273\n",
      "1    0.030062\n",
      "2    0.034996\n",
      "3    0.091575\n",
      "4    0.110923\n",
      "5    0.601172\n",
      "dtype: float64\n",
      "iphonesentiment\n",
      "0    0.151237\n",
      "1    0.030062\n",
      "2    0.034996\n",
      "3    0.091575\n",
      "4    0.110923\n",
      "5    0.581207\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# Class percentage\n",
    "print(galaxyData.groupby('galaxysentiment').size()/galaxyData['galaxysentiment'].count())\n",
    "print(iphoneData.groupby('iphonesentiment').size()/iphoneData['iphonesentiment'].count())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we can see the percentage regarding the levels of the sentiment in the sclase mentioned."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Histograms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-02-29T18:50:07.943260Z",
     "start_time": "2020-02-29T18:50:07.601109Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'count')"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# matplotlib histogram\n",
    "plt.hist(galaxyData['galaxysentiment'], color = 'Purple', edgecolor = 'white',\n",
    "         bins = int(180/5))\n",
    "# Adding labels\n",
    "plt.title('Histogram of sentiment')\n",
    "plt.xlabel('sentiment')\n",
    "plt.ylabel('count')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-02-29T18:51:04.356215Z",
     "start_time": "2020-02-29T18:51:03.933624Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'count')"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# matplotlib histogram\n",
    "plt.hist(iphoneData['iphonesentiment'], color = 'Purple', edgecolor = 'white',\n",
    "         bins = int(180/5))\n",
    "# Adding labels\n",
    "plt.title('Histogram of sentiment')\n",
    "plt.xlabel('sentiment')\n",
    "plt.ylabel('count')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see again that any future action for balance the data should be taken in consideration wich is why first we will amke a correlation analysis an then feature engineering actions will be performed."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-02-29T18:38:25.692221Z",
     "start_time": "2020-02-29T18:38:25.685539Z"
    }
   },
   "source": [
    "## Examine Correlation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-02-29T18:52:44.443829Z",
     "start_time": "2020-02-29T18:52:41.831285Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Sample figsize in inches\n",
    "fig, ax = plt.subplots(figsize=(20,10))\n",
    "\n",
    "# Imbalanced DataFrame Correlation\n",
    "corr = galaxyData.corr()\n",
    "sns.heatmap(corr, cmap='Purples', annot_kws={'size':30}, ax=ax)\n",
    "ax.set_title(\"Correlation Matrix GALAXY DATA-Imbalanced\", fontsize=14)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-02T11:02:21.581203Z",
     "start_time": "2020-03-02T11:02:21.137389Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>iphone</th>\n",
       "      <th>samsunggalaxy</th>\n",
       "      <th>sonyxperia</th>\n",
       "      <th>nokialumina</th>\n",
       "      <th>htcphone</th>\n",
       "      <th>ios</th>\n",
       "      <th>googleandroid</th>\n",
       "      <th>iphonecampos</th>\n",
       "      <th>samsungcampos</th>\n",
       "      <th>sonycampos</th>\n",
       "      <th>...</th>\n",
       "      <th>sonyperunc</th>\n",
       "      <th>nokiaperunc</th>\n",
       "      <th>htcperunc</th>\n",
       "      <th>iosperpos</th>\n",
       "      <th>googleperpos</th>\n",
       "      <th>iosperneg</th>\n",
       "      <th>googleperneg</th>\n",
       "      <th>iosperunc</th>\n",
       "      <th>googleperunc</th>\n",
       "      <th>iphonesentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>iphone</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.019786</td>\n",
       "      <td>-0.011618</td>\n",
       "      <td>-0.013423</td>\n",
       "      <td>-0.002731</td>\n",
       "      <td>0.922060</td>\n",
       "      <td>0.107530</td>\n",
       "      <td>0.078157</td>\n",
       "      <td>0.057395</td>\n",
       "      <td>-0.004594</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.003045</td>\n",
       "      <td>-0.009704</td>\n",
       "      <td>0.011414</td>\n",
       "      <td>-0.020059</td>\n",
       "      <td>0.118008</td>\n",
       "      <td>-0.019081</td>\n",
       "      <td>0.138742</td>\n",
       "      <td>-0.020368</td>\n",
       "      <td>0.067859</td>\n",
       "      <td>0.014859</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsunggalaxy</td>\n",
       "      <td>0.019786</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.366671</td>\n",
       "      <td>-0.006088</td>\n",
       "      <td>0.017899</td>\n",
       "      <td>-0.044678</td>\n",
       "      <td>0.236162</td>\n",
       "      <td>0.030556</td>\n",
       "      <td>0.252121</td>\n",
       "      <td>0.145969</td>\n",
       "      <td>...</td>\n",
       "      <td>0.037482</td>\n",
       "      <td>0.007305</td>\n",
       "      <td>0.044928</td>\n",
       "      <td>-0.005802</td>\n",
       "      <td>0.246046</td>\n",
       "      <td>-0.007839</td>\n",
       "      <td>0.290975</td>\n",
       "      <td>-0.015329</td>\n",
       "      <td>0.142252</td>\n",
       "      <td>-0.359173</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonyxperia</td>\n",
       "      <td>-0.011618</td>\n",
       "      <td>0.366671</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.006350</td>\n",
       "      <td>0.023682</td>\n",
       "      <td>-0.023884</td>\n",
       "      <td>-0.018288</td>\n",
       "      <td>0.005068</td>\n",
       "      <td>0.050140</td>\n",
       "      <td>0.396751</td>\n",
       "      <td>...</td>\n",
       "      <td>0.151675</td>\n",
       "      <td>-0.004253</td>\n",
       "      <td>-0.004888</td>\n",
       "      <td>-0.011009</td>\n",
       "      <td>-0.008467</td>\n",
       "      <td>-0.010323</td>\n",
       "      <td>-0.008570</td>\n",
       "      <td>-0.014802</td>\n",
       "      <td>-0.007916</td>\n",
       "      <td>-0.233170</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokialumina</td>\n",
       "      <td>-0.013423</td>\n",
       "      <td>-0.006088</td>\n",
       "      <td>-0.006350</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000673</td>\n",
       "      <td>-0.002819</td>\n",
       "      <td>-0.001115</td>\n",
       "      <td>0.029824</td>\n",
       "      <td>0.009299</td>\n",
       "      <td>-0.002754</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.001204</td>\n",
       "      <td>0.648441</td>\n",
       "      <td>0.023757</td>\n",
       "      <td>0.030719</td>\n",
       "      <td>0.006515</td>\n",
       "      <td>0.032721</td>\n",
       "      <td>0.000653</td>\n",
       "      <td>0.052887</td>\n",
       "      <td>0.007999</td>\n",
       "      <td>-0.055962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htcphone</td>\n",
       "      <td>-0.002731</td>\n",
       "      <td>0.017899</td>\n",
       "      <td>0.023682</td>\n",
       "      <td>0.000673</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.005002</td>\n",
       "      <td>0.016498</td>\n",
       "      <td>0.006952</td>\n",
       "      <td>0.010865</td>\n",
       "      <td>0.010432</td>\n",
       "      <td>...</td>\n",
       "      <td>0.005018</td>\n",
       "      <td>0.000112</td>\n",
       "      <td>0.021448</td>\n",
       "      <td>-0.002927</td>\n",
       "      <td>0.019186</td>\n",
       "      <td>-0.002758</td>\n",
       "      <td>0.020726</td>\n",
       "      <td>-0.002666</td>\n",
       "      <td>0.013305</td>\n",
       "      <td>-0.051285</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>ios</td>\n",
       "      <td>0.922060</td>\n",
       "      <td>-0.044678</td>\n",
       "      <td>-0.023884</td>\n",
       "      <td>-0.002819</td>\n",
       "      <td>-0.005002</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.026404</td>\n",
       "      <td>0.042128</td>\n",
       "      <td>-0.010741</td>\n",
       "      <td>-0.009369</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.004832</td>\n",
       "      <td>0.005030</td>\n",
       "      <td>-0.011930</td>\n",
       "      <td>0.118278</td>\n",
       "      <td>-0.016402</td>\n",
       "      <td>0.112330</td>\n",
       "      <td>-0.018028</td>\n",
       "      <td>0.117035</td>\n",
       "      <td>-0.010233</td>\n",
       "      <td>0.001656</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>googleandroid</td>\n",
       "      <td>0.107530</td>\n",
       "      <td>0.236162</td>\n",
       "      <td>-0.018288</td>\n",
       "      <td>-0.001115</td>\n",
       "      <td>0.016498</td>\n",
       "      <td>-0.026404</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.104420</td>\n",
       "      <td>0.315487</td>\n",
       "      <td>-0.000206</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.004135</td>\n",
       "      <td>-0.001407</td>\n",
       "      <td>0.109685</td>\n",
       "      <td>-0.016702</td>\n",
       "      <td>0.638581</td>\n",
       "      <td>-0.015825</td>\n",
       "      <td>0.716515</td>\n",
       "      <td>-0.016377</td>\n",
       "      <td>0.371998</td>\n",
       "      <td>-0.189142</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphonecampos</td>\n",
       "      <td>0.078157</td>\n",
       "      <td>0.030556</td>\n",
       "      <td>0.005068</td>\n",
       "      <td>0.029824</td>\n",
       "      <td>0.006952</td>\n",
       "      <td>0.042128</td>\n",
       "      <td>0.104420</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.062438</td>\n",
       "      <td>0.045009</td>\n",
       "      <td>...</td>\n",
       "      <td>0.019987</td>\n",
       "      <td>0.014827</td>\n",
       "      <td>0.067283</td>\n",
       "      <td>-0.003991</td>\n",
       "      <td>0.117902</td>\n",
       "      <td>-0.007060</td>\n",
       "      <td>0.124355</td>\n",
       "      <td>-0.001037</td>\n",
       "      <td>0.073004</td>\n",
       "      <td>-0.029731</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsungcampos</td>\n",
       "      <td>0.057395</td>\n",
       "      <td>0.252121</td>\n",
       "      <td>0.050140</td>\n",
       "      <td>0.009299</td>\n",
       "      <td>0.010865</td>\n",
       "      <td>-0.010741</td>\n",
       "      <td>0.315487</td>\n",
       "      <td>0.062438</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.145429</td>\n",
       "      <td>...</td>\n",
       "      <td>0.057860</td>\n",
       "      <td>0.033197</td>\n",
       "      <td>0.061304</td>\n",
       "      <td>0.102471</td>\n",
       "      <td>0.298281</td>\n",
       "      <td>0.075695</td>\n",
       "      <td>0.357362</td>\n",
       "      <td>0.044890</td>\n",
       "      <td>0.159171</td>\n",
       "      <td>-0.112743</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonycampos</td>\n",
       "      <td>-0.004594</td>\n",
       "      <td>0.145969</td>\n",
       "      <td>0.396751</td>\n",
       "      <td>-0.002754</td>\n",
       "      <td>0.010432</td>\n",
       "      <td>-0.009369</td>\n",
       "      <td>-0.000206</td>\n",
       "      <td>0.045009</td>\n",
       "      <td>0.145429</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>0.378812</td>\n",
       "      <td>-0.001845</td>\n",
       "      <td>0.015781</td>\n",
       "      <td>-0.003118</td>\n",
       "      <td>0.006673</td>\n",
       "      <td>-0.002863</td>\n",
       "      <td>0.008455</td>\n",
       "      <td>-0.006421</td>\n",
       "      <td>-0.003434</td>\n",
       "      <td>-0.090665</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokiacampos</td>\n",
       "      <td>-0.008439</td>\n",
       "      <td>-0.000400</td>\n",
       "      <td>-0.004232</td>\n",
       "      <td>0.700415</td>\n",
       "      <td>0.000465</td>\n",
       "      <td>0.005425</td>\n",
       "      <td>0.003284</td>\n",
       "      <td>0.030817</td>\n",
       "      <td>0.014860</td>\n",
       "      <td>-0.001836</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000802</td>\n",
       "      <td>0.858295</td>\n",
       "      <td>0.017261</td>\n",
       "      <td>0.103123</td>\n",
       "      <td>0.011564</td>\n",
       "      <td>0.103540</td>\n",
       "      <td>0.003941</td>\n",
       "      <td>0.165188</td>\n",
       "      <td>0.012518</td>\n",
       "      <td>-0.033375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htccampos</td>\n",
       "      <td>0.022717</td>\n",
       "      <td>0.065274</td>\n",
       "      <td>0.016507</td>\n",
       "      <td>0.021295</td>\n",
       "      <td>0.023189</td>\n",
       "      <td>-0.012390</td>\n",
       "      <td>0.148095</td>\n",
       "      <td>0.623912</td>\n",
       "      <td>0.090099</td>\n",
       "      <td>0.058852</td>\n",
       "      <td>...</td>\n",
       "      <td>0.018081</td>\n",
       "      <td>0.010478</td>\n",
       "      <td>0.253678</td>\n",
       "      <td>-0.006121</td>\n",
       "      <td>0.163145</td>\n",
       "      <td>-0.005761</td>\n",
       "      <td>0.177230</td>\n",
       "      <td>-0.006079</td>\n",
       "      <td>0.100031</td>\n",
       "      <td>-0.120434</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphonecamneg</td>\n",
       "      <td>0.490524</td>\n",
       "      <td>0.126063</td>\n",
       "      <td>-0.006715</td>\n",
       "      <td>0.063245</td>\n",
       "      <td>0.014155</td>\n",
       "      <td>0.386966</td>\n",
       "      <td>0.391802</td>\n",
       "      <td>0.541340</td>\n",
       "      <td>0.206020</td>\n",
       "      <td>0.013254</td>\n",
       "      <td>...</td>\n",
       "      <td>0.032570</td>\n",
       "      <td>0.026550</td>\n",
       "      <td>0.114716</td>\n",
       "      <td>-0.012229</td>\n",
       "      <td>0.417185</td>\n",
       "      <td>-0.013642</td>\n",
       "      <td>0.468075</td>\n",
       "      <td>-0.010749</td>\n",
       "      <td>0.241003</td>\n",
       "      <td>-0.083963</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsungcamneg</td>\n",
       "      <td>0.142553</td>\n",
       "      <td>0.342919</td>\n",
       "      <td>-0.004308</td>\n",
       "      <td>0.009546</td>\n",
       "      <td>0.020021</td>\n",
       "      <td>-0.015273</td>\n",
       "      <td>0.711403</td>\n",
       "      <td>0.117451</td>\n",
       "      <td>0.608840</td>\n",
       "      <td>0.032897</td>\n",
       "      <td>...</td>\n",
       "      <td>0.060837</td>\n",
       "      <td>0.036543</td>\n",
       "      <td>0.122048</td>\n",
       "      <td>0.110073</td>\n",
       "      <td>0.658644</td>\n",
       "      <td>0.081294</td>\n",
       "      <td>0.794282</td>\n",
       "      <td>0.047045</td>\n",
       "      <td>0.342120</td>\n",
       "      <td>-0.185989</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonycamneg</td>\n",
       "      <td>-0.001830</td>\n",
       "      <td>0.031821</td>\n",
       "      <td>0.345296</td>\n",
       "      <td>-0.001229</td>\n",
       "      <td>0.004909</td>\n",
       "      <td>-0.003854</td>\n",
       "      <td>0.013539</td>\n",
       "      <td>0.019994</td>\n",
       "      <td>0.053985</td>\n",
       "      <td>0.408991</td>\n",
       "      <td>...</td>\n",
       "      <td>0.604012</td>\n",
       "      <td>-0.000823</td>\n",
       "      <td>0.026290</td>\n",
       "      <td>-0.001276</td>\n",
       "      <td>0.020904</td>\n",
       "      <td>-0.001166</td>\n",
       "      <td>0.025126</td>\n",
       "      <td>-0.002865</td>\n",
       "      <td>-0.001532</td>\n",
       "      <td>-0.024826</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokiacamneg</td>\n",
       "      <td>-0.009186</td>\n",
       "      <td>-0.000979</td>\n",
       "      <td>-0.004467</td>\n",
       "      <td>0.729434</td>\n",
       "      <td>0.000191</td>\n",
       "      <td>0.004651</td>\n",
       "      <td>-0.001824</td>\n",
       "      <td>0.026855</td>\n",
       "      <td>0.014368</td>\n",
       "      <td>-0.001938</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000847</td>\n",
       "      <td>0.788927</td>\n",
       "      <td>0.016234</td>\n",
       "      <td>0.089002</td>\n",
       "      <td>0.002719</td>\n",
       "      <td>0.090311</td>\n",
       "      <td>-0.000445</td>\n",
       "      <td>0.143676</td>\n",
       "      <td>0.003772</td>\n",
       "      <td>-0.033069</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htccamneg</td>\n",
       "      <td>0.104613</td>\n",
       "      <td>0.222777</td>\n",
       "      <td>-0.012284</td>\n",
       "      <td>0.037256</td>\n",
       "      <td>0.036765</td>\n",
       "      <td>-0.023049</td>\n",
       "      <td>0.562703</td>\n",
       "      <td>0.206585</td>\n",
       "      <td>0.295428</td>\n",
       "      <td>0.013568</td>\n",
       "      <td>...</td>\n",
       "      <td>0.029574</td>\n",
       "      <td>0.019518</td>\n",
       "      <td>0.425361</td>\n",
       "      <td>-0.010934</td>\n",
       "      <td>0.578325</td>\n",
       "      <td>-0.009878</td>\n",
       "      <td>0.652644</td>\n",
       "      <td>-0.010191</td>\n",
       "      <td>0.333727</td>\n",
       "      <td>-0.222972</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphonecamunc</td>\n",
       "      <td>0.750403</td>\n",
       "      <td>-0.010155</td>\n",
       "      <td>-0.007638</td>\n",
       "      <td>0.016237</td>\n",
       "      <td>0.001174</td>\n",
       "      <td>0.732612</td>\n",
       "      <td>0.042955</td>\n",
       "      <td>0.473266</td>\n",
       "      <td>0.028875</td>\n",
       "      <td>0.016442</td>\n",
       "      <td>...</td>\n",
       "      <td>0.025256</td>\n",
       "      <td>0.009049</td>\n",
       "      <td>0.057397</td>\n",
       "      <td>-0.004920</td>\n",
       "      <td>0.076916</td>\n",
       "      <td>-0.008706</td>\n",
       "      <td>0.074858</td>\n",
       "      <td>-0.001336</td>\n",
       "      <td>0.058139</td>\n",
       "      <td>0.001443</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsungcamunc</td>\n",
       "      <td>0.073451</td>\n",
       "      <td>0.316134</td>\n",
       "      <td>0.058777</td>\n",
       "      <td>0.040922</td>\n",
       "      <td>0.015644</td>\n",
       "      <td>-0.012390</td>\n",
       "      <td>0.391433</td>\n",
       "      <td>0.076943</td>\n",
       "      <td>0.814799</td>\n",
       "      <td>0.164043</td>\n",
       "      <td>...</td>\n",
       "      <td>0.152542</td>\n",
       "      <td>0.123181</td>\n",
       "      <td>0.124516</td>\n",
       "      <td>0.129012</td>\n",
       "      <td>0.417375</td>\n",
       "      <td>0.097355</td>\n",
       "      <td>0.476690</td>\n",
       "      <td>0.057612</td>\n",
       "      <td>0.269432</td>\n",
       "      <td>-0.138046</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonycamunc</td>\n",
       "      <td>-0.003064</td>\n",
       "      <td>0.104123</td>\n",
       "      <td>0.376633</td>\n",
       "      <td>-0.001914</td>\n",
       "      <td>0.009843</td>\n",
       "      <td>-0.006484</td>\n",
       "      <td>-0.006578</td>\n",
       "      <td>0.029397</td>\n",
       "      <td>0.098836</td>\n",
       "      <td>0.528452</td>\n",
       "      <td>...</td>\n",
       "      <td>0.567358</td>\n",
       "      <td>-0.001282</td>\n",
       "      <td>0.031963</td>\n",
       "      <td>-0.000890</td>\n",
       "      <td>-0.003825</td>\n",
       "      <td>-0.000746</td>\n",
       "      <td>-0.004204</td>\n",
       "      <td>-0.004463</td>\n",
       "      <td>-0.002386</td>\n",
       "      <td>-0.050327</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokiacamunc</td>\n",
       "      <td>-0.008602</td>\n",
       "      <td>0.005691</td>\n",
       "      <td>-0.003972</td>\n",
       "      <td>0.634171</td>\n",
       "      <td>0.000364</td>\n",
       "      <td>0.006341</td>\n",
       "      <td>0.000325</td>\n",
       "      <td>0.021277</td>\n",
       "      <td>0.028323</td>\n",
       "      <td>-0.001723</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000753</td>\n",
       "      <td>0.958152</td>\n",
       "      <td>0.015949</td>\n",
       "      <td>0.108424</td>\n",
       "      <td>0.005909</td>\n",
       "      <td>0.108898</td>\n",
       "      <td>0.001299</td>\n",
       "      <td>0.173504</td>\n",
       "      <td>0.006829</td>\n",
       "      <td>-0.031550</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htccamunc</td>\n",
       "      <td>0.026138</td>\n",
       "      <td>0.072964</td>\n",
       "      <td>0.014249</td>\n",
       "      <td>0.036124</td>\n",
       "      <td>0.029152</td>\n",
       "      <td>-0.015785</td>\n",
       "      <td>0.166182</td>\n",
       "      <td>0.321523</td>\n",
       "      <td>0.104495</td>\n",
       "      <td>0.056574</td>\n",
       "      <td>...</td>\n",
       "      <td>0.050625</td>\n",
       "      <td>0.018802</td>\n",
       "      <td>0.601513</td>\n",
       "      <td>-0.007866</td>\n",
       "      <td>0.223305</td>\n",
       "      <td>-0.007102</td>\n",
       "      <td>0.227577</td>\n",
       "      <td>-0.005186</td>\n",
       "      <td>0.162431</td>\n",
       "      <td>-0.148881</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphonedispos</td>\n",
       "      <td>0.052625</td>\n",
       "      <td>-0.006526</td>\n",
       "      <td>-0.018121</td>\n",
       "      <td>0.028316</td>\n",
       "      <td>0.000253</td>\n",
       "      <td>0.014377</td>\n",
       "      <td>0.066953</td>\n",
       "      <td>0.272587</td>\n",
       "      <td>0.039427</td>\n",
       "      <td>0.019617</td>\n",
       "      <td>...</td>\n",
       "      <td>0.027681</td>\n",
       "      <td>0.012382</td>\n",
       "      <td>0.091895</td>\n",
       "      <td>0.020232</td>\n",
       "      <td>0.165576</td>\n",
       "      <td>0.015293</td>\n",
       "      <td>0.147023</td>\n",
       "      <td>0.024767</td>\n",
       "      <td>0.179686</td>\n",
       "      <td>0.014547</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsungdispos</td>\n",
       "      <td>0.061074</td>\n",
       "      <td>0.281379</td>\n",
       "      <td>0.040063</td>\n",
       "      <td>0.041456</td>\n",
       "      <td>0.013145</td>\n",
       "      <td>-0.009906</td>\n",
       "      <td>0.316132</td>\n",
       "      <td>0.060476</td>\n",
       "      <td>0.643692</td>\n",
       "      <td>0.111287</td>\n",
       "      <td>...</td>\n",
       "      <td>0.111113</td>\n",
       "      <td>0.123591</td>\n",
       "      <td>0.285206</td>\n",
       "      <td>0.118107</td>\n",
       "      <td>0.606458</td>\n",
       "      <td>0.092014</td>\n",
       "      <td>0.579951</td>\n",
       "      <td>0.057288</td>\n",
       "      <td>0.636103</td>\n",
       "      <td>-0.099262</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonydispos</td>\n",
       "      <td>-0.003827</td>\n",
       "      <td>0.061360</td>\n",
       "      <td>0.252589</td>\n",
       "      <td>-0.001528</td>\n",
       "      <td>0.006959</td>\n",
       "      <td>0.004207</td>\n",
       "      <td>-0.001669</td>\n",
       "      <td>0.017749</td>\n",
       "      <td>0.058122</td>\n",
       "      <td>0.404993</td>\n",
       "      <td>...</td>\n",
       "      <td>0.340766</td>\n",
       "      <td>-0.001024</td>\n",
       "      <td>0.019407</td>\n",
       "      <td>0.025392</td>\n",
       "      <td>0.000158</td>\n",
       "      <td>0.024833</td>\n",
       "      <td>0.001709</td>\n",
       "      <td>-0.003562</td>\n",
       "      <td>-0.001905</td>\n",
       "      <td>-0.038635</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokiadispos</td>\n",
       "      <td>-0.008202</td>\n",
       "      <td>0.010248</td>\n",
       "      <td>-0.003772</td>\n",
       "      <td>0.650253</td>\n",
       "      <td>0.000311</td>\n",
       "      <td>0.003065</td>\n",
       "      <td>-0.004174</td>\n",
       "      <td>0.026317</td>\n",
       "      <td>0.038371</td>\n",
       "      <td>-0.001636</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000715</td>\n",
       "      <td>0.836700</td>\n",
       "      <td>0.014490</td>\n",
       "      <td>0.079541</td>\n",
       "      <td>-0.002427</td>\n",
       "      <td>0.079405</td>\n",
       "      <td>-0.002668</td>\n",
       "      <td>0.127264</td>\n",
       "      <td>-0.001514</td>\n",
       "      <td>-0.025922</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htcdispos</td>\n",
       "      <td>0.007125</td>\n",
       "      <td>0.024839</td>\n",
       "      <td>0.003299</td>\n",
       "      <td>0.010554</td>\n",
       "      <td>0.977538</td>\n",
       "      <td>-0.005749</td>\n",
       "      <td>0.057552</td>\n",
       "      <td>0.067429</td>\n",
       "      <td>0.032923</td>\n",
       "      <td>0.016457</td>\n",
       "      <td>...</td>\n",
       "      <td>0.015451</td>\n",
       "      <td>0.005317</td>\n",
       "      <td>0.135495</td>\n",
       "      <td>-0.001147</td>\n",
       "      <td>0.118340</td>\n",
       "      <td>-0.001013</td>\n",
       "      <td>0.109239</td>\n",
       "      <td>0.000579</td>\n",
       "      <td>0.124018</td>\n",
       "      <td>-0.060406</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphonedisneg</td>\n",
       "      <td>0.175573</td>\n",
       "      <td>0.017824</td>\n",
       "      <td>-0.013590</td>\n",
       "      <td>0.023742</td>\n",
       "      <td>0.002796</td>\n",
       "      <td>0.113784</td>\n",
       "      <td>0.121821</td>\n",
       "      <td>0.148651</td>\n",
       "      <td>0.065279</td>\n",
       "      <td>0.006717</td>\n",
       "      <td>...</td>\n",
       "      <td>0.023878</td>\n",
       "      <td>0.009723</td>\n",
       "      <td>0.096514</td>\n",
       "      <td>0.015557</td>\n",
       "      <td>0.218541</td>\n",
       "      <td>0.016863</td>\n",
       "      <td>0.213640</td>\n",
       "      <td>0.018222</td>\n",
       "      <td>0.204416</td>\n",
       "      <td>0.003145</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsungdisneg</td>\n",
       "      <td>0.111821</td>\n",
       "      <td>0.304385</td>\n",
       "      <td>0.007706</td>\n",
       "      <td>0.022910</td>\n",
       "      <td>0.020154</td>\n",
       "      <td>-0.011706</td>\n",
       "      <td>0.542440</td>\n",
       "      <td>0.089813</td>\n",
       "      <td>0.487871</td>\n",
       "      <td>0.058932</td>\n",
       "      <td>...</td>\n",
       "      <td>0.108932</td>\n",
       "      <td>0.072670</td>\n",
       "      <td>0.317065</td>\n",
       "      <td>0.104504</td>\n",
       "      <td>0.808753</td>\n",
       "      <td>0.081942</td>\n",
       "      <td>0.826604</td>\n",
       "      <td>0.050730</td>\n",
       "      <td>0.735526</td>\n",
       "      <td>-0.139965</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonydisneg</td>\n",
       "      <td>-0.002777</td>\n",
       "      <td>0.006786</td>\n",
       "      <td>0.163285</td>\n",
       "      <td>-0.000644</td>\n",
       "      <td>0.003056</td>\n",
       "      <td>0.008063</td>\n",
       "      <td>0.000215</td>\n",
       "      <td>0.002884</td>\n",
       "      <td>0.011236</td>\n",
       "      <td>0.131892</td>\n",
       "      <td>...</td>\n",
       "      <td>0.112633</td>\n",
       "      <td>-0.000431</td>\n",
       "      <td>0.006683</td>\n",
       "      <td>0.030273</td>\n",
       "      <td>0.000580</td>\n",
       "      <td>0.029531</td>\n",
       "      <td>0.001907</td>\n",
       "      <td>-0.001501</td>\n",
       "      <td>-0.000803</td>\n",
       "      <td>-0.019956</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokiadisneg</td>\n",
       "      <td>-0.008790</td>\n",
       "      <td>0.005640</td>\n",
       "      <td>-0.004040</td>\n",
       "      <td>0.692268</td>\n",
       "      <td>0.000389</td>\n",
       "      <td>0.003432</td>\n",
       "      <td>-0.004470</td>\n",
       "      <td>0.026307</td>\n",
       "      <td>0.028454</td>\n",
       "      <td>-0.001752</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000766</td>\n",
       "      <td>0.861817</td>\n",
       "      <td>0.017639</td>\n",
       "      <td>0.087180</td>\n",
       "      <td>-0.002599</td>\n",
       "      <td>0.086986</td>\n",
       "      <td>-0.002857</td>\n",
       "      <td>0.139410</td>\n",
       "      <td>-0.001622</td>\n",
       "      <td>-0.028759</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htcdisneg</td>\n",
       "      <td>0.085273</td>\n",
       "      <td>0.188821</td>\n",
       "      <td>-0.002138</td>\n",
       "      <td>0.044222</td>\n",
       "      <td>0.037653</td>\n",
       "      <td>-0.019709</td>\n",
       "      <td>0.447013</td>\n",
       "      <td>0.110102</td>\n",
       "      <td>0.238425</td>\n",
       "      <td>0.037624</td>\n",
       "      <td>...</td>\n",
       "      <td>0.068252</td>\n",
       "      <td>0.023569</td>\n",
       "      <td>0.549764</td>\n",
       "      <td>0.000258</td>\n",
       "      <td>0.704117</td>\n",
       "      <td>0.001073</td>\n",
       "      <td>0.698363</td>\n",
       "      <td>0.009817</td>\n",
       "      <td>0.643174</td>\n",
       "      <td>-0.192727</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphonedisunc</td>\n",
       "      <td>0.250930</td>\n",
       "      <td>-0.027879</td>\n",
       "      <td>-0.017981</td>\n",
       "      <td>0.002681</td>\n",
       "      <td>-0.002108</td>\n",
       "      <td>0.218835</td>\n",
       "      <td>0.017791</td>\n",
       "      <td>0.188310</td>\n",
       "      <td>0.012313</td>\n",
       "      <td>0.007384</td>\n",
       "      <td>...</td>\n",
       "      <td>0.022878</td>\n",
       "      <td>0.001485</td>\n",
       "      <td>0.092743</td>\n",
       "      <td>0.024055</td>\n",
       "      <td>0.132862</td>\n",
       "      <td>0.023660</td>\n",
       "      <td>0.106084</td>\n",
       "      <td>0.030107</td>\n",
       "      <td>0.172276</td>\n",
       "      <td>0.027173</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsungdisunc</td>\n",
       "      <td>0.038727</td>\n",
       "      <td>0.190038</td>\n",
       "      <td>0.026314</td>\n",
       "      <td>0.046896</td>\n",
       "      <td>0.009413</td>\n",
       "      <td>-0.005951</td>\n",
       "      <td>0.188924</td>\n",
       "      <td>0.035791</td>\n",
       "      <td>0.389689</td>\n",
       "      <td>0.089489</td>\n",
       "      <td>...</td>\n",
       "      <td>0.129397</td>\n",
       "      <td>0.136216</td>\n",
       "      <td>0.344839</td>\n",
       "      <td>0.078623</td>\n",
       "      <td>0.594372</td>\n",
       "      <td>0.063341</td>\n",
       "      <td>0.512732</td>\n",
       "      <td>0.039951</td>\n",
       "      <td>0.738457</td>\n",
       "      <td>-0.059548</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonydisunc</td>\n",
       "      <td>-0.004553</td>\n",
       "      <td>0.060556</td>\n",
       "      <td>0.295428</td>\n",
       "      <td>-0.001383</td>\n",
       "      <td>0.005268</td>\n",
       "      <td>0.000626</td>\n",
       "      <td>-0.004753</td>\n",
       "      <td>0.019403</td>\n",
       "      <td>0.067668</td>\n",
       "      <td>0.388804</td>\n",
       "      <td>...</td>\n",
       "      <td>0.476597</td>\n",
       "      <td>-0.000927</td>\n",
       "      <td>0.037009</td>\n",
       "      <td>0.014600</td>\n",
       "      <td>-0.002764</td>\n",
       "      <td>0.014310</td>\n",
       "      <td>-0.003038</td>\n",
       "      <td>-0.003225</td>\n",
       "      <td>-0.001724</td>\n",
       "      <td>-0.032137</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokiadisunc</td>\n",
       "      <td>-0.007588</td>\n",
       "      <td>0.014661</td>\n",
       "      <td>-0.003233</td>\n",
       "      <td>0.491332</td>\n",
       "      <td>-0.000066</td>\n",
       "      <td>0.006110</td>\n",
       "      <td>-0.003577</td>\n",
       "      <td>0.009608</td>\n",
       "      <td>0.046812</td>\n",
       "      <td>-0.001402</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000613</td>\n",
       "      <td>0.923934</td>\n",
       "      <td>0.007326</td>\n",
       "      <td>0.102037</td>\n",
       "      <td>-0.002080</td>\n",
       "      <td>0.102012</td>\n",
       "      <td>-0.002286</td>\n",
       "      <td>0.162690</td>\n",
       "      <td>-0.001298</td>\n",
       "      <td>-0.023972</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htcdisunc</td>\n",
       "      <td>0.024322</td>\n",
       "      <td>0.071746</td>\n",
       "      <td>0.010003</td>\n",
       "      <td>0.021114</td>\n",
       "      <td>0.029195</td>\n",
       "      <td>-0.012652</td>\n",
       "      <td>0.147068</td>\n",
       "      <td>0.156063</td>\n",
       "      <td>0.086766</td>\n",
       "      <td>0.055055</td>\n",
       "      <td>...</td>\n",
       "      <td>0.083275</td>\n",
       "      <td>0.010655</td>\n",
       "      <td>0.721407</td>\n",
       "      <td>0.006433</td>\n",
       "      <td>0.483030</td>\n",
       "      <td>0.006752</td>\n",
       "      <td>0.406569</td>\n",
       "      <td>0.019677</td>\n",
       "      <td>0.593494</td>\n",
       "      <td>-0.132953</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphoneperpos</td>\n",
       "      <td>-0.009508</td>\n",
       "      <td>-0.003169</td>\n",
       "      <td>-0.028717</td>\n",
       "      <td>0.033345</td>\n",
       "      <td>0.000121</td>\n",
       "      <td>-0.021953</td>\n",
       "      <td>0.106061</td>\n",
       "      <td>0.348332</td>\n",
       "      <td>0.056272</td>\n",
       "      <td>0.009152</td>\n",
       "      <td>...</td>\n",
       "      <td>0.036380</td>\n",
       "      <td>0.015281</td>\n",
       "      <td>0.123390</td>\n",
       "      <td>0.210343</td>\n",
       "      <td>0.240267</td>\n",
       "      <td>0.224650</td>\n",
       "      <td>0.218848</td>\n",
       "      <td>0.211809</td>\n",
       "      <td>0.237625</td>\n",
       "      <td>0.029638</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsungperpos</td>\n",
       "      <td>0.051538</td>\n",
       "      <td>0.242866</td>\n",
       "      <td>0.020914</td>\n",
       "      <td>0.017459</td>\n",
       "      <td>0.009711</td>\n",
       "      <td>-0.002131</td>\n",
       "      <td>0.270355</td>\n",
       "      <td>0.045221</td>\n",
       "      <td>0.793899</td>\n",
       "      <td>0.046923</td>\n",
       "      <td>...</td>\n",
       "      <td>0.057896</td>\n",
       "      <td>0.055507</td>\n",
       "      <td>0.189565</td>\n",
       "      <td>0.274209</td>\n",
       "      <td>0.444302</td>\n",
       "      <td>0.214228</td>\n",
       "      <td>0.441229</td>\n",
       "      <td>0.137057</td>\n",
       "      <td>0.427542</td>\n",
       "      <td>-0.081063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonyperpos</td>\n",
       "      <td>-0.006327</td>\n",
       "      <td>0.067489</td>\n",
       "      <td>0.266142</td>\n",
       "      <td>-0.001919</td>\n",
       "      <td>0.004812</td>\n",
       "      <td>-0.004091</td>\n",
       "      <td>0.000836</td>\n",
       "      <td>0.013944</td>\n",
       "      <td>0.047395</td>\n",
       "      <td>0.387311</td>\n",
       "      <td>...</td>\n",
       "      <td>0.735802</td>\n",
       "      <td>-0.001285</td>\n",
       "      <td>0.014073</td>\n",
       "      <td>0.005758</td>\n",
       "      <td>0.007712</td>\n",
       "      <td>0.005731</td>\n",
       "      <td>0.008070</td>\n",
       "      <td>-0.004473</td>\n",
       "      <td>-0.002392</td>\n",
       "      <td>-0.038913</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokiaperpos</td>\n",
       "      <td>-0.010509</td>\n",
       "      <td>0.001846</td>\n",
       "      <td>-0.004606</td>\n",
       "      <td>0.737457</td>\n",
       "      <td>0.000454</td>\n",
       "      <td>0.002261</td>\n",
       "      <td>-0.002300</td>\n",
       "      <td>0.021178</td>\n",
       "      <td>0.021581</td>\n",
       "      <td>-0.001998</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000873</td>\n",
       "      <td>0.917333</td>\n",
       "      <td>0.016373</td>\n",
       "      <td>0.084948</td>\n",
       "      <td>0.002051</td>\n",
       "      <td>0.084529</td>\n",
       "      <td>-0.000824</td>\n",
       "      <td>0.135942</td>\n",
       "      <td>0.003141</td>\n",
       "      <td>-0.041595</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htcperpos</td>\n",
       "      <td>0.030621</td>\n",
       "      <td>0.088289</td>\n",
       "      <td>0.004677</td>\n",
       "      <td>0.039113</td>\n",
       "      <td>0.030909</td>\n",
       "      <td>-0.018167</td>\n",
       "      <td>0.209414</td>\n",
       "      <td>0.287085</td>\n",
       "      <td>0.115132</td>\n",
       "      <td>0.021326</td>\n",
       "      <td>...</td>\n",
       "      <td>0.026076</td>\n",
       "      <td>0.020553</td>\n",
       "      <td>0.849739</td>\n",
       "      <td>-0.002803</td>\n",
       "      <td>0.380278</td>\n",
       "      <td>-0.002767</td>\n",
       "      <td>0.358458</td>\n",
       "      <td>-0.000189</td>\n",
       "      <td>0.368327</td>\n",
       "      <td>-0.178427</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphoneperneg</td>\n",
       "      <td>0.013863</td>\n",
       "      <td>0.045963</td>\n",
       "      <td>-0.028774</td>\n",
       "      <td>0.033735</td>\n",
       "      <td>0.004285</td>\n",
       "      <td>-0.012566</td>\n",
       "      <td>0.212525</td>\n",
       "      <td>0.151919</td>\n",
       "      <td>0.112508</td>\n",
       "      <td>0.006280</td>\n",
       "      <td>...</td>\n",
       "      <td>0.042156</td>\n",
       "      <td>0.015347</td>\n",
       "      <td>0.140184</td>\n",
       "      <td>0.247457</td>\n",
       "      <td>0.345247</td>\n",
       "      <td>0.282779</td>\n",
       "      <td>0.348685</td>\n",
       "      <td>0.255736</td>\n",
       "      <td>0.296227</td>\n",
       "      <td>-0.004804</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsungperneg</td>\n",
       "      <td>0.115130</td>\n",
       "      <td>0.303560</td>\n",
       "      <td>-0.001931</td>\n",
       "      <td>0.017354</td>\n",
       "      <td>0.017457</td>\n",
       "      <td>-0.007168</td>\n",
       "      <td>0.558090</td>\n",
       "      <td>0.092030</td>\n",
       "      <td>0.546670</td>\n",
       "      <td>0.034149</td>\n",
       "      <td>...</td>\n",
       "      <td>0.060809</td>\n",
       "      <td>0.057204</td>\n",
       "      <td>0.271245</td>\n",
       "      <td>0.202892</td>\n",
       "      <td>0.758411</td>\n",
       "      <td>0.161560</td>\n",
       "      <td>0.796365</td>\n",
       "      <td>0.103158</td>\n",
       "      <td>0.641229</td>\n",
       "      <td>-0.138657</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonyperneg</td>\n",
       "      <td>-0.003625</td>\n",
       "      <td>0.009977</td>\n",
       "      <td>0.122407</td>\n",
       "      <td>-0.000948</td>\n",
       "      <td>0.001113</td>\n",
       "      <td>-0.002902</td>\n",
       "      <td>0.005657</td>\n",
       "      <td>0.007034</td>\n",
       "      <td>0.019366</td>\n",
       "      <td>0.182829</td>\n",
       "      <td>...</td>\n",
       "      <td>0.668018</td>\n",
       "      <td>-0.000635</td>\n",
       "      <td>0.006295</td>\n",
       "      <td>0.000040</td>\n",
       "      <td>0.010539</td>\n",
       "      <td>0.000099</td>\n",
       "      <td>0.012140</td>\n",
       "      <td>-0.002210</td>\n",
       "      <td>-0.001182</td>\n",
       "      <td>-0.030850</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokiaperneg</td>\n",
       "      <td>-0.010781</td>\n",
       "      <td>0.000481</td>\n",
       "      <td>-0.004699</td>\n",
       "      <td>0.736453</td>\n",
       "      <td>0.000462</td>\n",
       "      <td>0.002322</td>\n",
       "      <td>-0.003226</td>\n",
       "      <td>0.017987</td>\n",
       "      <td>0.018696</td>\n",
       "      <td>-0.002038</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000891</td>\n",
       "      <td>0.905222</td>\n",
       "      <td>0.017733</td>\n",
       "      <td>0.079553</td>\n",
       "      <td>0.000515</td>\n",
       "      <td>0.079819</td>\n",
       "      <td>-0.001606</td>\n",
       "      <td>0.128002</td>\n",
       "      <td>0.001635</td>\n",
       "      <td>-0.044219</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htcperneg</td>\n",
       "      <td>0.075975</td>\n",
       "      <td>0.178410</td>\n",
       "      <td>-0.012083</td>\n",
       "      <td>0.050051</td>\n",
       "      <td>0.033942</td>\n",
       "      <td>-0.021576</td>\n",
       "      <td>0.433411</td>\n",
       "      <td>0.109392</td>\n",
       "      <td>0.231172</td>\n",
       "      <td>0.009013</td>\n",
       "      <td>...</td>\n",
       "      <td>0.021705</td>\n",
       "      <td>0.026814</td>\n",
       "      <td>0.659652</td>\n",
       "      <td>-0.003590</td>\n",
       "      <td>0.628876</td>\n",
       "      <td>-0.002524</td>\n",
       "      <td>0.638941</td>\n",
       "      <td>0.000362</td>\n",
       "      <td>0.539902</td>\n",
       "      <td>-0.209196</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphoneperunc</td>\n",
       "      <td>-0.016037</td>\n",
       "      <td>-0.017389</td>\n",
       "      <td>-0.028220</td>\n",
       "      <td>0.020197</td>\n",
       "      <td>0.000194</td>\n",
       "      <td>-0.015482</td>\n",
       "      <td>0.056676</td>\n",
       "      <td>0.187260</td>\n",
       "      <td>0.031845</td>\n",
       "      <td>0.008176</td>\n",
       "      <td>...</td>\n",
       "      <td>0.050653</td>\n",
       "      <td>0.012553</td>\n",
       "      <td>0.171436</td>\n",
       "      <td>0.166660</td>\n",
       "      <td>0.242735</td>\n",
       "      <td>0.179411</td>\n",
       "      <td>0.196254</td>\n",
       "      <td>0.181783</td>\n",
       "      <td>0.297140</td>\n",
       "      <td>0.037200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>samsungperunc</td>\n",
       "      <td>0.046822</td>\n",
       "      <td>0.184775</td>\n",
       "      <td>0.008008</td>\n",
       "      <td>0.035274</td>\n",
       "      <td>0.010644</td>\n",
       "      <td>-0.004770</td>\n",
       "      <td>0.221726</td>\n",
       "      <td>0.040154</td>\n",
       "      <td>0.487767</td>\n",
       "      <td>0.053436</td>\n",
       "      <td>...</td>\n",
       "      <td>0.091928</td>\n",
       "      <td>0.103767</td>\n",
       "      <td>0.346705</td>\n",
       "      <td>0.102804</td>\n",
       "      <td>0.616442</td>\n",
       "      <td>0.083218</td>\n",
       "      <td>0.541504</td>\n",
       "      <td>0.053897</td>\n",
       "      <td>0.739887</td>\n",
       "      <td>-0.057920</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>sonyperunc</td>\n",
       "      <td>-0.003045</td>\n",
       "      <td>0.037482</td>\n",
       "      <td>0.151675</td>\n",
       "      <td>-0.001204</td>\n",
       "      <td>0.005018</td>\n",
       "      <td>-0.004832</td>\n",
       "      <td>-0.004135</td>\n",
       "      <td>0.019987</td>\n",
       "      <td>0.057860</td>\n",
       "      <td>0.378812</td>\n",
       "      <td>...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.000806</td>\n",
       "      <td>0.033233</td>\n",
       "      <td>-0.002861</td>\n",
       "      <td>-0.002405</td>\n",
       "      <td>-0.002711</td>\n",
       "      <td>-0.002643</td>\n",
       "      <td>-0.002806</td>\n",
       "      <td>-0.001500</td>\n",
       "      <td>-0.018084</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>nokiaperunc</td>\n",
       "      <td>-0.009704</td>\n",
       "      <td>0.007305</td>\n",
       "      <td>-0.004253</td>\n",
       "      <td>0.648441</td>\n",
       "      <td>0.000112</td>\n",
       "      <td>0.005030</td>\n",
       "      <td>-0.001407</td>\n",
       "      <td>0.014827</td>\n",
       "      <td>0.033197</td>\n",
       "      <td>-0.001845</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000806</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.012363</td>\n",
       "      <td>0.098336</td>\n",
       "      <td>0.003180</td>\n",
       "      <td>0.098859</td>\n",
       "      <td>-0.000137</td>\n",
       "      <td>0.157714</td>\n",
       "      <td>0.004180</td>\n",
       "      <td>-0.036167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>htcperunc</td>\n",
       "      <td>0.011414</td>\n",
       "      <td>0.044928</td>\n",
       "      <td>-0.004888</td>\n",
       "      <td>0.023757</td>\n",
       "      <td>0.021448</td>\n",
       "      <td>-0.011930</td>\n",
       "      <td>0.109685</td>\n",
       "      <td>0.067283</td>\n",
       "      <td>0.061304</td>\n",
       "      <td>0.015781</td>\n",
       "      <td>...</td>\n",
       "      <td>0.033233</td>\n",
       "      <td>0.012363</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000969</td>\n",
       "      <td>0.333022</td>\n",
       "      <td>0.000673</td>\n",
       "      <td>0.280893</td>\n",
       "      <td>0.008437</td>\n",
       "      <td>0.394552</td>\n",
       "      <td>-0.114171</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iosperpos</td>\n",
       "      <td>-0.020059</td>\n",
       "      <td>-0.005802</td>\n",
       "      <td>-0.011009</td>\n",
       "      <td>0.030719</td>\n",
       "      <td>-0.002927</td>\n",
       "      <td>0.118278</td>\n",
       "      <td>-0.016702</td>\n",
       "      <td>-0.003991</td>\n",
       "      <td>0.102471</td>\n",
       "      <td>-0.003118</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.002861</td>\n",
       "      <td>0.098336</td>\n",
       "      <td>0.000969</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.009712</td>\n",
       "      <td>0.932382</td>\n",
       "      <td>-0.010676</td>\n",
       "      <td>0.905079</td>\n",
       "      <td>-0.006060</td>\n",
       "      <td>-0.015758</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>googleperpos</td>\n",
       "      <td>0.118008</td>\n",
       "      <td>0.246046</td>\n",
       "      <td>-0.008467</td>\n",
       "      <td>0.006515</td>\n",
       "      <td>0.019186</td>\n",
       "      <td>-0.016402</td>\n",
       "      <td>0.638581</td>\n",
       "      <td>0.117902</td>\n",
       "      <td>0.298281</td>\n",
       "      <td>0.006673</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.002405</td>\n",
       "      <td>0.003180</td>\n",
       "      <td>0.333022</td>\n",
       "      <td>-0.009712</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.009203</td>\n",
       "      <td>0.957410</td>\n",
       "      <td>-0.009524</td>\n",
       "      <td>0.887033</td>\n",
       "      <td>-0.137261</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iosperneg</td>\n",
       "      <td>-0.019081</td>\n",
       "      <td>-0.007839</td>\n",
       "      <td>-0.010323</td>\n",
       "      <td>0.032721</td>\n",
       "      <td>-0.002758</td>\n",
       "      <td>0.112330</td>\n",
       "      <td>-0.015825</td>\n",
       "      <td>-0.007060</td>\n",
       "      <td>0.075695</td>\n",
       "      <td>-0.002863</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.002711</td>\n",
       "      <td>0.098859</td>\n",
       "      <td>0.000673</td>\n",
       "      <td>0.932382</td>\n",
       "      <td>-0.009203</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.010115</td>\n",
       "      <td>0.899819</td>\n",
       "      <td>-0.005742</td>\n",
       "      <td>-0.010179</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>googleperneg</td>\n",
       "      <td>0.138742</td>\n",
       "      <td>0.290975</td>\n",
       "      <td>-0.008570</td>\n",
       "      <td>0.000653</td>\n",
       "      <td>0.020726</td>\n",
       "      <td>-0.018028</td>\n",
       "      <td>0.716515</td>\n",
       "      <td>0.124355</td>\n",
       "      <td>0.357362</td>\n",
       "      <td>0.008455</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.002643</td>\n",
       "      <td>-0.000137</td>\n",
       "      <td>0.280893</td>\n",
       "      <td>-0.010676</td>\n",
       "      <td>0.957410</td>\n",
       "      <td>-0.010115</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.010468</td>\n",
       "      <td>0.756118</td>\n",
       "      <td>-0.163919</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iosperunc</td>\n",
       "      <td>-0.020368</td>\n",
       "      <td>-0.015329</td>\n",
       "      <td>-0.014802</td>\n",
       "      <td>0.052887</td>\n",
       "      <td>-0.002666</td>\n",
       "      <td>0.117035</td>\n",
       "      <td>-0.016377</td>\n",
       "      <td>-0.001037</td>\n",
       "      <td>0.044890</td>\n",
       "      <td>-0.006421</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.002806</td>\n",
       "      <td>0.157714</td>\n",
       "      <td>0.008437</td>\n",
       "      <td>0.905079</td>\n",
       "      <td>-0.009524</td>\n",
       "      <td>0.899819</td>\n",
       "      <td>-0.010468</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.005942</td>\n",
       "      <td>-0.011787</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>googleperunc</td>\n",
       "      <td>0.067859</td>\n",
       "      <td>0.142252</td>\n",
       "      <td>-0.007916</td>\n",
       "      <td>0.007999</td>\n",
       "      <td>0.013305</td>\n",
       "      <td>-0.010233</td>\n",
       "      <td>0.371998</td>\n",
       "      <td>0.073004</td>\n",
       "      <td>0.159171</td>\n",
       "      <td>-0.003434</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.001500</td>\n",
       "      <td>0.004180</td>\n",
       "      <td>0.394552</td>\n",
       "      <td>-0.006060</td>\n",
       "      <td>0.887033</td>\n",
       "      <td>-0.005742</td>\n",
       "      <td>0.756118</td>\n",
       "      <td>-0.005942</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.070284</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>iphonesentiment</td>\n",
       "      <td>0.014859</td>\n",
       "      <td>-0.359173</td>\n",
       "      <td>-0.233170</td>\n",
       "      <td>-0.055962</td>\n",
       "      <td>-0.051285</td>\n",
       "      <td>0.001656</td>\n",
       "      <td>-0.189142</td>\n",
       "      <td>-0.029731</td>\n",
       "      <td>-0.112743</td>\n",
       "      <td>-0.090665</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018084</td>\n",
       "      <td>-0.036167</td>\n",
       "      <td>-0.114171</td>\n",
       "      <td>-0.015758</td>\n",
       "      <td>-0.137261</td>\n",
       "      <td>-0.010179</td>\n",
       "      <td>-0.163919</td>\n",
       "      <td>-0.011787</td>\n",
       "      <td>-0.070284</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>59 rows × 59 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                   iphone  samsunggalaxy  sonyxperia  nokialumina  htcphone  \\\n",
       "iphone           1.000000       0.019786   -0.011618    -0.013423 -0.002731   \n",
       "samsunggalaxy    0.019786       1.000000    0.366671    -0.006088  0.017899   \n",
       "sonyxperia      -0.011618       0.366671    1.000000    -0.006350  0.023682   \n",
       "nokialumina     -0.013423      -0.006088   -0.006350     1.000000  0.000673   \n",
       "htcphone        -0.002731       0.017899    0.023682     0.000673  1.000000   \n",
       "ios              0.922060      -0.044678   -0.023884    -0.002819 -0.005002   \n",
       "googleandroid    0.107530       0.236162   -0.018288    -0.001115  0.016498   \n",
       "iphonecampos     0.078157       0.030556    0.005068     0.029824  0.006952   \n",
       "samsungcampos    0.057395       0.252121    0.050140     0.009299  0.010865   \n",
       "sonycampos      -0.004594       0.145969    0.396751    -0.002754  0.010432   \n",
       "nokiacampos     -0.008439      -0.000400   -0.004232     0.700415  0.000465   \n",
       "htccampos        0.022717       0.065274    0.016507     0.021295  0.023189   \n",
       "iphonecamneg     0.490524       0.126063   -0.006715     0.063245  0.014155   \n",
       "samsungcamneg    0.142553       0.342919   -0.004308     0.009546  0.020021   \n",
       "sonycamneg      -0.001830       0.031821    0.345296    -0.001229  0.004909   \n",
       "nokiacamneg     -0.009186      -0.000979   -0.004467     0.729434  0.000191   \n",
       "htccamneg        0.104613       0.222777   -0.012284     0.037256  0.036765   \n",
       "iphonecamunc     0.750403      -0.010155   -0.007638     0.016237  0.001174   \n",
       "samsungcamunc    0.073451       0.316134    0.058777     0.040922  0.015644   \n",
       "sonycamunc      -0.003064       0.104123    0.376633    -0.001914  0.009843   \n",
       "nokiacamunc     -0.008602       0.005691   -0.003972     0.634171  0.000364   \n",
       "htccamunc        0.026138       0.072964    0.014249     0.036124  0.029152   \n",
       "iphonedispos     0.052625      -0.006526   -0.018121     0.028316  0.000253   \n",
       "samsungdispos    0.061074       0.281379    0.040063     0.041456  0.013145   \n",
       "sonydispos      -0.003827       0.061360    0.252589    -0.001528  0.006959   \n",
       "nokiadispos     -0.008202       0.010248   -0.003772     0.650253  0.000311   \n",
       "htcdispos        0.007125       0.024839    0.003299     0.010554  0.977538   \n",
       "iphonedisneg     0.175573       0.017824   -0.013590     0.023742  0.002796   \n",
       "samsungdisneg    0.111821       0.304385    0.007706     0.022910  0.020154   \n",
       "sonydisneg      -0.002777       0.006786    0.163285    -0.000644  0.003056   \n",
       "nokiadisneg     -0.008790       0.005640   -0.004040     0.692268  0.000389   \n",
       "htcdisneg        0.085273       0.188821   -0.002138     0.044222  0.037653   \n",
       "iphonedisunc     0.250930      -0.027879   -0.017981     0.002681 -0.002108   \n",
       "samsungdisunc    0.038727       0.190038    0.026314     0.046896  0.009413   \n",
       "sonydisunc      -0.004553       0.060556    0.295428    -0.001383  0.005268   \n",
       "nokiadisunc     -0.007588       0.014661   -0.003233     0.491332 -0.000066   \n",
       "htcdisunc        0.024322       0.071746    0.010003     0.021114  0.029195   \n",
       "iphoneperpos    -0.009508      -0.003169   -0.028717     0.033345  0.000121   \n",
       "samsungperpos    0.051538       0.242866    0.020914     0.017459  0.009711   \n",
       "sonyperpos      -0.006327       0.067489    0.266142    -0.001919  0.004812   \n",
       "nokiaperpos     -0.010509       0.001846   -0.004606     0.737457  0.000454   \n",
       "htcperpos        0.030621       0.088289    0.004677     0.039113  0.030909   \n",
       "iphoneperneg     0.013863       0.045963   -0.028774     0.033735  0.004285   \n",
       "samsungperneg    0.115130       0.303560   -0.001931     0.017354  0.017457   \n",
       "sonyperneg      -0.003625       0.009977    0.122407    -0.000948  0.001113   \n",
       "nokiaperneg     -0.010781       0.000481   -0.004699     0.736453  0.000462   \n",
       "htcperneg        0.075975       0.178410   -0.012083     0.050051  0.033942   \n",
       "iphoneperunc    -0.016037      -0.017389   -0.028220     0.020197  0.000194   \n",
       "samsungperunc    0.046822       0.184775    0.008008     0.035274  0.010644   \n",
       "sonyperunc      -0.003045       0.037482    0.151675    -0.001204  0.005018   \n",
       "nokiaperunc     -0.009704       0.007305   -0.004253     0.648441  0.000112   \n",
       "htcperunc        0.011414       0.044928   -0.004888     0.023757  0.021448   \n",
       "iosperpos       -0.020059      -0.005802   -0.011009     0.030719 -0.002927   \n",
       "googleperpos     0.118008       0.246046   -0.008467     0.006515  0.019186   \n",
       "iosperneg       -0.019081      -0.007839   -0.010323     0.032721 -0.002758   \n",
       "googleperneg     0.138742       0.290975   -0.008570     0.000653  0.020726   \n",
       "iosperunc       -0.020368      -0.015329   -0.014802     0.052887 -0.002666   \n",
       "googleperunc     0.067859       0.142252   -0.007916     0.007999  0.013305   \n",
       "iphonesentiment  0.014859      -0.359173   -0.233170    -0.055962 -0.051285   \n",
       "\n",
       "                      ios  googleandroid  iphonecampos  samsungcampos  \\\n",
       "iphone           0.922060       0.107530      0.078157       0.057395   \n",
       "samsunggalaxy   -0.044678       0.236162      0.030556       0.252121   \n",
       "sonyxperia      -0.023884      -0.018288      0.005068       0.050140   \n",
       "nokialumina     -0.002819      -0.001115      0.029824       0.009299   \n",
       "htcphone        -0.005002       0.016498      0.006952       0.010865   \n",
       "ios              1.000000      -0.026404      0.042128      -0.010741   \n",
       "googleandroid   -0.026404       1.000000      0.104420       0.315487   \n",
       "iphonecampos     0.042128       0.104420      1.000000       0.062438   \n",
       "samsungcampos   -0.010741       0.315487      0.062438       1.000000   \n",
       "sonycampos      -0.009369      -0.000206      0.045009       0.145429   \n",
       "nokiacampos      0.005425       0.003284      0.030817       0.014860   \n",
       "htccampos       -0.012390       0.148095      0.623912       0.090099   \n",
       "iphonecamneg     0.386966       0.391802      0.541340       0.206020   \n",
       "samsungcamneg   -0.015273       0.711403      0.117451       0.608840   \n",
       "sonycamneg      -0.003854       0.013539      0.019994       0.053985   \n",
       "nokiacamneg      0.004651      -0.001824      0.026855       0.014368   \n",
       "htccamneg       -0.023049       0.562703      0.206585       0.295428   \n",
       "iphonecamunc     0.732612       0.042955      0.473266       0.028875   \n",
       "samsungcamunc   -0.012390       0.391433      0.076943       0.814799   \n",
       "sonycamunc      -0.006484      -0.006578      0.029397       0.098836   \n",
       "nokiacamunc      0.006341       0.000325      0.021277       0.028323   \n",
       "htccamunc       -0.015785       0.166182      0.321523       0.104495   \n",
       "iphonedispos     0.014377       0.066953      0.272587       0.039427   \n",
       "samsungdispos   -0.009906       0.316132      0.060476       0.643692   \n",
       "sonydispos       0.004207      -0.001669      0.017749       0.058122   \n",
       "nokiadispos      0.003065      -0.004174      0.026317       0.038371   \n",
       "htcdispos       -0.005749       0.057552      0.067429       0.032923   \n",
       "iphonedisneg     0.113784       0.121821      0.148651       0.065279   \n",
       "samsungdisneg   -0.011706       0.542440      0.089813       0.487871   \n",
       "sonydisneg       0.008063       0.000215      0.002884       0.011236   \n",
       "nokiadisneg      0.003432      -0.004470      0.026307       0.028454   \n",
       "htcdisneg       -0.019709       0.447013      0.110102       0.238425   \n",
       "iphonedisunc     0.218835       0.017791      0.188310       0.012313   \n",
       "samsungdisunc   -0.005951       0.188924      0.035791       0.389689   \n",
       "sonydisunc       0.000626      -0.004753      0.019403       0.067668   \n",
       "nokiadisunc      0.006110      -0.003577      0.009608       0.046812   \n",
       "htcdisunc       -0.012652       0.147068      0.156063       0.086766   \n",
       "iphoneperpos    -0.021953       0.106061      0.348332       0.056272   \n",
       "samsungperpos   -0.002131       0.270355      0.045221       0.793899   \n",
       "sonyperpos      -0.004091       0.000836      0.013944       0.047395   \n",
       "nokiaperpos      0.002261      -0.002300      0.021178       0.021581   \n",
       "htcperpos       -0.018167       0.209414      0.287085       0.115132   \n",
       "iphoneperneg    -0.012566       0.212525      0.151919       0.112508   \n",
       "samsungperneg   -0.007168       0.558090      0.092030       0.546670   \n",
       "sonyperneg      -0.002902       0.005657      0.007034       0.019366   \n",
       "nokiaperneg      0.002322      -0.003226      0.017987       0.018696   \n",
       "htcperneg       -0.021576       0.433411      0.109392       0.231172   \n",
       "iphoneperunc    -0.015482       0.056676      0.187260       0.031845   \n",
       "samsungperunc   -0.004770       0.221726      0.040154       0.487767   \n",
       "sonyperunc      -0.004832      -0.004135      0.019987       0.057860   \n",
       "nokiaperunc      0.005030      -0.001407      0.014827       0.033197   \n",
       "htcperunc       -0.011930       0.109685      0.067283       0.061304   \n",
       "iosperpos        0.118278      -0.016702     -0.003991       0.102471   \n",
       "googleperpos    -0.016402       0.638581      0.117902       0.298281   \n",
       "iosperneg        0.112330      -0.015825     -0.007060       0.075695   \n",
       "googleperneg    -0.018028       0.716515      0.124355       0.357362   \n",
       "iosperunc        0.117035      -0.016377     -0.001037       0.044890   \n",
       "googleperunc    -0.010233       0.371998      0.073004       0.159171   \n",
       "iphonesentiment  0.001656      -0.189142     -0.029731      -0.112743   \n",
       "\n",
       "                 sonycampos  ...  sonyperunc  nokiaperunc  htcperunc  \\\n",
       "iphone            -0.004594  ...   -0.003045    -0.009704   0.011414   \n",
       "samsunggalaxy      0.145969  ...    0.037482     0.007305   0.044928   \n",
       "sonyxperia         0.396751  ...    0.151675    -0.004253  -0.004888   \n",
       "nokialumina       -0.002754  ...   -0.001204     0.648441   0.023757   \n",
       "htcphone           0.010432  ...    0.005018     0.000112   0.021448   \n",
       "ios               -0.009369  ...   -0.004832     0.005030  -0.011930   \n",
       "googleandroid     -0.000206  ...   -0.004135    -0.001407   0.109685   \n",
       "iphonecampos       0.045009  ...    0.019987     0.014827   0.067283   \n",
       "samsungcampos      0.145429  ...    0.057860     0.033197   0.061304   \n",
       "sonycampos         1.000000  ...    0.378812    -0.001845   0.015781   \n",
       "nokiacampos       -0.001836  ...   -0.000802     0.858295   0.017261   \n",
       "htccampos          0.058852  ...    0.018081     0.010478   0.253678   \n",
       "iphonecamneg       0.013254  ...    0.032570     0.026550   0.114716   \n",
       "samsungcamneg      0.032897  ...    0.060837     0.036543   0.122048   \n",
       "sonycamneg         0.408991  ...    0.604012    -0.000823   0.026290   \n",
       "nokiacamneg       -0.001938  ...   -0.000847     0.788927   0.016234   \n",
       "htccamneg          0.013568  ...    0.029574     0.019518   0.425361   \n",
       "iphonecamunc       0.016442  ...    0.025256     0.009049   0.057397   \n",
       "samsungcamunc      0.164043  ...    0.152542     0.123181   0.124516   \n",
       "sonycamunc         0.528452  ...    0.567358    -0.001282   0.031963   \n",
       "nokiacamunc       -0.001723  ...   -0.000753     0.958152   0.015949   \n",
       "htccamunc          0.056574  ...    0.050625     0.018802   0.601513   \n",
       "iphonedispos       0.019617  ...    0.027681     0.012382   0.091895   \n",
       "samsungdispos      0.111287  ...    0.111113     0.123591   0.285206   \n",
       "sonydispos         0.404993  ...    0.340766    -0.001024   0.019407   \n",
       "nokiadispos       -0.001636  ...   -0.000715     0.836700   0.014490   \n",
       "htcdispos          0.016457  ...    0.015451     0.005317   0.135495   \n",
       "iphonedisneg       0.006717  ...    0.023878     0.009723   0.096514   \n",
       "samsungdisneg      0.058932  ...    0.108932     0.072670   0.317065   \n",
       "sonydisneg         0.131892  ...    0.112633    -0.000431   0.006683   \n",
       "nokiadisneg       -0.001752  ...   -0.000766     0.861817   0.017639   \n",
       "htcdisneg          0.037624  ...    0.068252     0.023569   0.549764   \n",
       "iphonedisunc       0.007384  ...    0.022878     0.001485   0.092743   \n",
       "samsungdisunc      0.089489  ...    0.129397     0.136216   0.344839   \n",
       "sonydisunc         0.388804  ...    0.476597    -0.000927   0.037009   \n",
       "nokiadisunc       -0.001402  ...   -0.000613     0.923934   0.007326   \n",
       "htcdisunc          0.055055  ...    0.083275     0.010655   0.721407   \n",
       "iphoneperpos       0.009152  ...    0.036380     0.015281   0.123390   \n",
       "samsungperpos      0.046923  ...    0.057896     0.055507   0.189565   \n",
       "sonyperpos         0.387311  ...    0.735802    -0.001285   0.014073   \n",
       "nokiaperpos       -0.001998  ...   -0.000873     0.917333   0.016373   \n",
       "htcperpos          0.021326  ...    0.026076     0.020553   0.849739   \n",
       "iphoneperneg       0.006280  ...    0.042156     0.015347   0.140184   \n",
       "samsungperneg      0.034149  ...    0.060809     0.057204   0.271245   \n",
       "sonyperneg         0.182829  ...    0.668018    -0.000635   0.006295   \n",
       "nokiaperneg       -0.002038  ...   -0.000891     0.905222   0.017733   \n",
       "htcperneg          0.009013  ...    0.021705     0.026814   0.659652   \n",
       "iphoneperunc       0.008176  ...    0.050653     0.012553   0.171436   \n",
       "samsungperunc      0.053436  ...    0.091928     0.103767   0.346705   \n",
       "sonyperunc         0.378812  ...    1.000000    -0.000806   0.033233   \n",
       "nokiaperunc       -0.001845  ...   -0.000806     1.000000   0.012363   \n",
       "htcperunc          0.015781  ...    0.033233     0.012363   1.000000   \n",
       "iosperpos         -0.003118  ...   -0.002861     0.098336   0.000969   \n",
       "googleperpos       0.006673  ...   -0.002405     0.003180   0.333022   \n",
       "iosperneg         -0.002863  ...   -0.002711     0.098859   0.000673   \n",
       "googleperneg       0.008455  ...   -0.002643    -0.000137   0.280893   \n",
       "iosperunc         -0.006421  ...   -0.002806     0.157714   0.008437   \n",
       "googleperunc      -0.003434  ...   -0.001500     0.004180   0.394552   \n",
       "iphonesentiment   -0.090665  ...   -0.018084    -0.036167  -0.114171   \n",
       "\n",
       "                 iosperpos  googleperpos  iosperneg  googleperneg  iosperunc  \\\n",
       "iphone           -0.020059      0.118008  -0.019081      0.138742  -0.020368   \n",
       "samsunggalaxy    -0.005802      0.246046  -0.007839      0.290975  -0.015329   \n",
       "sonyxperia       -0.011009     -0.008467  -0.010323     -0.008570  -0.014802   \n",
       "nokialumina       0.030719      0.006515   0.032721      0.000653   0.052887   \n",
       "htcphone         -0.002927      0.019186  -0.002758      0.020726  -0.002666   \n",
       "ios               0.118278     -0.016402   0.112330     -0.018028   0.117035   \n",
       "googleandroid    -0.016702      0.638581  -0.015825      0.716515  -0.016377   \n",
       "iphonecampos     -0.003991      0.117902  -0.007060      0.124355  -0.001037   \n",
       "samsungcampos     0.102471      0.298281   0.075695      0.357362   0.044890   \n",
       "sonycampos       -0.003118      0.006673  -0.002863      0.008455  -0.006421   \n",
       "nokiacampos       0.103123      0.011564   0.103540      0.003941   0.165188   \n",
       "htccampos        -0.006121      0.163145  -0.005761      0.177230  -0.006079   \n",
       "iphonecamneg     -0.012229      0.417185  -0.013642      0.468075  -0.010749   \n",
       "samsungcamneg     0.110073      0.658644   0.081294      0.794282   0.047045   \n",
       "sonycamneg       -0.001276      0.020904  -0.001166      0.025126  -0.002865   \n",
       "nokiacamneg       0.089002      0.002719   0.090311     -0.000445   0.143676   \n",
       "htccamneg        -0.010934      0.578325  -0.009878      0.652644  -0.010191   \n",
       "iphonecamunc     -0.004920      0.076916  -0.008706      0.074858  -0.001336   \n",
       "samsungcamunc     0.129012      0.417375   0.097355      0.476690   0.057612   \n",
       "sonycamunc       -0.000890     -0.003825  -0.000746     -0.004204  -0.004463   \n",
       "nokiacamunc       0.108424      0.005909   0.108898      0.001299   0.173504   \n",
       "htccamunc        -0.007866      0.223305  -0.007102      0.227577  -0.005186   \n",
       "iphonedispos      0.020232      0.165576   0.015293      0.147023   0.024767   \n",
       "samsungdispos     0.118107      0.606458   0.092014      0.579951   0.057288   \n",
       "sonydispos        0.025392      0.000158   0.024833      0.001709  -0.003562   \n",
       "nokiadispos       0.079541     -0.002427   0.079405     -0.002668   0.127264   \n",
       "htcdispos        -0.001147      0.118340  -0.001013      0.109239   0.000579   \n",
       "iphonedisneg      0.015557      0.218541   0.016863      0.213640   0.018222   \n",
       "samsungdisneg     0.104504      0.808753   0.081942      0.826604   0.050730   \n",
       "sonydisneg        0.030273      0.000580   0.029531      0.001907  -0.001501   \n",
       "nokiadisneg       0.087180     -0.002599   0.086986     -0.002857   0.139410   \n",
       "htcdisneg         0.000258      0.704117   0.001073      0.698363   0.009817   \n",
       "iphonedisunc      0.024055      0.132862   0.023660      0.106084   0.030107   \n",
       "samsungdisunc     0.078623      0.594372   0.063341      0.512732   0.039951   \n",
       "sonydisunc        0.014600     -0.002764   0.014310     -0.003038  -0.003225   \n",
       "nokiadisunc       0.102037     -0.002080   0.102012     -0.002286   0.162690   \n",
       "htcdisunc         0.006433      0.483030   0.006752      0.406569   0.019677   \n",
       "iphoneperpos      0.210343      0.240267   0.224650      0.218848   0.211809   \n",
       "samsungperpos     0.274209      0.444302   0.214228      0.441229   0.137057   \n",
       "sonyperpos        0.005758      0.007712   0.005731      0.008070  -0.004473   \n",
       "nokiaperpos       0.084948      0.002051   0.084529     -0.000824   0.135942   \n",
       "htcperpos        -0.002803      0.380278  -0.002767      0.358458  -0.000189   \n",
       "iphoneperneg      0.247457      0.345247   0.282779      0.348685   0.255736   \n",
       "samsungperneg     0.202892      0.758411   0.161560      0.796365   0.103158   \n",
       "sonyperneg        0.000040      0.010539   0.000099      0.012140  -0.002210   \n",
       "nokiaperneg       0.079553      0.000515   0.079819     -0.001606   0.128002   \n",
       "htcperneg        -0.003590      0.628876  -0.002524      0.638941   0.000362   \n",
       "iphoneperunc      0.166660      0.242735   0.179411      0.196254   0.181783   \n",
       "samsungperunc     0.102804      0.616442   0.083218      0.541504   0.053897   \n",
       "sonyperunc       -0.002861     -0.002405  -0.002711     -0.002643  -0.002806   \n",
       "nokiaperunc       0.098336      0.003180   0.098859     -0.000137   0.157714   \n",
       "htcperunc         0.000969      0.333022   0.000673      0.280893   0.008437   \n",
       "iosperpos         1.000000     -0.009712   0.932382     -0.010676   0.905079   \n",
       "googleperpos     -0.009712      1.000000  -0.009203      0.957410  -0.009524   \n",
       "iosperneg         0.932382     -0.009203   1.000000     -0.010115   0.899819   \n",
       "googleperneg     -0.010676      0.957410  -0.010115      1.000000  -0.010468   \n",
       "iosperunc         0.905079     -0.009524   0.899819     -0.010468   1.000000   \n",
       "googleperunc     -0.006060      0.887033  -0.005742      0.756118  -0.005942   \n",
       "iphonesentiment  -0.015758     -0.137261  -0.010179     -0.163919  -0.011787   \n",
       "\n",
       "                 googleperunc  iphonesentiment  \n",
       "iphone               0.067859         0.014859  \n",
       "samsunggalaxy        0.142252        -0.359173  \n",
       "sonyxperia          -0.007916        -0.233170  \n",
       "nokialumina          0.007999        -0.055962  \n",
       "htcphone             0.013305        -0.051285  \n",
       "ios                 -0.010233         0.001656  \n",
       "googleandroid        0.371998        -0.189142  \n",
       "iphonecampos         0.073004        -0.029731  \n",
       "samsungcampos        0.159171        -0.112743  \n",
       "sonycampos          -0.003434        -0.090665  \n",
       "nokiacampos          0.012518        -0.033375  \n",
       "htccampos            0.100031        -0.120434  \n",
       "iphonecamneg         0.241003        -0.083963  \n",
       "samsungcamneg        0.342120        -0.185989  \n",
       "sonycamneg          -0.001532        -0.024826  \n",
       "nokiacamneg          0.003772        -0.033069  \n",
       "htccamneg            0.333727        -0.222972  \n",
       "iphonecamunc         0.058139         0.001443  \n",
       "samsungcamunc        0.269432        -0.138046  \n",
       "sonycamunc          -0.002386        -0.050327  \n",
       "nokiacamunc          0.006829        -0.031550  \n",
       "htccamunc            0.162431        -0.148881  \n",
       "iphonedispos         0.179686         0.014547  \n",
       "samsungdispos        0.636103        -0.099262  \n",
       "sonydispos          -0.001905        -0.038635  \n",
       "nokiadispos         -0.001514        -0.025922  \n",
       "htcdispos            0.124018        -0.060406  \n",
       "iphonedisneg         0.204416         0.003145  \n",
       "samsungdisneg        0.735526        -0.139965  \n",
       "sonydisneg          -0.000803        -0.019956  \n",
       "nokiadisneg         -0.001622        -0.028759  \n",
       "htcdisneg            0.643174        -0.192727  \n",
       "iphonedisunc         0.172276         0.027173  \n",
       "samsungdisunc        0.738457        -0.059548  \n",
       "sonydisunc          -0.001724        -0.032137  \n",
       "nokiadisunc         -0.001298        -0.023972  \n",
       "htcdisunc            0.593494        -0.132953  \n",
       "iphoneperpos         0.237625         0.029638  \n",
       "samsungperpos        0.427542        -0.081063  \n",
       "sonyperpos          -0.002392        -0.038913  \n",
       "nokiaperpos          0.003141        -0.041595  \n",
       "htcperpos            0.368327        -0.178427  \n",
       "iphoneperneg         0.296227        -0.004804  \n",
       "samsungperneg        0.641229        -0.138657  \n",
       "sonyperneg          -0.001182        -0.030850  \n",
       "nokiaperneg          0.001635        -0.044219  \n",
       "htcperneg            0.539902        -0.209196  \n",
       "iphoneperunc         0.297140         0.037200  \n",
       "samsungperunc        0.739887        -0.057920  \n",
       "sonyperunc          -0.001500        -0.018084  \n",
       "nokiaperunc          0.004180        -0.036167  \n",
       "htcperunc            0.394552        -0.114171  \n",
       "iosperpos           -0.006060        -0.015758  \n",
       "googleperpos         0.887033        -0.137261  \n",
       "iosperneg           -0.005742        -0.010179  \n",
       "googleperneg         0.756118        -0.163919  \n",
       "iosperunc           -0.005942        -0.011787  \n",
       "googleperunc         1.000000        -0.070284  \n",
       "iphonesentiment     -0.070284         1.000000  \n",
       "\n",
       "[59 rows x 59 columns]"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-02T11:04:33.213081Z",
     "start_time": "2020-03-02T11:04:30.394624Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Sample figsize in inches\n",
    "fig, ax = plt.subplots(figsize=(20,10))\n",
    "# Imbalanced DataFrame Correlation\n",
    "corr = iphoneData.corr()\n",
    "sns.heatmap(corr, cmap='Purples', annot_kws={'size':30}, ax=ax)\n",
    "ax.set_title(\"Correlation Matrix IPHONE DATA-Imbalanced\", fontsize=14)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-02T10:50:33.236946Z",
     "start_time": "2020-03-02T10:50:33.198050Z"
    }
   },
   "source": [
    "# Feature Selection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## SelectKBest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 573,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:12:20.509270Z",
     "start_time": "2020-03-03T13:12:20.432140Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Names</th>\n",
       "      <th>Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>iphone</td>\n",
       "      <td>28329.784605</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>samsunggalaxy</td>\n",
       "      <td>2814.821562</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>sonyxperia</td>\n",
       "      <td>1363.834695</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>nokialumina</td>\n",
       "      <td>77.116552</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>htcphone</td>\n",
       "      <td>7088.170837</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>ios</td>\n",
       "      <td>8644.241528</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>googleandroid</td>\n",
       "      <td>1224.005877</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>7</td>\n",
       "      <td>iphonecampos</td>\n",
       "      <td>2250.044378</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8</td>\n",
       "      <td>samsungcampos</td>\n",
       "      <td>2563.338005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>9</td>\n",
       "      <td>sonycampos</td>\n",
       "      <td>421.912493</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>10</td>\n",
       "      <td>nokiacampos</td>\n",
       "      <td>185.232055</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>11</td>\n",
       "      <td>htccampos</td>\n",
       "      <td>6604.927397</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12</td>\n",
       "      <td>iphonecamneg</td>\n",
       "      <td>4032.668048</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>13</td>\n",
       "      <td>samsungcamneg</td>\n",
       "      <td>3743.068630</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>14</td>\n",
       "      <td>sonycamneg</td>\n",
       "      <td>45.277649</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>15</td>\n",
       "      <td>nokiacamneg</td>\n",
       "      <td>149.083453</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>16</td>\n",
       "      <td>htccamneg</td>\n",
       "      <td>5265.111536</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>17</td>\n",
       "      <td>iphonecamunc</td>\n",
       "      <td>10883.576075</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>18</td>\n",
       "      <td>samsungcamunc</td>\n",
       "      <td>897.283397</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>19</td>\n",
       "      <td>sonycamunc</td>\n",
       "      <td>97.822889</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>20</td>\n",
       "      <td>nokiacamunc</td>\n",
       "      <td>100.598829</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>21</td>\n",
       "      <td>htccamunc</td>\n",
       "      <td>2148.526492</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>22</td>\n",
       "      <td>iphonedispos</td>\n",
       "      <td>5780.122571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>23</td>\n",
       "      <td>samsungdispos</td>\n",
       "      <td>2180.939785</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>24</td>\n",
       "      <td>sonydispos</td>\n",
       "      <td>290.047297</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>25</td>\n",
       "      <td>nokiadispos</td>\n",
       "      <td>268.737798</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>26</td>\n",
       "      <td>htcdispos</td>\n",
       "      <td>8555.754039</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>27</td>\n",
       "      <td>iphonedisneg</td>\n",
       "      <td>7111.885621</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>28</td>\n",
       "      <td>samsungdisneg</td>\n",
       "      <td>3358.496836</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>29</td>\n",
       "      <td>sonydisneg</td>\n",
       "      <td>486.685230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>30</td>\n",
       "      <td>nokiadisneg</td>\n",
       "      <td>197.088017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>31</td>\n",
       "      <td>htcdisneg</td>\n",
       "      <td>5160.932338</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>32</td>\n",
       "      <td>iphonedisunc</td>\n",
       "      <td>8808.523543</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>33</td>\n",
       "      <td>samsungdisunc</td>\n",
       "      <td>880.138819</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>34</td>\n",
       "      <td>sonydisunc</td>\n",
       "      <td>107.045515</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>35</td>\n",
       "      <td>nokiadisunc</td>\n",
       "      <td>143.582265</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>36</td>\n",
       "      <td>htcdisunc</td>\n",
       "      <td>2339.833508</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>37</td>\n",
       "      <td>iphoneperpos</td>\n",
       "      <td>1097.680198</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>38</td>\n",
       "      <td>samsungperpos</td>\n",
       "      <td>2076.111931</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>39</td>\n",
       "      <td>sonyperpos</td>\n",
       "      <td>127.221499</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>40</td>\n",
       "      <td>nokiaperpos</td>\n",
       "      <td>379.297318</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>41</td>\n",
       "      <td>htcperpos</td>\n",
       "      <td>6808.397768</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>42</td>\n",
       "      <td>iphoneperneg</td>\n",
       "      <td>773.527562</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>43</td>\n",
       "      <td>samsungperneg</td>\n",
       "      <td>3230.779623</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>44</td>\n",
       "      <td>sonyperneg</td>\n",
       "      <td>260.360089</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>45</td>\n",
       "      <td>nokiaperneg</td>\n",
       "      <td>383.961856</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>46</td>\n",
       "      <td>htcperneg</td>\n",
       "      <td>6110.308144</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>47</td>\n",
       "      <td>iphoneperunc</td>\n",
       "      <td>1697.064146</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>48</td>\n",
       "      <td>samsungperunc</td>\n",
       "      <td>744.446352</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>49</td>\n",
       "      <td>sonyperunc</td>\n",
       "      <td>27.257485</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>50</td>\n",
       "      <td>nokiaperunc</td>\n",
       "      <td>177.335626</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>51</td>\n",
       "      <td>htcperunc</td>\n",
       "      <td>2802.957625</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>52</td>\n",
       "      <td>iosperpos</td>\n",
       "      <td>88.725817</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>53</td>\n",
       "      <td>googleperpos</td>\n",
       "      <td>1858.426371</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>54</td>\n",
       "      <td>iosperneg</td>\n",
       "      <td>69.709824</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>55</td>\n",
       "      <td>googleperneg</td>\n",
       "      <td>3296.681474</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>56</td>\n",
       "      <td>iosperunc</td>\n",
       "      <td>32.567238</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>57</td>\n",
       "      <td>googleperunc</td>\n",
       "      <td>600.583776</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Names         Score\n",
       "0          iphone  28329.784605\n",
       "1   samsunggalaxy   2814.821562\n",
       "2      sonyxperia   1363.834695\n",
       "3     nokialumina     77.116552\n",
       "4        htcphone   7088.170837\n",
       "5             ios   8644.241528\n",
       "6   googleandroid   1224.005877\n",
       "7    iphonecampos   2250.044378\n",
       "8   samsungcampos   2563.338005\n",
       "9      sonycampos    421.912493\n",
       "10    nokiacampos    185.232055\n",
       "11      htccampos   6604.927397\n",
       "12   iphonecamneg   4032.668048\n",
       "13  samsungcamneg   3743.068630\n",
       "14     sonycamneg     45.277649\n",
       "15    nokiacamneg    149.083453\n",
       "16      htccamneg   5265.111536\n",
       "17   iphonecamunc  10883.576075\n",
       "18  samsungcamunc    897.283397\n",
       "19     sonycamunc     97.822889\n",
       "20    nokiacamunc    100.598829\n",
       "21      htccamunc   2148.526492\n",
       "22   iphonedispos   5780.122571\n",
       "23  samsungdispos   2180.939785\n",
       "24     sonydispos    290.047297\n",
       "25    nokiadispos    268.737798\n",
       "26      htcdispos   8555.754039\n",
       "27   iphonedisneg   7111.885621\n",
       "28  samsungdisneg   3358.496836\n",
       "29     sonydisneg    486.685230\n",
       "30    nokiadisneg    197.088017\n",
       "31      htcdisneg   5160.932338\n",
       "32   iphonedisunc   8808.523543\n",
       "33  samsungdisunc    880.138819\n",
       "34     sonydisunc    107.045515\n",
       "35    nokiadisunc    143.582265\n",
       "36      htcdisunc   2339.833508\n",
       "37   iphoneperpos   1097.680198\n",
       "38  samsungperpos   2076.111931\n",
       "39     sonyperpos    127.221499\n",
       "40    nokiaperpos    379.297318\n",
       "41      htcperpos   6808.397768\n",
       "42   iphoneperneg    773.527562\n",
       "43  samsungperneg   3230.779623\n",
       "44     sonyperneg    260.360089\n",
       "45    nokiaperneg    383.961856\n",
       "46      htcperneg   6110.308144\n",
       "47   iphoneperunc   1697.064146\n",
       "48  samsungperunc    744.446352\n",
       "49     sonyperunc     27.257485\n",
       "50    nokiaperunc    177.335626\n",
       "51      htcperunc   2802.957625\n",
       "52      iosperpos     88.725817\n",
       "53   googleperpos   1858.426371\n",
       "54      iosperneg     69.709824\n",
       "55   googleperneg   3296.681474\n",
       "56      iosperunc     32.567238\n",
       "57   googleperunc    600.583776"
      ]
     },
     "execution_count": 573,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = galaxyData.iloc[:,0:58]  #independent columns\n",
    "y = galaxyData.iloc[:,-1] \n",
    "#apply SelectKBest class to extract top 15 best features based on chi2\n",
    "bestfeatures = SelectKBest(score_func=chi2, k=15)\n",
    "fit = bestfeatures.fit(X,y)\n",
    "dfscores = pd.DataFrame(fit.scores_)\n",
    "dfcolumns = pd.DataFrame(X.columns)\n",
    "#concat two dataframes for better visualization \n",
    "featureScores = pd.concat([dfcolumns,dfscores],axis=1)\n",
    "featureScores.columns = ['Names','Score']  #naming the dataframe columns\n",
    "featureScores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 484,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:44:25.534340Z",
     "start_time": "2020-03-03T12:44:25.500694Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "           Names         Score\n",
      "0         iphone  28329.784605\n",
      "17  iphonecamunc  10883.576075\n",
      "32  iphonedisunc   8808.523543\n",
      "5            ios   8644.241528\n",
      "26     htcdispos   8555.754039\n",
      "27  iphonedisneg   7111.885621\n",
      "4       htcphone   7088.170837\n",
      "41     htcperpos   6808.397768\n",
      "11     htccampos   6604.927397\n",
      "46     htcperneg   6110.308144\n"
     ]
    }
   ],
   "source": [
    "print(featureScores.nlargest(10,'Score'))  #print 10 best features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 485,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:44:27.856946Z",
     "start_time": "2020-03-03T12:44:27.808305Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "features_kbest_galaxy = ['iphone', 'iphonecamunc', 'iphonedisunc', 'ios', 'htcdispos',\n",
    "       'iphonedisneg', 'htcphone', 'htcperpos', 'htccampos', 'htcperneg','galaxysentiment']\n",
    "FinalTraining_kbest_g = galaxyData.loc[:, features_kbest_galaxy]\n",
    "# Separating out the target\n",
    "ys_kbest_g = FinalTraining_kbest_g.loc[:,['galaxysentiment']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 486,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:44:30.098408Z",
     "start_time": "2020-03-03T12:44:30.007822Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Names</th>\n",
       "      <th>Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>iphone</td>\n",
       "      <td>28777.355470</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>samsunggalaxy</td>\n",
       "      <td>2441.843235</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>sonyxperia</td>\n",
       "      <td>1167.695509</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>nokialumina</td>\n",
       "      <td>63.756128</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>htcphone</td>\n",
       "      <td>5991.315289</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>ios</td>\n",
       "      <td>8595.095528</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>googleandroid</td>\n",
       "      <td>1098.541022</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>7</td>\n",
       "      <td>iphonecampos</td>\n",
       "      <td>2264.487048</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8</td>\n",
       "      <td>samsungcampos</td>\n",
       "      <td>2187.472021</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>9</td>\n",
       "      <td>sonycampos</td>\n",
       "      <td>360.845654</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>10</td>\n",
       "      <td>nokiacampos</td>\n",
       "      <td>153.601918</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>11</td>\n",
       "      <td>htccampos</td>\n",
       "      <td>5571.639332</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12</td>\n",
       "      <td>iphonecamneg</td>\n",
       "      <td>3971.763273</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>13</td>\n",
       "      <td>samsungcamneg</td>\n",
       "      <td>3161.122559</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>14</td>\n",
       "      <td>sonycamneg</td>\n",
       "      <td>39.620371</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>15</td>\n",
       "      <td>nokiacamneg</td>\n",
       "      <td>122.645958</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>16</td>\n",
       "      <td>htccamneg</td>\n",
       "      <td>4427.974129</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>17</td>\n",
       "      <td>iphonecamunc</td>\n",
       "      <td>11102.694879</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>18</td>\n",
       "      <td>samsungcamunc</td>\n",
       "      <td>758.704880</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>19</td>\n",
       "      <td>sonycamunc</td>\n",
       "      <td>82.864001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>20</td>\n",
       "      <td>nokiacamunc</td>\n",
       "      <td>83.868282</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>21</td>\n",
       "      <td>htccamunc</td>\n",
       "      <td>1811.165692</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>22</td>\n",
       "      <td>iphonedispos</td>\n",
       "      <td>5764.142162</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>23</td>\n",
       "      <td>samsungdispos</td>\n",
       "      <td>1841.241608</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>24</td>\n",
       "      <td>sonydispos</td>\n",
       "      <td>245.658749</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>25</td>\n",
       "      <td>nokiadispos</td>\n",
       "      <td>234.817037</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>26</td>\n",
       "      <td>htcdispos</td>\n",
       "      <td>7217.826400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>27</td>\n",
       "      <td>iphonedisneg</td>\n",
       "      <td>7098.070559</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>28</td>\n",
       "      <td>samsungdisneg</td>\n",
       "      <td>2831.777692</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>29</td>\n",
       "      <td>sonydisneg</td>\n",
       "      <td>406.959046</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>30</td>\n",
       "      <td>nokiadisneg</td>\n",
       "      <td>167.621657</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>31</td>\n",
       "      <td>htcdisneg</td>\n",
       "      <td>4361.693907</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>32</td>\n",
       "      <td>iphonedisunc</td>\n",
       "      <td>8743.312822</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>33</td>\n",
       "      <td>samsungdisunc</td>\n",
       "      <td>764.224277</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>34</td>\n",
       "      <td>sonydisunc</td>\n",
       "      <td>91.730020</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>35</td>\n",
       "      <td>nokiadisunc</td>\n",
       "      <td>123.250946</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>36</td>\n",
       "      <td>htcdisunc</td>\n",
       "      <td>1992.257829</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>37</td>\n",
       "      <td>iphoneperpos</td>\n",
       "      <td>1085.736420</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>38</td>\n",
       "      <td>samsungperpos</td>\n",
       "      <td>1807.207022</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>39</td>\n",
       "      <td>sonyperpos</td>\n",
       "      <td>113.921737</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>40</td>\n",
       "      <td>nokiaperpos</td>\n",
       "      <td>317.690312</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>41</td>\n",
       "      <td>htcperpos</td>\n",
       "      <td>5756.273157</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>42</td>\n",
       "      <td>iphoneperneg</td>\n",
       "      <td>779.606315</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>43</td>\n",
       "      <td>samsungperneg</td>\n",
       "      <td>2723.142335</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>44</td>\n",
       "      <td>sonyperneg</td>\n",
       "      <td>219.627664</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>45</td>\n",
       "      <td>nokiaperneg</td>\n",
       "      <td>320.050397</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>46</td>\n",
       "      <td>htcperneg</td>\n",
       "      <td>5114.081981</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>47</td>\n",
       "      <td>iphoneperunc</td>\n",
       "      <td>1680.555394</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>48</td>\n",
       "      <td>samsungperunc</td>\n",
       "      <td>651.666224</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>49</td>\n",
       "      <td>sonyperunc</td>\n",
       "      <td>23.998801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>50</td>\n",
       "      <td>nokiaperunc</td>\n",
       "      <td>147.846354</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>51</td>\n",
       "      <td>htcperunc</td>\n",
       "      <td>2361.473822</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>52</td>\n",
       "      <td>iosperpos</td>\n",
       "      <td>62.666054</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>53</td>\n",
       "      <td>googleperpos</td>\n",
       "      <td>1581.008784</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>54</td>\n",
       "      <td>iosperneg</td>\n",
       "      <td>43.457339</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>55</td>\n",
       "      <td>googleperneg</td>\n",
       "      <td>2790.610785</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>56</td>\n",
       "      <td>iosperunc</td>\n",
       "      <td>22.401322</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>57</td>\n",
       "      <td>googleperunc</td>\n",
       "      <td>527.370027</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Names         Score\n",
       "0          iphone  28777.355470\n",
       "1   samsunggalaxy   2441.843235\n",
       "2      sonyxperia   1167.695509\n",
       "3     nokialumina     63.756128\n",
       "4        htcphone   5991.315289\n",
       "5             ios   8595.095528\n",
       "6   googleandroid   1098.541022\n",
       "7    iphonecampos   2264.487048\n",
       "8   samsungcampos   2187.472021\n",
       "9      sonycampos    360.845654\n",
       "10    nokiacampos    153.601918\n",
       "11      htccampos   5571.639332\n",
       "12   iphonecamneg   3971.763273\n",
       "13  samsungcamneg   3161.122559\n",
       "14     sonycamneg     39.620371\n",
       "15    nokiacamneg    122.645958\n",
       "16      htccamneg   4427.974129\n",
       "17   iphonecamunc  11102.694879\n",
       "18  samsungcamunc    758.704880\n",
       "19     sonycamunc     82.864001\n",
       "20    nokiacamunc     83.868282\n",
       "21      htccamunc   1811.165692\n",
       "22   iphonedispos   5764.142162\n",
       "23  samsungdispos   1841.241608\n",
       "24     sonydispos    245.658749\n",
       "25    nokiadispos    234.817037\n",
       "26      htcdispos   7217.826400\n",
       "27   iphonedisneg   7098.070559\n",
       "28  samsungdisneg   2831.777692\n",
       "29     sonydisneg    406.959046\n",
       "30    nokiadisneg    167.621657\n",
       "31      htcdisneg   4361.693907\n",
       "32   iphonedisunc   8743.312822\n",
       "33  samsungdisunc    764.224277\n",
       "34     sonydisunc     91.730020\n",
       "35    nokiadisunc    123.250946\n",
       "36      htcdisunc   1992.257829\n",
       "37   iphoneperpos   1085.736420\n",
       "38  samsungperpos   1807.207022\n",
       "39     sonyperpos    113.921737\n",
       "40    nokiaperpos    317.690312\n",
       "41      htcperpos   5756.273157\n",
       "42   iphoneperneg    779.606315\n",
       "43  samsungperneg   2723.142335\n",
       "44     sonyperneg    219.627664\n",
       "45    nokiaperneg    320.050397\n",
       "46      htcperneg   5114.081981\n",
       "47   iphoneperunc   1680.555394\n",
       "48  samsungperunc    651.666224\n",
       "49     sonyperunc     23.998801\n",
       "50    nokiaperunc    147.846354\n",
       "51      htcperunc   2361.473822\n",
       "52      iosperpos     62.666054\n",
       "53   googleperpos   1581.008784\n",
       "54      iosperneg     43.457339\n",
       "55   googleperneg   2790.610785\n",
       "56      iosperunc     22.401322\n",
       "57   googleperunc    527.370027"
      ]
     },
     "execution_count": 486,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = iphoneData.iloc[:,0:58]  #independent columns\n",
    "y = iphoneData.iloc[:,-1] \n",
    "#apply SelectKBest class to extract top 15 best features based on chi2\n",
    "bestfeatures = SelectKBest(score_func=chi2, k=15)\n",
    "fit = bestfeatures.fit(X,y)\n",
    "dfscores = pd.DataFrame(fit.scores_)\n",
    "dfcolumns = pd.DataFrame(X.columns)\n",
    "#concat two dataframes for better visualization \n",
    "featureScores_ip = pd.concat([dfcolumns,dfscores],axis=1)\n",
    "featureScores_ip.columns = ['Names','Score']  #naming the dataframe columns\n",
    "featureScores_ip"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For future selection, we have been taken in consideration the features that has the best 10 scores and we have prepared them for the cv analysis wich can be the first one in the row."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-02T10:51:43.449118Z",
     "start_time": "2020-03-02T10:51:43.444130Z"
    }
   },
   "source": [
    "# VarianceThreshold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:32:26.055292Z",
     "start_time": "2020-03-03T17:32:26.017438Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12973, 33)"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Features_iphone=iphoneData.iloc[:,0:58]\n",
    "def variance_threshold_select(df,thresh =0.2):\n",
    "    selector = VarianceThreshold(thresh)\n",
    "    selector.fit(df)\n",
    "    df1 = df.iloc[:,selector.get_support(indices=False)]\n",
    "    return df1\n",
    "Features_iphone=variance_threshold_select(Features_iphone)\n",
    "onlySentiment_iphone=iphoneData.iloc[:,-1]\n",
    "finalTraining_ip=pd.concat([Features_iphone,onlySentiment_iphone],axis=1)\n",
    "finalTraining_ip.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:48:56.754029Z",
     "start_time": "2020-03-03T17:48:56.720147Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12973, 33)"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Features_galaxy=galaxyData.iloc[:,0:58]\n",
    "def variance_threshold_select(df,thresh =0.2):\n",
    "    selector = VarianceThreshold(thresh)\n",
    "    selector.fit(df)\n",
    "    df1 = df.iloc[:,selector.get_support(indices=False)]\n",
    "    return df1\n",
    "Features_galaxy=variance_threshold_select(Features_galaxy)\n",
    "onlySentiment_galaxy=galaxyData.iloc[:,-1]\n",
    "finalTraining_g=pd.concat([Features_galaxy,onlySentiment_galaxy],axis=1)\n",
    "finalTraining_g.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 288,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:52:49.690535Z",
     "start_time": "2020-03-03T19:52:49.159169Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x22802d30b48>"
      ]
     },
     "execution_count": 288,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "metrics=(finalTraining_ip.describe())\n",
    "metrics=metrics.transpose()\n",
    "metrics=metrics[:-1]\n",
    "metrics[\"std\"].hist(bins=40)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 289,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:52:54.750163Z",
     "start_time": "2020-03-03T19:52:54.232494Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x22802d68208>"
      ]
     },
     "execution_count": 289,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "metrics=(finalTraining_g.describe())\n",
    "metrics=metrics.transpose()\n",
    "metrics=metrics[:-1]\n",
    "metrics[\"std\"].hist(bins=40)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As aw can see in the first part, now we have a better distribution and a smaller data set wich can help us to model better later on."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Selection based on mutual information (MI)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 574,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:12:50.579958Z",
     "start_time": "2020-03-03T13:12:34.383061Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Specs</th>\n",
       "      <th>Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>iphone</td>\n",
       "      <td>0.235711</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>htcphone</td>\n",
       "      <td>0.101445</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>samsunggalaxy</td>\n",
       "      <td>0.079112</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>27</td>\n",
       "      <td>iphonedisneg</td>\n",
       "      <td>0.077182</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>32</td>\n",
       "      <td>iphonedisunc</td>\n",
       "      <td>0.074731</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12</td>\n",
       "      <td>iphonecamneg</td>\n",
       "      <td>0.069619</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>17</td>\n",
       "      <td>iphonecamunc</td>\n",
       "      <td>0.066254</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>22</td>\n",
       "      <td>iphonedispos</td>\n",
       "      <td>0.056963</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>26</td>\n",
       "      <td>htcdispos</td>\n",
       "      <td>0.053308</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>11</td>\n",
       "      <td>htccampos</td>\n",
       "      <td>0.052371</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Specs     Score\n",
       "0          iphone  0.235711\n",
       "4        htcphone  0.101445\n",
       "1   samsunggalaxy  0.079112\n",
       "27   iphonedisneg  0.077182\n",
       "32   iphonedisunc  0.074731\n",
       "12   iphonecamneg  0.069619\n",
       "17   iphonecamunc  0.066254\n",
       "22   iphonedispos  0.056963\n",
       "26      htcdispos  0.053308\n",
       "11      htccampos  0.052371"
      ]
     },
     "execution_count": 574,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.feature_selection import mutual_info_classif\n",
    "Xg = galaxyData.iloc[:,0:58]  #independent columns\n",
    "yg = galaxyData.iloc[:,-1]\n",
    "mi = mutual_info_classif(Xg, yg, n_neighbors=5, copy=True, random_state=None)\n",
    "scores = pd.DataFrame(mi)\n",
    "columns = pd.DataFrame(Xg.columns)\n",
    "#concat two dataframes for better visualization \n",
    "featScores = pd.concat([columns,scores],axis=1)\n",
    "featScores.columns = ['Specs','Score']  #naming the dataframe columns\n",
    "featScores.nlargest(10,'Score')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 549,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:05:36.351726Z",
     "start_time": "2020-03-03T13:05:19.805877Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Specs</th>\n",
       "      <th>Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>iphone</td>\n",
       "      <td>0.219403</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>htcphone</td>\n",
       "      <td>0.096618</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>27</td>\n",
       "      <td>iphonedisneg</td>\n",
       "      <td>0.076487</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>samsunggalaxy</td>\n",
       "      <td>0.075412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>32</td>\n",
       "      <td>iphonedisunc</td>\n",
       "      <td>0.073271</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12</td>\n",
       "      <td>iphonecamneg</td>\n",
       "      <td>0.065790</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>17</td>\n",
       "      <td>iphonecamunc</td>\n",
       "      <td>0.063967</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>22</td>\n",
       "      <td>iphonedispos</td>\n",
       "      <td>0.054078</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>ios</td>\n",
       "      <td>0.053146</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>16</td>\n",
       "      <td>htccamneg</td>\n",
       "      <td>0.048565</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Specs     Score\n",
       "0          iphone  0.219403\n",
       "4        htcphone  0.096618\n",
       "27   iphonedisneg  0.076487\n",
       "1   samsunggalaxy  0.075412\n",
       "32   iphonedisunc  0.073271\n",
       "12   iphonecamneg  0.065790\n",
       "17   iphonecamunc  0.063967\n",
       "22   iphonedispos  0.054078\n",
       "5             ios  0.053146\n",
       "16      htccamneg  0.048565"
      ]
     },
     "execution_count": 549,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.feature_selection import mutual_info_classif\n",
    "Xg = iphoneData.iloc[:,0:58]  #independent columns\n",
    "yg = iphoneData.iloc[:,-1]\n",
    "mi = mutual_info_classif(Xg, yg, n_neighbors=5, copy=True, random_state=None)\n",
    "\n",
    "scores = pd.DataFrame(mi)\n",
    "columns = pd.DataFrame(Xg.columns)\n",
    "\n",
    "#concat two dataframes for better visualization \n",
    "featScores = pd.concat([columns,scores],axis=1)\n",
    "featScores.columns = ['Specs','Score']  #naming the dataframe columns\n",
    "featScores.nlargest(10,'Score')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T09:48:10.888196Z",
     "start_time": "2020-03-03T09:48:10.880180Z"
    }
   },
   "source": [
    "## IPHONE ANALYSIS-MUTUAL INFORMATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T15:14:05.670823Z",
     "start_time": "2020-03-03T15:14:05.650843Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "features = ['iphone', 'htcphone', 'iphonedisneg', 'samsunggalaxy', 'iphonedisunc',\n",
    "       'iphonecamneg', 'iphonecamunc', 'iphonedispos', 'googleandroid', 'iphonecampos','iphonesentiment']\n",
    "\n",
    "FinalTraining_mi_ip = iphoneData.loc[:, features]\n",
    "# Separating out the target\n",
    "ys_mi_iphone = FinalTraining_mi_ip.loc[:,['iphonesentiment']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T15:14:07.335664Z",
     "start_time": "2020-03-03T15:14:07.317728Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12973, 11)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "FinalTraining_mi_ip.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T15:14:09.046604Z",
     "start_time": "2020-03-03T15:14:09.032166Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12973, 1)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ys_mi_iphone.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T09:51:49.953377Z",
     "start_time": "2020-03-03T09:51:49.935499Z"
    }
   },
   "source": [
    "# GALAXY ANALYSIS-MUTUAL INFORMATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 557,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:07:29.276228Z",
     "start_time": "2020-03-03T13:07:29.258223Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "features_galaxy = ['iphone', 'htcphone', 'samsunggalaxy', 'iphonedisunc','iphonedisneg',\n",
    "       'iphonecamneg', 'iphonecamunc', 'iphonedispos', 'htccampos', 'ios','galaxysentiment']\n",
    "FinalTraining_mi_g = galaxyData.loc[:, features_galaxy]\n",
    "# Separating out the target\n",
    "ys_mi_galaxy= FinalTraining_mi_g.loc[:,['galaxysentiment']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 558,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:07:31.081383Z",
     "start_time": "2020-03-03T13:07:31.065595Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12973, 1)"
      ]
     },
     "execution_count": 558,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "FinalTraining_mi_ip.shape\n",
    "ys_mi_iphone.shape\n",
    "\n",
    "FinalTraining_mi_g.shape\n",
    "ys_mi_galaxy.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 559,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:07:36.768854Z",
     "start_time": "2020-03-03T13:07:36.753567Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12973, 1)"
      ]
     },
     "execution_count": 559,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ys_mi_galaxy.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-02T10:54:51.210561Z",
     "start_time": "2020-03-02T10:54:51.205575Z"
    }
   },
   "source": [
    "# RFE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "C:\\Respaldo FR\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Features:15\n",
      "Selected Features:[False  True  True  True  True  True  True False False False False False\n",
      " False False  True False  True False False  True False  True False False\n",
      " False False False False False False False False False False False False\n",
      " False False  True  True False False False False  True False False False\n",
      " False  True False False False  True False False False False]\n",
      "Feature Ranking:[ 5  1  1  1  1  1  1 44 15 23 17  6 22  3  1 29  1 42  2  1 20  1 43  9\n",
      " 24 18 16 39 21 25 35 32 38  8 14 31 12 36  1  1 19 26 37  4  1 30 27 34\n",
      "  7  1 28 13 40  1 41 11 33 10]\n"
     ]
    }
   ],
   "source": [
    "#Recursive feature selection,Feature Elimination\n",
    "from sklearn.feature_selection import RFE\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "model = LogisticRegression()\n",
    "rfe = RFE(estimator=model, n_features_to_select=15)\n",
    "fit = rfe.fit(X_Ip, y_Ip)\n",
    "\n",
    "print(\"Features:{}\".format(fit.n_features_))\n",
    "print(\"Selected Features:{}\".format(fit.support_))\n",
    "print(\"Feature Ranking:{}\".format(fit.ranking_))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Recursive Feature Elimination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-02T10:57:13.746791Z",
     "start_time": "2020-03-02T10:57:10.125489Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.ensemble import ExtraTreesClassifier\n",
    "model = ExtraTreesClassifier()\n",
    "model.fit(X,y)\n",
    "#plot graph of feature importances for better visualization\n",
    "feat_importances = pd.Series(model.feature_importances_, index=X.columns)\n",
    "feat_importances.nlargest(10).plot(kind='barh')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 590,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:31:03.248317Z",
     "start_time": "2020-03-03T13:31:03.156696Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "features = ['iphone', 'samsunggalaxy', 'sonyxperia', 'nokialumina', 'htcphone',\n",
    "       'ios', 'googleandroid', 'iphonecampos', 'samsungcampos', 'sonycampos',\n",
    "       'nokiacampos', 'htccampos', 'iphonecamneg', 'samsungcamneg',\n",
    "       'sonycamneg', 'nokiacamneg', 'htccamneg', 'iphonecamunc',\n",
    "       'samsungcamunc', 'sonycamunc', 'nokiacamunc', 'htccamunc',\n",
    "       'iphonedispos', 'samsungdispos', 'sonydispos', 'nokiadispos',\n",
    "       'htcdispos', 'iphonedisneg', 'samsungdisneg', 'sonydisneg',\n",
    "       'nokiadisneg', 'htcdisneg', 'iphonedisunc', 'samsungdisunc',\n",
    "       'sonydisunc', 'nokiadisunc', 'htcdisunc', 'iphoneperpos',\n",
    "       'samsungperpos', 'sonyperpos', 'nokiaperpos', 'htcperpos',\n",
    "       'iphoneperneg', 'samsungperneg', 'sonyperneg', 'nokiaperneg',\n",
    "       'htcperneg', 'iphoneperunc', 'samsungperunc', 'sonyperunc',\n",
    "       'nokiaperunc', 'htcperunc', 'iosperpos', 'googleperpos', 'iosperneg',\n",
    "       'googleperneg', 'iosperunc', 'googleperunc']\n",
    "# Separating out the features\n",
    "xs = galaxyData.loc[:, features].values\n",
    "# Separating out the target\n",
    "ys = galaxyData.loc[:,['galaxysentiment']].values\n",
    "# Standardizing the features\n",
    "xs = StandardScaler().fit_transform(xs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 591,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:33:01.933386Z",
     "start_time": "2020-03-03T13:33:01.689038Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[11.89945302  8.44286842  4.83194744  4.43642461  3.65368988  2.96870509\n",
      "  2.68384876  2.12124299  1.93194245  1.75589697  1.72592392  1.3302627\n",
      "  1.18396326  0.9834249   0.68837548]\n",
      "[[ 4.39113971e-02  8.60337106e-02  1.09680809e-02  4.39818368e-02\n",
      "   1.27994052e-02  1.06516910e-02  1.59719745e-01  6.95557211e-02\n",
      "   1.51765953e-01  2.90282291e-02  5.45666950e-02  9.06501229e-02\n",
      "   1.40258587e-01  2.09668456e-01  3.74241251e-02  5.17454943e-02\n",
      "   1.91069799e-01  5.30166269e-02  1.92572867e-01  3.98689361e-02\n",
      "   5.96208755e-02  1.22809009e-01  9.32654960e-02  2.33617009e-01\n",
      "   2.84322015e-02  6.18245745e-02  4.92452134e-02  1.02716377e-01\n",
      "   2.59956725e-01  1.43856558e-02  5.96321542e-02  2.30227299e-01\n",
      "   8.23084360e-02  2.17167223e-01  4.06025376e-02  6.11403056e-02\n",
      "   1.86350865e-01  1.11055916e-01  1.98739925e-01  2.77969555e-02\n",
      "   5.62937537e-02  1.60391477e-01  1.34467721e-01  2.51671760e-01\n",
      "   1.79037707e-02  5.34964090e-02  2.14173140e-01  1.08122050e-01\n",
      "   2.23397145e-01  3.58066606e-02  5.94084140e-02  1.42847957e-01\n",
      "   4.06792326e-02  2.36537960e-01  3.64228218e-02  2.36295502e-01\n",
      "   3.23771879e-02  2.11428908e-01]\n",
      " [-1.66058718e-02 -2.17208204e-02 -5.98997555e-03  2.52519831e-01\n",
      "  -3.96213819e-03  7.03879440e-04 -4.69478173e-02 -1.06980244e-02\n",
      "  -2.37022924e-02 -9.10936483e-03  3.19133657e-01 -2.15213803e-02\n",
      "  -2.25924966e-02 -4.36982188e-02 -1.12537962e-02  3.04656369e-01\n",
      "  -4.60565599e-02 -9.67165939e-03 -7.40164483e-03 -1.13349506e-02\n",
      "   3.24227339e-01 -2.71970805e-02 -1.40871778e-02 -1.76306890e-02\n",
      "  -8.69663889e-03  3.10881823e-01 -1.12934926e-02 -1.91210678e-02\n",
      "  -4.37746031e-02 -4.81401427e-03  3.20211106e-01 -5.19680544e-02\n",
      "  -1.88297252e-02 -1.05010872e-02 -1.15518759e-02  2.99735198e-01\n",
      "  -4.56054341e-02 -1.34657449e-02 -2.55817615e-02 -9.16340580e-03\n",
      "   3.21026248e-01 -3.58669243e-02 -1.97723409e-02 -4.42696227e-02\n",
      "  -6.17258596e-03  3.09661660e-01 -4.71536869e-02 -1.72299510e-02\n",
      "  -2.11826176e-02 -1.05334770e-02  3.18632171e-01 -3.45725477e-02\n",
      "   4.41927465e-02 -6.42715240e-02  4.50434405e-02 -6.56523359e-02\n",
      "   6.54715654e-02 -5.53499363e-02]\n",
      " [-5.00800868e-02  6.60040310e-02  2.04811356e-01 -6.29225793e-03\n",
      "   1.04358381e-03 -4.23685917e-02 -3.46415008e-02 -5.55262112e-02\n",
      "   6.78993337e-02  2.64090670e-01 -3.98782608e-03 -4.16654355e-02\n",
      "  -6.92201584e-02  7.61221329e-03  3.14988668e-01 -3.95795429e-03\n",
      "  -5.13878242e-02 -6.27447474e-02  1.03837091e-01  3.28192927e-01\n",
      "   1.24944082e-03 -3.33781784e-02 -9.83101679e-02  6.01240471e-02\n",
      "   3.21329040e-01  2.91195742e-03 -8.51660759e-03 -9.63094984e-02\n",
      "   2.30192712e-02  2.37967186e-01 -1.26765840e-04 -3.22567564e-02\n",
      "  -9.89460166e-02  5.66195339e-02  3.52308636e-01  8.97949913e-03\n",
      "  -2.07425489e-02 -9.55825530e-02  4.78146932e-02  3.35818878e-01\n",
      "  -8.39819745e-04 -5.24335949e-02 -9.13306415e-02  7.15464784e-03\n",
      "   2.38752958e-01 -1.24995689e-03 -5.54270747e-02 -8.93868485e-02\n",
      "   3.35555289e-02  3.20687384e-01  3.99353495e-03 -4.11310534e-02\n",
      "  -3.71435094e-03 -5.23763080e-02 -7.32021737e-03 -4.84547970e-02\n",
      "  -1.71674909e-02 -4.94455900e-02]\n",
      " [ 1.22702945e-01 -7.80124509e-02  3.62082363e-02  6.30559827e-03\n",
      "  -3.23290938e-03  1.42803514e-01 -8.56522212e-02  1.67701585e-01\n",
      "  -1.19822758e-01  6.87511210e-02  6.15756778e-03  6.68175327e-02\n",
      "   1.50675474e-01 -1.10954780e-01  1.00283773e-01  6.54402473e-03\n",
      "  -3.50513644e-02  2.22905811e-01 -1.09880214e-01  9.45647851e-02\n",
      "  -5.40007560e-03  4.11546870e-02  3.47881651e-01 -1.07716557e-01\n",
      "   9.90487197e-02 -6.87960064e-03  1.27485862e-03  3.24107288e-01\n",
      "  -1.00588840e-01  7.84809392e-02 -1.01058346e-03 -3.85504412e-02\n",
      "   3.59076186e-01 -7.56332012e-02  1.06472448e-01 -2.01134760e-02\n",
      "   8.70934117e-03  3.09810677e-01 -1.18733370e-01  1.06242865e-01\n",
      "  -2.07732532e-03  9.64083112e-03  2.78249423e-01 -1.10441564e-01\n",
      "   8.19425195e-02 -2.32996239e-03 -4.54229588e-02  3.04106751e-01\n",
      "  -9.12464997e-02  1.04895941e-01 -1.16913522e-02 -4.55816687e-03\n",
      "   7.11700909e-02 -7.61847711e-02  8.11078160e-02 -8.75630565e-02\n",
      "   8.68268790e-02 -5.50625505e-02]\n",
      " [-2.51848212e-02  2.86635006e-02 -2.68459016e-02 -5.27631571e-02\n",
      "  -5.16572498e-02  5.89209350e-03 -1.28895292e-02 -1.48909859e-01\n",
      "   1.70021973e-01 -2.98626243e-02 -3.34698206e-02 -2.69261038e-01\n",
      "  -8.04314141e-02  7.04830185e-02 -3.62304910e-02 -3.51946305e-02\n",
      "  -2.21020884e-01 -7.38645098e-02  1.50982380e-01 -3.86438823e-02\n",
      "  -1.88686359e-02 -3.30632893e-01  7.36414934e-02  1.62324617e-01\n",
      "  -3.99302499e-02 -1.33777450e-02 -9.30164475e-02  8.28565404e-02\n",
      "   9.93115860e-02 -3.30090508e-02 -2.47148975e-02 -1.53429074e-01\n",
      "   8.07215319e-02  1.25367838e-01 -4.42103931e-02  8.88985913e-03\n",
      "  -2.12479328e-01  8.66928017e-02  2.24337218e-01 -3.97284583e-02\n",
      "  -3.22979198e-02 -2.98388552e-01  1.36521892e-01  1.36302374e-01\n",
      "  -2.91488572e-02 -3.61693020e-02 -2.01294519e-01  1.12695781e-01\n",
      "   1.47810144e-01 -3.59340595e-02 -1.28953537e-02 -2.51145166e-01\n",
      "   2.60438698e-01 -2.17859746e-03  2.50372159e-01 -6.87852836e-04\n",
      "   2.27004556e-01  1.45615209e-02]\n",
      " [-4.43764058e-01 -7.53976228e-02 -1.43829972e-02 -1.12028555e-02\n",
      "   3.60885946e-02 -3.92116945e-01 -1.37853753e-01 -9.99058079e-02\n",
      "  -1.17948413e-01 -1.03349049e-02 -6.67633779e-03  4.46915231e-02\n",
      "  -3.23478981e-01 -1.61517017e-01 -3.72198070e-03 -9.82748806e-03\n",
      "  -3.45320953e-02 -3.83743283e-01 -1.09094147e-01 -5.05572444e-03\n",
      "  -7.16976040e-03  1.01828583e-01  6.69510633e-02  5.77145459e-04\n",
      "   7.77891694e-03 -1.41151121e-02  6.48262868e-02 -4.53196868e-03\n",
      "  -2.80061007e-02  9.14787535e-03 -1.08806456e-02  7.05833563e-02\n",
      "  -1.88232692e-02  7.04860575e-02  8.11342803e-03 -1.09908531e-02\n",
      "   1.93128240e-01  1.46239795e-01 -2.46449390e-02  6.28873868e-03\n",
      "  -8.63673831e-03  1.57281505e-01  1.33191497e-01 -4.74028465e-02\n",
      "   4.70648198e-03 -7.75952513e-03  8.09900896e-02  1.62753176e-01\n",
      "   5.72644642e-02  7.85353736e-03 -9.56797482e-03  1.97020393e-01\n",
      "   1.61008669e-01 -3.22804512e-02  1.69636546e-01 -7.89615210e-02\n",
      "   1.70265021e-01  6.63024228e-02]\n",
      " [ 1.34129318e-01 -5.17482302e-03 -1.83033377e-03 -4.01649514e-02\n",
      "   2.97126426e-02  2.04822035e-01  2.72481992e-03  1.00194913e-01\n",
      "   3.41328193e-02  2.67652396e-03 -1.08584257e-02  1.68842150e-01\n",
      "   8.07212227e-02  5.47579851e-02 -1.29093870e-02 -1.57293110e-02\n",
      "   1.19305112e-01  1.33757588e-01  2.52360010e-02  1.74439268e-03\n",
      "  -1.18480816e-02  1.78773995e-01 -2.04594539e-01 -7.10173394e-02\n",
      "   1.87350727e-02 -3.33222104e-02  3.87214828e-02 -1.94401652e-01\n",
      "  -5.47869567e-02  2.38562624e-02 -2.63387624e-02  9.24531074e-03\n",
      "  -1.74818401e-01 -1.18259225e-01  8.28292385e-03 -1.82332819e-02\n",
      "   2.89466806e-02 -4.01373128e-02  3.68291871e-02 -4.46286787e-03\n",
      "  -2.57058339e-02  1.23669301e-01 -5.13253404e-02  6.84738564e-03\n",
      "  -1.31692676e-02 -2.37343620e-02  5.39180067e-02 -1.07968608e-01\n",
      "  -9.85213195e-02 -1.50432846e-02 -1.97842260e-02  7.69720483e-02\n",
      "   4.53677005e-01 -7.87964457e-02  4.51348206e-01 -5.21381667e-02\n",
      "   4.44010417e-01 -1.27744083e-01]\n",
      " [ 2.52408078e-01 -1.59516602e-01 -5.36119215e-02 -2.57473454e-02\n",
      "   9.10305930e-02  3.13781008e-01 -3.22701660e-01 -1.24414902e-01\n",
      "  -1.02315304e-01 -3.16451392e-02 -2.03365318e-02 -1.19615477e-01\n",
      "  -1.15378226e-01 -3.06901918e-01 -1.62048275e-02 -2.22425102e-02\n",
      "  -2.53765292e-01  1.90424654e-01 -8.57737697e-02  4.19936778e-03\n",
      "  -1.94150255e-03 -3.96688973e-02 -6.36893519e-02  1.63919029e-01\n",
      "   6.23405699e-03  5.32382474e-03  1.19644259e-01 -4.33666562e-02\n",
      "   3.18267658e-02  6.86285038e-03 -5.81675317e-03  3.19039800e-02\n",
      "   5.46384731e-02  3.02229283e-01  2.76771069e-02  2.36029496e-02\n",
      "   2.50420668e-01 -1.14994396e-01  7.38226303e-02 -3.24577759e-02\n",
      "  -1.05719637e-02  8.03605539e-02 -1.45609079e-01 -3.25101231e-02\n",
      "  -3.81889693e-02 -1.23389526e-02 -8.49139769e-03 -1.96348985e-02\n",
      "   2.71893062e-01 -7.74678531e-03  5.31488521e-03  2.11246379e-01\n",
      "  -1.22200092e-03 -5.07452620e-02 -7.42486426e-03 -1.53022376e-01\n",
      "  -6.20580336e-03  1.84054951e-01]\n",
      " [-2.77076345e-02  5.04174324e-02  3.45160828e-02 -2.26552012e-04\n",
      "   7.00735261e-01 -3.74265418e-02  2.74521639e-02  3.30338337e-02\n",
      "   6.43841912e-02  2.17431507e-02 -6.43216145e-04  1.70827443e-02\n",
      "   1.51087696e-02  4.61397163e-02 -6.73711859e-03 -2.09627490e-04\n",
      "  -4.11644718e-03 -1.38487182e-02  5.14210581e-02  6.96985906e-03\n",
      "  -3.08206386e-04 -3.67896669e-02  3.04444751e-02  9.03702250e-04\n",
      "  -1.17310619e-02  3.04591490e-03  6.74769837e-01  2.31492465e-02\n",
      "  -6.86116197e-03 -1.95892832e-02  1.57573174e-03 -4.16104605e-02\n",
      "   1.34650341e-02 -3.38064043e-02 -1.21904676e-02  1.85142034e-03\n",
      "  -7.24493236e-02  1.58212738e-02  2.51412057e-02 -1.84337829e-02\n",
      "  -6.54473242e-04 -6.29904333e-02  1.14135733e-02  6.49731533e-03\n",
      "  -2.17989464e-02 -1.42192051e-03 -5.52012070e-02 -7.29391323e-04\n",
      "  -2.63232073e-02 -1.59943058e-02  8.14387879e-05 -9.86382939e-02\n",
      "  -1.31566685e-02 -2.25396261e-02 -1.55940919e-02 -3.99312741e-03\n",
      "  -1.90435984e-02 -5.45151467e-02]\n",
      " [-1.05704141e-01  9.90006653e-02  9.18945053e-02 -2.39351859e-02\n",
      "  -7.76903537e-02 -1.00413289e-01 -2.04256564e-01  3.04356476e-01\n",
      "   3.37316404e-01  1.10748881e-01 -3.17393437e-02  3.10771034e-01\n",
      "  -3.93659715e-02 -3.67166786e-02 -6.61312341e-02 -2.95835305e-02\n",
      "  -6.91240447e-02  5.63349868e-02  2.77391105e-01  9.80592355e-02\n",
      "  -8.39251040e-03  1.83118797e-01  9.32950984e-02  1.47691187e-01\n",
      "  -1.39148973e-02  2.14631941e-02 -6.59971870e-02  4.14447623e-03\n",
      "  -6.34654625e-02 -5.30093360e-02  1.24562360e-03 -1.53188191e-01\n",
      "   5.02415770e-02  6.77463978e-02  2.91365450e-02  3.10479795e-02\n",
      "   4.58255647e-03  5.62007456e-02  2.42957625e-01 -1.72945216e-01\n",
      "  -1.55128397e-02  9.65548210e-02 -8.85479016e-02 -2.44620771e-02\n",
      "  -2.33247373e-01 -2.07033043e-02 -1.21627232e-01 -4.31851515e-03\n",
      "   9.04872825e-02 -1.21876588e-01  1.96954713e-03  1.06871305e-02\n",
      "  -6.29980900e-02 -2.44725045e-01 -8.62226216e-02 -2.33602936e-01\n",
      "  -1.10682437e-01 -2.09340482e-01]\n",
      " [ 7.60728119e-03  2.86513902e-02  5.18409640e-02  7.21475932e-03\n",
      "   5.21847588e-03  4.32306480e-03  5.85671618e-02 -6.90081720e-02\n",
      "  -6.31998688e-02 -3.02165869e-02  7.55496136e-03 -6.72058782e-02\n",
      "   2.80397360e-03  1.58605916e-02 -1.77951248e-01  7.29000259e-03\n",
      "   2.48163926e-02 -2.71160357e-02 -4.78019445e-02  8.36657513e-03\n",
      "   1.90235904e-03 -3.83619266e-02  1.24194965e-02 -3.11425853e-02\n",
      "   4.33757190e-01 -3.68897952e-03  6.91009656e-04  3.10668363e-02\n",
      "   1.19705374e-02  5.24623242e-01  7.35458629e-04  3.66739168e-02\n",
      "   1.76851489e-02 -1.84690878e-02  2.93211415e-01 -7.15837952e-03\n",
      "  -4.98494115e-03 -1.95115650e-02 -6.66441421e-02 -2.34476655e-01\n",
      "   4.30034529e-03 -3.40045151e-02  3.97482686e-03 -3.35482888e-03\n",
      "  -4.42640976e-01  5.33985737e-03  2.60334574e-02 -1.78312323e-03\n",
      "  -3.52827248e-02 -3.46740601e-01 -3.16168015e-04 -1.67351589e-02\n",
      "   9.65295098e-03  5.49344003e-02  1.43827230e-02  5.66797941e-02\n",
      "   2.69310918e-03  4.35488843e-02]\n",
      " [-7.14908933e-02 -4.23544442e-01 -5.39336297e-01 -3.44353893e-03\n",
      "   7.72216613e-03 -6.83027875e-02 -5.66597517e-04  2.10938017e-01\n",
      "   9.01071394e-02 -3.14564911e-01 -1.18632789e-02  1.45655581e-01\n",
      "   4.36845794e-02  4.11226930e-02 -1.09086788e-01 -1.04166204e-02\n",
      "  -1.52774412e-02  2.07524004e-02  4.58621254e-02 -2.22113817e-01\n",
      "  -3.96660040e-03  7.42977956e-03 -1.28623316e-02  4.49943569e-02\n",
      "   1.87447742e-01  1.02006438e-02  2.27532793e-02 -6.55483591e-02\n",
      "   2.05434214e-02  2.96214428e-01  7.98739711e-04 -6.37370643e-02\n",
      "  -5.56754671e-02  1.97671574e-02  3.82358857e-02  1.22505213e-02\n",
      "  -5.03324870e-02  4.93424382e-02  1.11255843e-01  1.91013586e-01\n",
      "  -5.96557143e-03 -2.40911123e-03 -2.94270436e-02  4.58985670e-02\n",
      "   2.25355083e-01 -8.68129658e-03 -6.79303885e-02 -9.84215912e-03\n",
      "   5.68006871e-02  9.53116744e-02  1.21278034e-03 -7.51380979e-02\n",
      "  -2.56437419e-02 -3.04487284e-02 -3.93627130e-02 -2.59571702e-02\n",
      "  -6.60760749e-02 -3.24194492e-02]\n",
      " [-1.12880417e-01  8.77453079e-02  1.72928875e-01 -7.43036175e-03\n",
      "  -1.92315640e-02 -1.00693800e-01  4.11260926e-03  4.99897064e-01\n",
      "  -2.60181191e-01  7.40561188e-02  1.03590788e-02  2.47965170e-01\n",
      "   1.15339204e-01 -1.32469676e-01  1.65763826e-02  4.69952641e-03\n",
      "  -1.81266952e-01  1.06338444e-01 -2.01948935e-01 -4.94571686e-03\n",
      "   2.98600885e-03 -1.33369434e-01 -1.08269571e-01  3.44417736e-02\n",
      "  -9.09218216e-03 -1.20224462e-03  9.39767178e-04 -2.22561639e-01\n",
      "   8.14677680e-02 -1.61104378e-02 -3.90006803e-03 -7.19681497e-02\n",
      "  -1.76038706e-01  1.49893101e-01 -2.53421655e-02  1.16719023e-03\n",
      "  -6.58357624e-03  1.37025342e-01 -1.12621645e-01 -8.10212418e-03\n",
      "  -5.28534976e-03 -9.35484577e-02 -3.69948941e-02  2.07791377e-02\n",
      "  -2.10980182e-02 -9.74487892e-03 -1.84440057e-01  3.85740833e-02\n",
      "   1.07371061e-01 -4.18268880e-02  4.11971984e-04 -2.24129830e-01\n",
      "   1.54646749e-02  2.02852926e-01  2.92712428e-02  1.26595548e-01\n",
      "   4.20233523e-02  3.00035944e-01]\n",
      " [ 4.23657732e-02  5.04446248e-01  2.97880119e-01 -1.04899451e-02\n",
      "   5.36127177e-04  4.41015861e-02 -8.18206513e-02 -1.54547480e-02\n",
      "   4.56617648e-02 -1.40295762e-01 -3.59159437e-03  4.13519003e-02\n",
      "  -7.32724572e-02 -4.04093960e-02 -2.75622073e-01 -6.42591529e-03\n",
      "  -4.33751779e-02 -4.20265434e-03 -7.13117755e-02 -4.14715118e-01\n",
      "   2.60015902e-03 -6.75504756e-03  4.33434065e-02  3.01161575e-02\n",
      "   1.30143239e-01  5.73229476e-03 -7.85932812e-03  3.72174305e-02\n",
      "  -3.37303466e-02  2.42395849e-01  2.95973538e-03 -5.43483108e-02\n",
      "   4.88971392e-02 -2.76817520e-02 -2.15413868e-01  9.87263529e-03\n",
      "  -1.96143516e-02 -2.41806088e-03  8.77400144e-02  3.12421570e-01\n",
      "   1.72734460e-03  1.03001710e-01 -1.20078413e-02  2.80153806e-03\n",
      "   3.02162207e-01  1.88641264e-04  9.66430399e-03 -1.18489025e-02\n",
      "   2.11549417e-02 -8.73463015e-02  4.02744336e-03  8.52347367e-02\n",
      "   3.84779873e-04 -3.55546890e-02 -5.33120468e-03 -3.78441713e-02\n",
      "  -2.32604848e-02 -1.24170431e-02]\n",
      " [-3.40488376e-02  1.08446190e-01 -2.82873428e-02 -3.58994904e-01\n",
      "  -1.62980311e-02 -5.17092631e-02  7.74258582e-02  5.79600299e-02\n",
      "  -2.16632876e-01 -2.53318596e-01 -2.25320141e-02  1.92585802e-01\n",
      "  -7.50270976e-02  4.07535246e-02  1.08670230e-01 -1.06917341e-01\n",
      "   2.66005943e-02 -8.53945975e-02 -3.73476991e-02  4.36740322e-02\n",
      "   9.45966686e-02  4.25494403e-02  2.27363714e-01  1.74766505e-02\n",
      "  -5.34415457e-02  7.45868137e-02  3.29212636e-03  2.92574689e-01\n",
      "   9.15369269e-02 -1.70416610e-02  2.87118601e-02  1.75952987e-02\n",
      "   2.71062246e-01  8.26102663e-02  9.35717160e-02  2.35384373e-01\n",
      "   5.19603725e-02 -3.10670956e-01 -8.52139089e-02 -4.78840111e-02\n",
      "  -4.25930872e-02 -3.65998823e-02 -2.61911969e-01  5.26798662e-02\n",
      "   2.01112457e-02 -8.57660673e-02 -8.50090532e-02 -3.16099076e-01\n",
      "   1.31799038e-03  4.65102682e-02  8.91727575e-02 -1.18480266e-01\n",
      "   8.16265742e-02 -5.83696817e-03  5.80763020e-02  1.33047498e-02\n",
      "   1.03409653e-01 -1.51360123e-02]]\n",
      "\n",
      "original shape:    (12973, 58)\n",
      "transformed shape: (12973, 15)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "pca = PCA(n_components=15)\n",
    "pca.fit(xs)\n",
    "print(pca.explained_variance_)\n",
    "print(pca.components_)\n",
    "print('')\n",
    "X_pca = pca.transform(xs)\n",
    "print(\"original shape:   \", xs.shape)\n",
    "print(\"transformed shape:\", X_pca.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 592,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:34:51.451221Z",
     "start_time": "2020-03-03T13:34:51.059242Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(np.cumsum(pca.explained_variance_ratio_))\n",
    "plt.xlabel('number of components')\n",
    "plt.ylabel('cumulative explained variance');\n",
    "\n",
    "X_train_pca = pca.transform(xs)\n",
    "#Variable renaming\n",
    "X_training=X_pca.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 597,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:38:04.918088Z",
     "start_time": "2020-03-03T13:38:04.855256Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Combined space has 10 features\n",
      "[[-1.41735084 -0.75059864 -0.07800208 ...  0.          0.\n",
      "   0.        ]\n",
      " [-1.41735084 -0.75059864 -0.07800208 ...  0.          0.\n",
      "   0.        ]\n",
      " [-1.41735084 -0.75059864 -0.07800208 ...  0.          0.\n",
      "   0.        ]\n",
      " ...\n",
      " [-1.41735084 -0.75059864 -0.07800208 ...  0.          0.\n",
      "   0.        ]\n",
      " [-0.18572988 -0.36744363 -0.10717553 ...  0.          0.\n",
      "   1.        ]\n",
      " [-1.41735084 -0.75059864 -0.07800208 ...  0.          0.\n",
      "   0.        ]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    }
   ],
   "source": [
    "#Finaly we can combine features obtained by PCA, univariate selection and other selection methods.\n",
    "from sklearn.pipeline import Pipeline, FeatureUnion\n",
    "#transformed features\n",
    "pca = PCA(n_components=5)\n",
    "# original features \n",
    "selection = SelectKBest(k=5)\n",
    "# Build estimator from PCA and Univariate selection:\n",
    "combined_features = FeatureUnion([(\"pca\", pca), (\"univ_select\", selection)])\n",
    "# Use combined features to transform dataset:\n",
    "X_features = combined_features.fit(X, y).transform(X)\n",
    "print(\"Combined space has\", X_features.shape[1], \"features\")\n",
    "print(X_features)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CROSS VALIDATION-TUNE AND MODEL FOR VARIANCE ANALYSYS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 500,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:48:57.045940Z",
     "start_time": "2020-03-03T12:48:56.977934Z"
    }
   },
   "outputs": [],
   "source": [
    "#CROSS VALIDATION-VARIANCE ANALYSYS\n",
    "X= finalTraining_ip.iloc[:, 0:31]\n",
    "y=finalTraining_ip['iphonesentiment']\n",
    "X_train, X_validation, Y_train, Y_validation = train_test_split(X, y,\n",
    "                                                                test_size=0.20,\n",
    "                                                                random_state=1,\n",
    "                                                                shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 501,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:48:59.631395Z",
     "start_time": "2020-03-03T12:48:59.515036Z"
    }
   },
   "outputs": [],
   "source": [
    "finalTraining_ip.iphonesentiment = finalTraining_ip.astype(\"category\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 502,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:49:02.259017Z",
     "start_time": "2020-03-03T12:49:02.198166Z"
    }
   },
   "outputs": [],
   "source": [
    "# Spot Check Algorithms\n",
    "models = []\n",
    "models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))\n",
    "models.append(('LDA', LinearDiscriminantAnalysis()))\n",
    "models.append(('CART', DecisionTreeClassifier()))\n",
    "models.append(('RF', RandomForestClassifier()))\n",
    "models.append((\"KNN\", KNeighborsClassifier()))\n",
    "#models.append(('SVM', SVC(gamma='auto')))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 503,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:49:30.156968Z",
     "start_time": "2020-03-03T12:49:05.118620Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LR: 0.673734 (0.006414)\n",
      "Testing accuracy: 0.668979\n",
      "Confusion matrix: \n",
      "[[ 200    1    3    1    7   32]\n",
      " [   0    0    0    0    0    0]\n",
      " [   1    0    0    0    0    0]\n",
      " [   2    0    2    4    1    5]\n",
      " [   5    1    1    2   78   17]\n",
      " [ 204   80   77  225  192 1454]]\n",
      "LDA: 0.663520 (0.007623)\n",
      "Testing accuracy: 0.662428\n",
      "Confusion matrix: \n",
      "[[ 176    1    3    1   10   31]\n",
      " [   0    0    0    0    0    0]\n",
      " [   0    1    0    0    0    0]\n",
      " [   3    0    0    4    1    2]\n",
      " [   6    1    2    2   83   19]\n",
      " [ 227   79   78  225  184 1456]]\n",
      "CART: 0.718054 (0.007935)\n",
      "Testing accuracy: 0.710212\n",
      "Confusion matrix: \n",
      "[[ 253    0   17    2    5   24]\n",
      " [   0    0    0    1    3   11]\n",
      " [   2    1    0    0    1   18]\n",
      " [   5    1    0   89    2   14]\n",
      " [   4    0    1    2   88   28]\n",
      " [ 148   80   65  138  179 1413]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RF: 0.735592 (0.005275)\n",
      "Testing accuracy: 0.727553\n",
      "Confusion matrix: \n",
      "[[ 254    0   15    1    5   15]\n",
      " [   0    0    0    0    0    1]\n",
      " [   2    1    0    0    0    5]\n",
      " [   3    0    0   89    3   11]\n",
      " [   2    0    0    3   87   18]\n",
      " [ 151   81   68  139  183 1458]]\n",
      "KNN: 0.689906 (0.082702)\n",
      "Testing accuracy: 0.710983\n",
      "Confusion matrix: \n",
      "[[ 251    1   14    1    6   23]\n",
      " [   0    0    0    0    0   11]\n",
      " [   0    0    1    0    0    4]\n",
      " [   2    1    1   89    8   29]\n",
      " [   4    0    0    1   83   20]\n",
      " [ 155   80   67  141  181 1421]]\n"
     ]
    }
   ],
   "source": [
    "results = []\n",
    "names = []\n",
    "for name, model in models:\n",
    "    kfold = StratifiedKFold(n_splits=10, random_state=1)\n",
    "    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')\n",
    "    results.append(cv_results)\n",
    "    names.append(name)\n",
    "    model.fit(X_train, Y_train)\n",
    "    predictions = model.predict (X_validation)\n",
    "    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))\n",
    "    print('Testing accuracy: %f' % accuracy_score(predictions, Y_validation))\n",
    "    print('Confusion matrix: ')\n",
    "    print(confusion_matrix(predictions, Y_validation))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 504,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:49:45.820954Z",
     "start_time": "2020-03-03T12:49:42.889527Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy 0.331792\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.22      0.92      0.36       412\n",
      "           1       0.07      0.24      0.11        82\n",
      "           2       0.15      0.07      0.10        83\n",
      "           3       0.79      0.40      0.53       232\n",
      "           4       0.85      0.32      0.47       278\n",
      "           5       0.82      0.18      0.30      1508\n",
      "\n",
      "    accuracy                           0.33      2595\n",
      "   macro avg       0.48      0.36      0.31      2595\n",
      "weighted avg       0.68      0.33      0.33      2595\n",
      "\n",
      "Confusion matrix\n",
      "[[381  56  58  86 136 994]\n",
      " [ 11  20  11  40  25 185]\n",
      " [  2   0   6   3   1  28]\n",
      " [  3   0   0  92   5  16]\n",
      " [  2   0   0   1  90  13]\n",
      " [ 13   6   8  10  21 272]]\n"
     ]
    }
   ],
   "source": [
    "#Tune-Knn\n",
    "model=KNeighborsClassifier(n_neighbors=3)\n",
    "model.fit(X,y)\n",
    "pf=model.predict(X_validation)\n",
    "\n",
    "print(\"Accuracy %f\" % (accuracy_score(Y_validation, pf)))\n",
    "print(classification_report(Y_validation, pf))\n",
    "\n",
    "print(\"Confusion matrix\")\n",
    "print(confusion_matrix(pf, Y_validation))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 506,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:50:32.884801Z",
     "start_time": "2020-03-03T12:50:31.037232Z"
    }
   },
   "outputs": [],
   "source": [
    "#Tunemodel-Tune-RF\n",
    "model=RandomForestClassifier(n_estimators=100)\n",
    "model.fit(X, y)\n",
    "prediction2 = model.predict(X_validation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 511,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T12:55:58.689312Z",
     "start_time": "2020-03-03T12:55:58.633301Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RandomForest evaluation\n",
      "Accuracy 0.775723 Kappa 0.560490\n",
      "Confusion matrix\n",
      "[[ 277    0    1    1    1  132]\n",
      " [   0    8    0    0    0   74]\n",
      " [  14    0    7    0    0   62]\n",
      " [   0    0    0  101    0  131]\n",
      " [   0    0    0    0  114  164]\n",
      " [   2    0    0    0    0 1506]]\n",
      "Classification report\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.95      0.67      0.79       412\n",
      "           1       1.00      0.10      0.18        82\n",
      "           2       0.88      0.08      0.15        83\n",
      "           3       0.99      0.44      0.60       232\n",
      "           4       0.99      0.41      0.58       278\n",
      "           5       0.73      1.00      0.84      1508\n",
      "\n",
      "    accuracy                           0.78      2595\n",
      "   macro avg       0.92      0.45      0.52      2595\n",
      "weighted avg       0.83      0.78      0.74      2595\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Evaluate the predictions\n",
    "print('RandomForest evaluation')\n",
    "print('Accuracy %f Kappa %f' % (accuracy_score(Y_validation, prediction2),\n",
    "                                cohen_kappa_score(Y_validation,prediction2)))\n",
    "print('Confusion matrix')\n",
    "print(confusion_matrix(Y_validation, prediction2))\n",
    "print('Classification report')\n",
    "print(classification_report(Y_validation, prediction2))\n",
    "\n",
    "# Save the accuracy \n",
    "accuracy_rf_ = accuracy_score(Y_validation, prediction2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T10:05:24.542239Z",
     "start_time": "2020-03-03T10:05:24.534292Z"
    }
   },
   "source": [
    "# Cross Validation-Tune and Modeling for M.I. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 576,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:13:19.945615Z",
     "start_time": "2020-03-03T13:13:19.937113Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12973, 1)"
      ]
     },
     "execution_count": 576,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Data set-clean and ready to model\n",
    "FinalTraining_mi_ip.shape\n",
    "ys_mi_iphone.shape\n",
    "FinalTraining_mi_g.shape\n",
    "ys_mi_galaxy.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 577,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:13:21.972485Z",
     "start_time": "2020-03-03T13:13:21.925880Z"
    }
   },
   "outputs": [],
   "source": [
    "FinalTraining_mi_ip.iphonesentiment = FinalTraining_mi_ip.astype(\"category\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 562,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:08:44.459609Z",
     "start_time": "2020-03-03T13:08:44.383856Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>iphone</th>\n",
       "      <th>htcphone</th>\n",
       "      <th>iphonedisneg</th>\n",
       "      <th>samsunggalaxy</th>\n",
       "      <th>iphonedisunc</th>\n",
       "      <th>iphonecamneg</th>\n",
       "      <th>iphonecamunc</th>\n",
       "      <th>iphonedispos</th>\n",
       "      <th>googleandroid</th>\n",
       "      <th>iphonecampos</th>\n",
       "      <th>iphonesentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12968</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12969</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12970</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12971</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12972</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>12973 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       iphone  htcphone  iphonedisneg  samsunggalaxy  iphonedisunc  \\\n",
       "0           1         0             0              0             0   \n",
       "1           1         0             0              0             0   \n",
       "2           1         0             0              0             0   \n",
       "3           1         0             0              0             0   \n",
       "4           1         0             0              0             0   \n",
       "...       ...       ...           ...            ...           ...   \n",
       "12968       1         0             0              0             0   \n",
       "12969       2         1             0              1             0   \n",
       "12970       1         0             0              0             0   \n",
       "12971       2         0             0              0             0   \n",
       "12972       1         0             0              0             0   \n",
       "\n",
       "       iphonecamneg  iphonecamunc  iphonedispos  googleandroid  iphonecampos  \\\n",
       "0                 0             0             0              0             0   \n",
       "1                 0             0             0              0             0   \n",
       "2                 0             0             0              0             0   \n",
       "3                 0             0             0              0             0   \n",
       "4                 0             0             0              0             0   \n",
       "...             ...           ...           ...            ...           ...   \n",
       "12968             0             0             0              0             0   \n",
       "12969             0             0             0              0             0   \n",
       "12970             0             0             0              0             0   \n",
       "12971             0             1             1              0             1   \n",
       "12972             0             0             0              0             0   \n",
       "\n",
       "      iphonesentiment  \n",
       "0                   1  \n",
       "1                   1  \n",
       "2                   1  \n",
       "3                   1  \n",
       "4                   1  \n",
       "...               ...  \n",
       "12968               1  \n",
       "12969               2  \n",
       "12970               1  \n",
       "12971               2  \n",
       "12972               1  \n",
       "\n",
       "[12973 rows x 11 columns]"
      ]
     },
     "execution_count": 562,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "FinalTraining_mi_ip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 578,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:14:19.711255Z",
     "start_time": "2020-03-03T13:14:19.691598Z"
    }
   },
   "outputs": [],
   "source": [
    "#cross validation\n",
    "X= FinalTraining_mi_ip.iloc[:,0:9]\n",
    "y= ys_mi_iphone \n",
    "X_train, X_validation, Y_train, Y_validation = train_test_split(X, y,\n",
    "                                                                test_size=0.20,\n",
    "                                                                random_state=1,\n",
    "                                                                shuffle=True)\n",
    "FinalTraining_mi_ip.iphonesentiment = FinalTraining_mi_ip.astype(\"category\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 581,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:15:25.495992Z",
     "start_time": "2020-03-03T13:15:25.447564Z"
    }
   },
   "outputs": [],
   "source": [
    "FinalTraining_mi_ip.iphonesentiment = FinalTraining_mi_ip.astype(\"category\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 582,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:15:27.998618Z",
     "start_time": "2020-03-03T13:15:27.986165Z"
    }
   },
   "outputs": [],
   "source": [
    "# Spot Check Algorithms\n",
    "models = []\n",
    "models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))\n",
    "models.append(('LDA', LinearDiscriminantAnalysis()))\n",
    "models.append(('CART', DecisionTreeClassifier()))\n",
    "models.append(('RF', RandomForestClassifier()))\n",
    "models.append((\"KNN\", KNeighborsClassifier()))\n",
    "#models.append(('SVM', SVC(gamma='auto')))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 583,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:15:39.633747Z",
     "start_time": "2020-03-03T13:15:29.566962Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LR: 0.683178 (0.005141)\n",
      "Testing accuracy: 0.675915\n",
      "Confusion matrix: \n",
      "[[ 226    1    4   36    8   41]\n",
      " [   0    0    0    0    0    0]\n",
      " [   0    0    0    0    0    0]\n",
      " [   1    0   16   23    2    3]\n",
      " [   2    1    1    5   56   15]\n",
      " [ 183   80   62  168  212 1449]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LDA: 0.689249 (0.008102)\n",
      "Testing accuracy: 0.684393\n",
      "Confusion matrix: \n",
      "[[ 227    1    4   36    8   41]\n",
      " [   0    0    0    0    0    0]\n",
      " [   0    0    0    2    0    0]\n",
      " [   0    0   15   21    2    1]\n",
      " [   3    1    2    5   81   19]\n",
      " [ 182   80   62  168  187 1447]]\n",
      "CART: 0.737816 (0.008581)\n",
      "Testing accuracy: 0.724085\n",
      "Confusion matrix: \n",
      "[[ 250    0    0    1    7   22]\n",
      " [   0    0    0    0    0    4]\n",
      " [   0    1   14    0    0    7]\n",
      " [  10    3    2  133    7   65]\n",
      " [   0    0    2    0   85   13]\n",
      " [ 152   78   65   98  179 1397]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:8: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  \n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RF: 0.743500 (0.008967)\n",
      "Testing accuracy: 0.733333\n",
      "Confusion matrix: \n",
      "[[ 250    0    1    0    3   18]\n",
      " [   0    0    0    0    0    0]\n",
      " [   0    0   14    0    0    3]\n",
      " [   9    3    1  133    6   60]\n",
      " [   3    1    1    0   88    9]\n",
      " [ 150   78   66   99  181 1418]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:516: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  estimator.fit(X_train, y_train, **fit_params)\n",
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:8: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNN: 0.733474 (0.011420)\n",
      "Testing accuracy: 0.719075\n",
      "Confusion matrix: \n",
      "[[ 244    0    0    0    3   15]\n",
      " [   0    0    0    1    1    1]\n",
      " [   1    0   16    0    0    5]\n",
      " [   6    3    3  141   10   83]\n",
      " [   4    0    0    1   79   18]\n",
      " [ 157   79   64   89  185 1386]]\n"
     ]
    }
   ],
   "source": [
    "results = []\n",
    "names = []\n",
    "for name, model in models:\n",
    "    kfold = StratifiedKFold(n_splits=10, random_state=1)\n",
    "    cv_results =cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')\n",
    "    results.append(cv_results)\n",
    "    names.append(name)\n",
    "    model.fit(X_train, Y_train)\n",
    "    predictions = model.predict(X_validation)\n",
    "    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))\n",
    "    print('Testing accuracy: %f' % accuracy_score(predictions, Y_validation))\n",
    "    print('Confusion matrix: ')\n",
    "    print(confusion_matrix(predictions, Y_validation))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 587,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:20:18.075455Z",
     "start_time": "2020-03-03T13:20:18.068476Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\n",
       "                       max_depth=None, max_features='auto', max_leaf_nodes=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, n_estimators=100,\n",
       "                       n_jobs=None, oob_score=False, random_state=42, verbose=0,\n",
       "                       warm_start=False)"
      ]
     },
     "execution_count": 587,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Import the model we are using\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "# Instantiate model with 1000 decision trees\n",
    "rf = RandomForestClassifier(n_estimators = 100, random_state = 42)\n",
    "rf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 585,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:16:05.981950Z",
     "start_time": "2020-03-03T13:16:05.470128Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\fabi_\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:3: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  This is separate from the ipykernel package so we can avoid doing imports until\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(2595,)"
      ]
     },
     "execution_count": 585,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Tunemodel-Tune-RF\n",
    "model=RandomForestClassifier(n_estimators=50)\n",
    "model.fit(X, y)\n",
    "prediction3 = model.predict(X_validation)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 586,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T13:19:54.672373Z",
     "start_time": "2020-03-03T13:19:54.634476Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RandomForest evaluation\n",
      "Accuracy 0.757611 Kappa 0.538594\n",
      "Confusion matrix\n",
      "[[ 263    0    0    6    0  143]\n",
      " [   0    3    0    3    0   76]\n",
      " [   0    0   20    1    0   62]\n",
      " [   0    0    0  132    0  100]\n",
      " [   1    0    0    3  105  169]\n",
      " [  13    0    0   50    2 1443]]\n",
      "Classification report\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.95      0.64      0.76       412\n",
      "           1       1.00      0.04      0.07        82\n",
      "           2       1.00      0.24      0.39        83\n",
      "           3       0.68      0.57      0.62       232\n",
      "           4       0.98      0.38      0.55       278\n",
      "           5       0.72      0.96      0.82      1508\n",
      "\n",
      "    accuracy                           0.76      2595\n",
      "   macro avg       0.89      0.47      0.54      2595\n",
      "weighted avg       0.80      0.76      0.73      2595\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Evaluate the predictions\n",
    "print('RandomForest evaluation')\n",
    "print('Accuracy %f Kappa %f' % (accuracy_score(Y_validation, prediction3),\n",
    "                                cohen_kappa_score(Y_validation,prediction3)))\n",
    "print('Confusion matrix')\n",
    "print(confusion_matrix(Y_validation, prediction3))\n",
    "print('Classification report')\n",
    "print(classification_report(Y_validation, prediction3))\n",
    "\n",
    "# Save the accuracy \n",
    "accuracy_rf_ip = accuracy_score(Y_validation, prediction3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "I have tarined many models, C5.0 decision tree, random forest, SVM and kkNN algorithms on multiple subsetting, sampling and feature selection preprocesses for all the pre processing and feature selection.\n",
    "I have decided to use the variance analysis for my final predicitons with RF \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T14:22:02.795166Z",
     "start_time": "2020-03-03T14:22:02.787216Z"
    }
   },
   "source": [
    "# Final Predicitons Iphone-Large Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "matrix=MAtrix_forLazayPeople_as_me.drop([\"Unnamed: 0\",\"id\"], axis=1)\n",
    "matrix.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:34:53.018706Z",
     "start_time": "2020-03-03T17:34:53.008736Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['iphone', 'htcphone', 'ios', 'iphonecampos', 'samsungcampos',\n",
       "       'htccampos', 'iphonecamneg', 'samsungcamneg', 'htccamneg',\n",
       "       'iphonecamunc', 'htccamunc', 'iphonedispos', 'samsungdispos',\n",
       "       'htcdispos', 'iphonedisneg', 'samsungdisneg', 'sonydisneg', 'htcdisneg',\n",
       "       'iphonedisunc', 'samsungdisunc', 'htcdisunc', 'iphoneperpos',\n",
       "       'samsungperpos', 'htcperpos', 'iphoneperneg', 'samsungperneg',\n",
       "       'htcperneg', 'iphoneperunc', 'htcperunc', 'iosperpos', 'iosperneg',\n",
       "       'googleperneg', 'iphonesentiment'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 151,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "finalTraining_ip.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:36:28.757089Z",
     "start_time": "2020-03-03T17:36:28.746119Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['iphone', 'htcphone', 'ios', 'iphonecampos', 'samsungcampos',\n",
       "       'htccampos', 'iphonecamneg', 'samsungcamneg', 'htccamneg',\n",
       "       'iphonecamunc', 'htccamunc', 'iphonedispos', 'samsungdispos',\n",
       "       'htcdispos', 'iphonedisneg', 'samsungdisneg', 'sonydisneg', 'htcdisneg',\n",
       "       'iphonedisunc', 'samsungdisunc', 'htcdisunc', 'iphoneperpos',\n",
       "       'samsungperpos', 'htcperpos', 'iphoneperneg', 'samsungperneg',\n",
       "       'htcperneg', 'iphoneperunc', 'htcperunc', 'iosperpos', 'iosperneg',\n",
       "       'googleperneg'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 155,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "finalTraining_ip.columns[:-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:22:57.447470Z",
     "start_time": "2020-03-03T17:22:57.403590Z"
    }
   },
   "outputs": [],
   "source": [
    "finalTraining_ip=finalTraining_ip[Features_lm.columns]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:36:58.441731Z",
     "start_time": "2020-03-03T17:36:58.413836Z"
    }
   },
   "outputs": [],
   "source": [
    "matrix=matrix[finalTraining_ip.columns[:-1]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:37:39.099781Z",
     "start_time": "2020-03-03T17:37:39.082855Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True])"
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "matrix.columns == finalTraining_ip.columns[:-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 272,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:03:35.881786Z",
     "start_time": "2020-03-03T19:03:35.831926Z"
    }
   },
   "outputs": [],
   "source": [
    "#split\n",
    "X= finalTraining_ip.iloc[:,:-1]\n",
    "y= finalTraining_ip[\"iphonesentiment\"] \n",
    "X_test = finalTraining23.iloc[:,0:33]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 273,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:03:37.306393Z",
     "start_time": "2020-03-03T19:03:37.296420Z"
    }
   },
   "outputs": [],
   "source": [
    "finalTraining_ip.iphonesentiment=finalTraining_ip.iphonesentiment.astype(\"category\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 274,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:03:47.177195Z",
     "start_time": "2020-03-03T19:03:44.072288Z"
    }
   },
   "outputs": [],
   "source": [
    "model=RandomForestClassifier(n_estimators=100)\n",
    "model.fit(X,y)\n",
    "prediction_iphone_lm = model.predict(matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 275,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:04:50.612435Z",
     "start_time": "2020-03-03T19:04:50.597476Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Predicted Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114548</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114549</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114550</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114551</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114552</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>114553 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        Predicted Sentiment\n",
       "0                         0\n",
       "1                         5\n",
       "2                         0\n",
       "3                         5\n",
       "4                         5\n",
       "...                     ...\n",
       "114548                    0\n",
       "114549                    0\n",
       "114550                    5\n",
       "114551                    5\n",
       "114552                    5\n",
       "\n",
       "[114553 rows x 1 columns]"
      ]
     },
     "execution_count": 275,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prediction_iphone_lm= pd.DataFrame(prediction_iphone_lm)\n",
    "prediction_iphone_lm.columns=[\"Predicted Sentiment\"]\n",
    "prediction_iphone_lm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 276,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:07:31.446814Z",
     "start_time": "2020-03-03T19:07:31.425898Z"
    }
   },
   "outputs": [],
   "source": [
    "#creating a copy for graphs and merge\n",
    "matrix_iphone=matrix.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 277,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:07:32.913401Z",
     "start_time": "2020-03-03T19:07:32.901438Z"
    }
   },
   "outputs": [],
   "source": [
    "df3=matrix_iphone\n",
    "df4=prediction_iphone_lm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 278,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:07:52.554845Z",
     "start_time": "2020-03-03T19:07:52.538887Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['iphone', 'htcphone', 'ios', 'iphonecampos', 'samsungcampos',\n",
       "       'htccampos', 'iphonecamneg', 'samsungcamneg', 'htccamneg',\n",
       "       'iphonecamunc', 'htccamunc', 'iphonedispos', 'samsungdispos',\n",
       "       'htcdispos', 'iphonedisneg', 'samsungdisneg', 'sonydisneg', 'htcdisneg',\n",
       "       'iphonedisunc', 'samsungdisunc', 'htcdisunc', 'iphoneperpos',\n",
       "       'samsungperpos', 'htcperpos', 'iphoneperneg', 'samsungperneg',\n",
       "       'htcperneg', 'iphoneperunc', 'htcperunc', 'iosperpos', 'iosperneg',\n",
       "       'googleperneg', 'Predicted Sentiment'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 278,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Iphone_predicitons=pd.concat([df3, df4], axis=1)\n",
    "Iphone_predicitons.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 279,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:08:26.915218Z",
     "start_time": "2020-03-03T19:08:26.894231Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Summary-Predicted Sentiment Iphone 0    54019\n",
      "5    51529\n",
      "4     3157\n",
      "3     3109\n",
      "2     2685\n",
      "1       54\n",
      "Name: Predicted Sentiment, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(\"Summary-Predicted Sentiment Iphone\", Iphone_predicitons['Predicted Sentiment'].value_counts())\n",
    "Iphone_predicitons[\"Predicted Sentiment\"]=Iphone_predicitons[\"Predicted Sentiment\"].replace([0, 1, 2, 3, 4, 5],\n",
    "                                            ['Negative', 'Negative', 'Neutral', 'Neutral', 'Positive', 'Positive'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 280,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T19:08:44.454242Z",
     "start_time": "2020-03-03T19:08:44.302649Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Plotting Final Prediction\n",
    "Iphone_predicitons['Predicted Sentiment'].value_counts().reindex([\"Negative\", \"Neutral\", \"Positive\"]).plot(kind='bar', figsize=(4,4))\n",
    "plt.title (\"Iphone Sentiment\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Final Predictions Galaxy - Large Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:52:59.258685Z",
     "start_time": "2020-03-03T17:52:59.230781Z"
    }
   },
   "outputs": [],
   "source": [
    "matrix=matrix[finalTraining_g.columns[:-1]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:53:07.254291Z",
     "start_time": "2020-03-03T17:53:07.240333Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True])"
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "matrix.columns == finalTraining_ip.columns[:-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:49:26.328920Z",
     "start_time": "2020-03-03T17:49:26.319970Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['iphone', 'htcphone', 'ios', 'iphonecampos', 'samsungcampos',\n",
       "       'htccampos', 'iphonecamneg', 'samsungcamneg', 'htccamneg',\n",
       "       'iphonecamunc', 'htccamunc', 'iphonedispos', 'samsungdispos',\n",
       "       'htcdispos', 'iphonedisneg', 'samsungdisneg', 'sonydisneg', 'htcdisneg',\n",
       "       'iphonedisunc', 'samsungdisunc', 'htcdisunc', 'iphoneperpos',\n",
       "       'samsungperpos', 'htcperpos', 'iphoneperneg', 'samsungperneg',\n",
       "       'htcperneg', 'iphoneperunc', 'htcperunc', 'iosperpos', 'iosperneg',\n",
       "       'googleperneg', 'galaxysentiment'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 168,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "finalTraining_g.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "finalTraining_ip.columns[:-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:53:27.777376Z",
     "start_time": "2020-03-03T17:53:27.765409Z"
    }
   },
   "outputs": [],
   "source": [
    "#split\n",
    "X= finalTraining_g.iloc[:,:-1]\n",
    "y= finalTraining_g[\"galaxysentiment\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T17:54:31.990059Z",
     "start_time": "2020-03-03T17:54:31.983078Z"
    }
   },
   "outputs": [],
   "source": [
    "finalTraining_g.galaxysentiment=finalTraining_g.galaxysentiment.astype(\"category\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 257,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T18:57:36.318950Z",
     "start_time": "2020-03-03T18:57:33.098516Z"
    }
   },
   "outputs": [],
   "source": [
    "model=RandomForestClassifier(n_estimators=100)\n",
    "model.fit(X,y)\n",
    "prediction5 = model.predict(matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 258,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T18:57:40.057729Z",
     "start_time": "2020-03-03T18:57:36.821605Z"
    }
   },
   "outputs": [],
   "source": [
    "model=RandomForestClassifier(n_estimators=100)\n",
    "model.fit(X,y)\n",
    "prediction5 = model.predict(matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 259,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T18:57:40.497518Z",
     "start_time": "2020-03-03T18:57:40.487544Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(114553,)"
      ]
     },
     "execution_count": 259,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prediction5.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 260,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T18:57:40.986210Z",
     "start_time": "2020-03-03T18:57:40.972249Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Predicted Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114548</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114549</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114550</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114551</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>114552</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>114553 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        Predicted Sentiment\n",
       "0                         0\n",
       "1                         5\n",
       "2                         0\n",
       "3                         5\n",
       "4                         5\n",
       "...                     ...\n",
       "114548                    0\n",
       "114549                    0\n",
       "114550                    5\n",
       "114551                    5\n",
       "114552                    5\n",
       "\n",
       "[114553 rows x 1 columns]"
      ]
     },
     "execution_count": 260,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prediction5= pd.DataFrame(prediction5)\n",
    "prediction5.columns=[\"Predicted Sentiment\"]\n",
    "prediction5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 261,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T18:57:41.462933Z",
     "start_time": "2020-03-03T18:57:41.433014Z"
    }
   },
   "outputs": [],
   "source": [
    "#prediction5=[prediciton5]\n",
    "matrix_galaxy=matrix.copy()\n",
    "matrix_iphone=matrix.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 262,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T18:57:44.275603Z",
     "start_time": "2020-03-03T18:57:44.264443Z"
    }
   },
   "outputs": [],
   "source": [
    "df1=matrix_galaxy\n",
    "df2=prediction5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 263,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T18:57:47.131832Z",
     "start_time": "2020-03-03T18:57:47.107902Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['iphone', 'htcphone', 'ios', 'iphonecampos', 'samsungcampos',\n",
       "       'htccampos', 'iphonecamneg', 'samsungcamneg', 'htccamneg',\n",
       "       'iphonecamunc', 'htccamunc', 'iphonedispos', 'samsungdispos',\n",
       "       'htcdispos', 'iphonedisneg', 'samsungdisneg', 'sonydisneg', 'htcdisneg',\n",
       "       'iphonedisunc', 'samsungdisunc', 'htcdisunc', 'iphoneperpos',\n",
       "       'samsungperpos', 'htcperpos', 'iphoneperneg', 'samsungperneg',\n",
       "       'htcperneg', 'iphoneperunc', 'htcperunc', 'iosperpos', 'iosperneg',\n",
       "       'googleperneg', 'Predicted Sentiment'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 263,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Galaxy_predicitons=pd.concat([df1, df2], axis=1)\n",
    "Galaxy_predicitons.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 266,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T18:59:05.844534Z",
     "start_time": "2020-03-03T18:59:05.804640Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Summary-Predicted Sentiment Galaxy Positive    54919\n",
      "Negative    53983\n",
      "Neutral      5651\n",
      "Name: Predicted Sentiment, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(\"Summary-Predicted Sentiment Galaxy\", Galaxy_predicitons['Predicted Sentiment'].value_counts())\n",
    "Galaxy_predicitons[\"Predicted Sentiment\"]=Galaxy_predicitons[\"Predicted Sentiment\"].replace([0, 1, 2, 3, 4, 5],\n",
    "                                            ['Negative', 'Negative', 'Neutral', 'Neutral', 'Positive', 'Positive'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 268,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T18:59:58.786910Z",
     "start_time": "2020-03-03T18:59:58.637312Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Plotting Final Prediction\n",
    "Galaxy_predicitons['Predicted Sentiment'].value_counts().reindex([\"Negative\", \"Neutral\", \"Positive\"]).plot(kind='bar', figsize=(4,4))\n",
    "plt.title (\"Galaxy Sentiment\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-03-03T20:08:03.363455Z",
     "start_time": "2020-03-03T20:08:03.358409Z"
    }
   },
   "source": [
    "# CONCLUSIONS "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is important to clarify that both positives sentiments ratios in both data sets are very similar\n",
    "We can conclude that regarding the count of positives and negatives galaxy has better positive sentiment wich is why we recommendt galaxy better than iphone."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  },
  "notify_time": "0",
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": true,
   "toc_position": {
    "height": "885.33px",
    "left": "27px",
    "top": "234.14px",
    "width": "364.8px"
   },
   "toc_section_display": true,
   "toc_window_display": true
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "position": {
    "height": "342.86px",
    "left": "1012px",
    "right": "20px",
    "top": "138px",
    "width": "522px"
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": true
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
