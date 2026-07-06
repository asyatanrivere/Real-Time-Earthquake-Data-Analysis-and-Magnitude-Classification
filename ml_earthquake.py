from earthquake_analysis import main
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # for z score
from sklearn.tree import DecisionTreeClassifier,plot_tree

df=pd.read_csv("dataset/dataset_edited.csv")

"""       Date & TimeUTC  Lat.degrees  Lon.degrees  Depthkm  Mag.                           Region  Year        Date  Month  Day      Time
0 2026-07-06 12:50:46      -36.970      -73.210     19.0   3.8          OFFSHORE BIO-BIO, CHILE  2026  2026-07-06      7    6  12:50:46
1 2026-07-06 12:49:44       12.720      -88.670     32.0   3.1             OFFSHORE EL SALVADOR  2026  2026-07-06      7    6  12:49:44
2 2026-07-06 12:47:59       64.531      -16.876      5.0   4.2                          ICELAND  2026  2026-07-06      7    6  12:47:59
3 2026-07-06 12:43:14       33.347     -118.339     10.0   3.0      CHANNEL ISLANDS, CALIFORNIA  2026  2026-07-06      7    6  12:43:14
4 2026-07-06 12:43:14      -31.900      -69.170    111.0   3.2              SAN JUAN, ARGENTINA  2026  2026-07-06      7    6  12:43:14
5 2026-07-06 12:37:31      -36.970      -73.580     26.0   4.2          OFFSHORE BIO-BIO, CHILE  2026  2026-07-06      7    6  12:37:31
6 2026-07-06 12:32:00      -32.790      -71.730     27.0   2.5       OFFSHORE VALPARAISO, CHILE  2026  2026-07-06      7    6  12:32:00
7 2026-07-06 12:26:56      -31.280      -68.550    100.0   3.8              SAN JUAN, ARGENTINA  2026  2026-07-06      7    6  12:26:56
8 2026-07-06 12:26:17      -21.900      -68.700    106.0   2.6               ANTOFAGASTA, CHILE  2026  2026-07-06      7    6  12:26:17
9 2026-07-06 12:19:16       -7.360      127.640    240.0   4.0  KEPULAUAN BARAT DAYA, INDONESIA  2026  2026-07-06      7    6  12:19:16
"""
scaler=StandardScaler()
le=LabelEncoder()

borders = [0.0 , 2.0 , 4.0 , 5.0, 6.0 , 7.0]
earthquake_categ = ["Micro","Minor","Light","Moderate","Strong"]

df['Magnitude'] = pd.cut(df['Mag.'], bins=borders, labels=earthquake_categ)

df['Region_Num'] = le.fit_transform(df['Region'])


classes = le.classes_
dicti = dict(zip(classes, range(len(classes))))

features=["Lat.degrees","Lon.degrees","Depthkm","Region_Num"]

x=df[features]
y = df["Magnitude"].cat.remove_unused_categories()

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

dtree=DecisionTreeClassifier()
dtree=dtree.fit(x_train,y_train)

y_predict=dtree.predict(x_test)

plt.figure(figsize=(15,9))
plot_tree(dtree,feature_names=features)
plt.savefig("images/decision_tree_plot.png")


labels = dtree.classes_

c_matrix = metrics.confusion_matrix(
    y_test,
    y_predict,
    labels=labels
)

cm = metrics.ConfusionMatrixDisplay(
    confusion_matrix=c_matrix,
    display_labels=labels
)
cm.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_plot.png")
plt.show()

# ACCURACY SCORE
#-----------------------------------
print(f"Accuracy: {metrics.accuracy_score(y_test, y_predict)}")