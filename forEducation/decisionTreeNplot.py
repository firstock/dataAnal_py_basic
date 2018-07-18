# coding: utf-8
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer= load_breast_cancer()
X_train, X_test, y_train, y_test= train_test_split(cancer.data, cancer.target, stratify= cancer.target,
                                                  random_state= 42)
tree= DecisionTreeClassifier(random_state= 0)
tree.fit(X_train, y_train)
print('train정확도%8.3f'%tree.score(X_train, y_train))
print('test 정확도{:8.3f}'.format(tree.score(X_test, y_test)))
def dtClassifyShow(max_depth1= None):
    tree= DecisionTreeClassifier(random_state= 0, max_depth= max_depth1)
    tree.fit(X_train, y_train)
    accu_train= round(tree.score(X_train, y_train), 5)
    accu_test= round(tree.score(X_test, y_test), 5)
    
    return accu_train, accu_test
dtClassifyShow()
dtClassifyShow(4)
trainPitch= 0
testMax= 0
iMax= 0
for i in range(1,19):
    accTr, accTs= dtClassifyShow(i)
    if testMax <= accTs:
        trainPitch= accTr
        testMax= accTs
        iMax= i

trainPitch, testMax, iMax
dtClassifyShow(5)
tree= DecisionTreeClassifier(random_state= 0, max_depth= 5)
tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file= "data/tree.dot", class_names=["malignant","benign"],
               feature_names= cancer.feature_names,
               impurity=False, filled= True)
import graphviz
with open("data/tree.dot", encoding='utf-8') as f:
    dot_graph= f.read()
dot= graphviz.Source(dot_graph)#, encoding= 'cp949'
# dot2= graphviz.Graph(dot_graph, node_attr= "gulim")#, edge_attr="'fontname='gulim.ttc'")
display(dot)
#!mkdir result
# %ls
dot.format= 'png'
dot.render(filename= 'result/treeSample')
