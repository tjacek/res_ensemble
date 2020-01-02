from collections import defaultdict

def make_balanced(y):
    by_cat=defaultdict(lambda :[])
    for i,y_i in enumerate(y):
        by_cat[y_i].append(i)
    return by_cat	

y=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
print(make_balanced(y))