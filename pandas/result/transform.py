# coding: utf-8
people = pd.DataFrame(np.random.randn(5, 5),
                      columns=['a', 'b', 'c', 'd', 'e'],
                      index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.iloc[2:3, [1, 2]] = np.nan # Add a few NA values
people
key= ['one','two','one','two','one']
people.groupby(key).mean()
people.groupby(key).transform(np.mean)
def demean(arr):
    return arr-arr.mean()
demeaned= people.groupby(key).transform(demean)
demeaned
demeaned.groupby(key).mean()
