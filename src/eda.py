import pandas as pd
import numpy as np

class EDA():
    def __init__(self, data):
        import numpy as np
        import pandas as pd
        self.data = data
        self.nrows = data.shape[0]
        self.ncols = data.shape[1]

    def explore_stats(self):

        # create columns list and check dtype
        feature = []
        type_lst = []
        for key, value in self.data.dtypes.items():
            feature.append(key)
            type_lst.append(value)

        # check distinct value
        distinct = []
        for i in self.data.columns:
            num_distinct = len(self.data[i].unique())
            distinct_pct = num_distinct / self.nrows * 100
            distinct.append("{} ({:0.2f}%)".format(num_distinct, distinct_pct))

        # check null values
        null = []
        for i in self.data.columns:
            num_null = self.data[i].isna().sum()
            null_pct = num_null / self.nrows * 100
            null.append("{} ({:0.2f}%)".format(num_null, null_pct))

        # check negative values
        negative = []
        for i in self.data.columns:
            try:
                num_neg = (self.data[i].astype('float') < 0).sum()
                neg_pct = num_neg / self.nrows * 100
                negative.append("{} ({:0.2f}%)".format(num_neg, neg_pct))
            except:
                negative.append(str(0) + " (0%)")
                continue

        # check zeros
        zeros = []
        for i in self.data.columns:
            try:
                num_zero = (self.data[i] == 0).sum()
                zero_pct = num_zero / self.nrows * 100
                zeros.append("{} ({:0.2f}%)".format(num_zero, zero_pct))
            except:
                zeros.append(str(0) + " (0%)")
                continue

        # check stats measure
        stats = self.data.describe(include='all').transpose()

        # put measures into a dataframe
        data = {'feature': feature,
                'data_type': type_lst,
                'n_distinct': distinct,
                'n_missing': null,
                'n_negative': negative,
                'n_zeros': zeros}
        for y in stats.columns:
            data[y] = []
            for x in self.data.columns:
                try:
                    data[y].append(stats.loc[x, y])
                except:
                    data[y].append(0.0)

        data_stats = pd.DataFrame(data)

        return data_stats