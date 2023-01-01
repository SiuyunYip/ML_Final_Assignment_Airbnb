import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


listings = pd.read_csv('data/listings.csv', index_col=0, sep=',')
# Plot Rating Scores Distribution Plot
hist_kws = {"alpha": 0.3}
plt.figure(figsize=(20, 10))
plt.xticks(np.arange(0, 6, step=1))
sns.distplot(listings['review_scores_rating'], hist_kws=hist_kws)
plt.show()

