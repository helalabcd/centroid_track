from aim import Run
import aim

import pickle

aim_repo = aim.Repo.from_path(".")
query = 'run.experiment == "labelscale_frozen" and run.epochs == 1000'
df = aim_repo.query_metrics(query).dataframe()


print(df)


pickle.dump( df , open( "data_df.p", "wb" ) )

