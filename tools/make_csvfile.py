import pandas as pd
import numpy as np

def make_csv(caption_names, audio_names_, top10_index, csv_output_dir):
    df_rank=pd.DataFrame([audio_names_[i.astype(int)] for i in top10_index],index=caption_names)
    df_rank.columns=["fname_1","fname_2","fname_3","fname_4","fname_5","fname_6","fname_7","fname_8","fname_9","fname_10"]
    df_rank.to_csv('../'+csv_output_dir.joinpath('results.csv'),index=True)


