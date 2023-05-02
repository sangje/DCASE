import pandas as pd
import numpy as np

def make_csv(caption_names, audio_names_, top5_index, csv_output_dir):
    df_rank=pd.DataFrame([audio_names_[i] for i in top5_index],index=caption_names)
    df_rank.to_csv(csv_output_dir.joinpath('results.csv',index=True))

