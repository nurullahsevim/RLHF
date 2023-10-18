from AgentSetup import Student
from AgentSetup import school
import pandas as pd
import matplotlib.pyplot as plt
import random
import re
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = school(50, 12, 12)
    for i in range(168):
        model.step()
        all_agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
        data_at_step_0 = all_agent_data[all_agent_data["Step"] == i]
        print(data_at_step_0)
        info_array = data_at_step_0['info'].to_numpy()
        unique_values, frequencies = np.unique(info_array, return_counts=True)
        frequency_dict = dict(zip(unique_values, frequencies))
        print(frequency_dict)
        # pdb.set_trace()

    # all_agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
    # print(all_agent_data.head())
