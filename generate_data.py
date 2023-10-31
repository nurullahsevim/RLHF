from AgentSetup import Student
from AgentSetup import school
import pandas as pd
import matplotlib.pyplot as plt
import random
import re
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset




def extract_course_number(course_id):
    # Search for a number in the course_id string
    match = re.search(r'\d+', course_id)
    if match:
        return int(match.group())  # Convert the found number to an integer
    return None  # Return None if no number is found


def convert_time(hour):
    day_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    day = day_mapping[hour // 24]
    hour_in_day = hour % 24
    return f"{day}, {hour_in_day}:00"

def generate_readable_schedule(schedule_df):
    DAYS = 5
    HOURS = 24
    WORK_START_HOUR = 9
    WORK_END_HOUR = 17  # 5 PM is the 17th hour in 24-hour time
    LECTURES_PER_COURSE = 3
    NUM_COURSES = 8
    NUM_ROOMS = 6


    course_names = [
        "Classical Archaeology",
        "Molecular Gastronomy", "Astrophysics", "Comparative Literature",
        "Organic Chemistry", "World History", "Artificial Intelligence",
        "Marine Biology", "Advanced Potions", "Quantum Mechanics"
    ]
    schedule_df = schedule_df.copy(deep=True)
    schedule_df['Course Name'] = schedule_df['Course'].apply(
        lambda x: course_names[x - 1])  # assuming courses are 1-indexed

    # Convert the 'Time' column to a more readable format
    schedule_df['Readable Time'] = schedule_df['Time'].apply(convert_time)

    schedule_df.drop(columns=["Course", "Time"], inplace=True)
    schedule_df = schedule_df.reindex(columns=["Course Name", "Readable Time", "Room"])
    return schedule_df

# Press the green button in the gutter to run the script.
def generative_dataset():

    model = school(50, 12, 12)
    schedule_df = generate_readable_schedule(model.schedule_df)
    # print(schedule_df.to_string(index=False))
    density=""
    for i in range(24*4):
        temp = ""
        model.step()
        all_agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
        data_at_step_i = all_agent_data[all_agent_data["Step"] == i]
        info_array = data_at_step_i['info'].to_numpy()
        unique_values, frequencies = np.unique(info_array, return_counts=True)
        for value,freq in zip(unique_values, frequencies):
            if value:
                temp+=(f"Room {value}: {freq}, ")
        if temp!="":
            density = density + f"{convert_time(i)}"+"\n" + temp+"\n"


    friday = np.zeros((8,24))
    for i in range(24):
        temp = ""
        model.step()
        all_agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
        data_at_step_i = all_agent_data[all_agent_data["Step"] == i]
        info_array = data_at_step_i['info'].to_numpy()
        unique_values, frequencies = np.unique(info_array, return_counts=True)
        for value,freq in zip(unique_values, frequencies):
            if value:
                friday[value-1,i] = freq

    # print(density)
    # print(friday)

    prompt_string = "Predict the number of students in each room in Friday based on the information provided below:\nCourse Schedules:\n"+schedule_df.to_string(index=False)+"\nNumber of students in each class at given time before Friday (0 if not specified):\n"+density
    # print(prompt_string)
    return {"prompt":prompt_string,"label":friday.flatten()}

def generate_data(N):
    dataset = []
    for i in range(N):
        print(f"Generating {i}")
        dataset.append(generative_dataset())
    return dataset

class MyDataset(Dataset):
    def __init__(self, N):
        self.dataset = generate_data(N)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx]["prompt"]
        label = self.dataset[idx]["label"]
        return prompt, label

if __name__ == "__main__":
    dataset = MyDataset(1)
    print(dataset[0][0])