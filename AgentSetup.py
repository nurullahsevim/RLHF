import mesa
import random
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import pdb
import numpy as np
import pandas as pd


class Student(mesa.Agent):
  '''
  student follows schedule in every tick, according its assigned courses it will go to the room
  '''

  def __init__(self, unique_id, model,schedule_df):
    super().__init__(unique_id, model)
    #print(f"DEBUG: Passed schedule: {schedule_df}")  # Debug print
    self.course_schedule = schedule_df
    #print(f"DEBUG: Self schedule: {self.course_schedule}")  # Debug print

    #self.course_schedule = schedule_df
    self.assign_courses()
    self.room = 0  # No room assigned initially
    print("I am student ", unique_id, " and I take courses: ", self.courses)

  def assign_courses(self):
      # Assuming the course names are strings like 'Course 1', 'Course 2', etc.
      #all_courses = ['Course 1', 'Course 2', 'Course 3', 'Course 4', 'Course 5', 'Course 6']
      all_courses = [1,2,3,4,5,6,7,8]
      self.courses = random.sample(all_courses, 5)  # Randomly select 5 courses


  def step(self):
    current_time = self.model.schedule.time  # Assuming there's a time attribute in model's schedule
    #print(current_time)
      # Check if there's a course at this time
    for course in self.courses:
      #pdb.set_trace()
      course_times = self.course_schedule[(self.course_schedule['Course'] == course) & (self.course_schedule['Time'] == current_time)]
      #print([(self.course_schedule['Course'] == course) & (self.course_schedule['Time'] == current_time)])
      #if not course_times.empty:
      course_at_this_time = course_times[course_times['Time'] == current_time]
      if not course_at_this_time.empty:
        #print(course_at_this_time)
        #pdb.set_trace()
        self.room = course_at_this_time.iloc[0]['Room']  # Update room
        #print(self.room)
        break  # Exit the loop if a course is found at this time



class school(mesa.Model):
  '''
  A model class to manage the agents
  '''
  def __init__(self, N, width, height):
    self.num_agents = N
    self.grid = MultiGrid(width, height,False)
    self.schedule= RandomActivation(self)
    self.datacollector = DataCollector(
            agent_reporters={"info": "room"} )
    #self.student=Student()
    self.schedule_df = self.generate_schedule()
    #print(f"DEBUG: Schedule in model: {self.schedule_df}")
    for i in range(self.num_agents):
      agent = Student(i, self,self.schedule_df)
      self.schedule.add(agent)
    self.running = True
  def step(self):
        self.datacollector.collect(self)  # This collects the data each step
        self.schedule.step()
        print("time",self.schedule.time )


  def generate_schedule(self):
    # Constants
    DAYS = 5
    HOURS = 24
    WORK_START_HOUR = 9
    WORK_END_HOUR = 17  # 5 PM is the 17th hour in 24-hour time
    LECTURES_PER_COURSE = 3
    NUM_COURSES = 8
    NUM_ROOMS = 6

    # Generate data
    data = []
    for course_num in range(1, NUM_COURSES + 1):
        #course_name = f"Course {course_num}"
        course_name = course_num
        for _ in range(LECTURES_PER_COURSE):
            # Randomly select a day (0: Monday, 4: Friday)
            day = random.randint(0, DAYS - 1)

            # Randomly select a time within the working hours
            hour = random.randint(WORK_START_HOUR, WORK_END_HOUR - 1)  # Subtract 1 to end at 4 PM, which starts the last work hour

            # Convert day and hour to a single hour count from the start of the week
            time = day * HOURS + hour

            # Randomly assign a room
            #room = f"room{random.randint(1, NUM_ROOMS)}"
            room = random.randint(1, NUM_ROOMS)

            data.append((course_name, time, room))

    # Create a DataFrame
    schedule_df = pd.DataFrame(data, columns=["Course", "Time", "Room"])

    # Sorting by Course and Time for better readability
    schedule_df.sort_values(by=["Course", "Time"], inplace=True)

    # Reset index after sorting
    schedule_df.reset_index(drop=True, inplace=True)
    print(schedule_df)
    return schedule_df


