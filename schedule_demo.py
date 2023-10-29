import random
import matplotlib.pyplot as plt
import numpy as np


def generate_schedule(num_courses, num_classrooms):
    # Days of the week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Dictionary to store the schedule
    schedule = {day: {} for day in days}

    # List of classrooms
    classrooms = [f'Classroom {i + 1}' for i in range(num_classrooms)]

    # Time slots (assuming 1-hour sessions and a 9 AM - 5 PM class day)
    time_slots = [f"{hour}:00" for hour in range(9, 17)]

    # For each course, randomly assign class hours
    for i in range(1, num_courses + 1):
        course_name = f'Course {i}'
        course_hours = random.randint(2, 6)

        # Distribute class hours across the days and time slots
        while course_hours > 0:
            day = random.choice(days)
            time_slot = random.choice(time_slots)
            classroom = random.choice(classrooms)

            if time_slot not in schedule[day]:
                schedule[day][time_slot] = {}

            if classroom not in schedule[day][time_slot]:
                schedule[day][time_slot][classroom] = course_name
                course_hours -= 1

    # Sort the schedule for better readability
    for day, time_slots in schedule.items():
        for time_slot, classrooms in sorted(time_slots.items()):
            schedule[day][time_slot] = {classroom: course for classroom, course in sorted(classrooms.items())}

    return schedule

if __name__ == '__main__':



    num_courses = 16
    num_classrooms = 8
    schedule = generate_schedule(num_courses, num_classrooms)
    # Days of the week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    # Time slots
    time_slots = [f"{hour}:00" for hour in range(9, 17)]

    # Initialize a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a table with rows for each time slot and columns for each day
    cell_text = []
    for time_slot in time_slots:
        row = []
        for day in days:
            if time_slot in schedule[day]:
                classrooms = schedule[day][time_slot]
                text = '\n'.join([f'{room}: {course}' for room, course in classrooms.items()])
            else:
                text = ''
            row.append(text)
        cell_text.append(row)

    # Add a table to the plot
    table = plt.table(cellText=cell_text,
                      rowLabels=time_slots,
                      colLabels=[''] + days,
                      colWidths=[0.2] * (len(days) + 1),
                      loc='center',
                      cellLoc='center')

    # Hide axes
    ax.axis('off')

    # Show the plot
    plt.show()