import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from math import gcd

# Task class to hold the period and cost of each task
class Task:
    def __init__(self, period, cost, deadline=None):
        self.period = period
        self.cost = cost
        self.deadline = deadline if deadline else period
        self.id = None  # Will be assigned later automatically

# Function to calculate processor utilization for RMS, EDF, LLF, and DMS
def processor_utilization(tasks):
    utilization = sum([task.cost / task.period for task in tasks])
    return utilization

# RMS Schedulability Test (Utilization-based Test)
def rms_schedulability(tasks):
    utilization = processor_utilization(tasks)
    if utilization > len(tasks) * (2 ** (1 / len(tasks)) - 1):
        return rms_exact_analysis(tasks)
    return True

# EDF Schedulability Test (Utilization-based Test)
def edf_schedulability(tasks):
    utilization = processor_utilization(tasks)
    return utilization <= 1

# LLF Schedulability Test (Utilization-based Test)
def llf_schedulability(tasks):
    utilization = processor_utilization(tasks)
    return utilization <= 1

# DMS Schedulability Test (Utilization-based Test)
def dms_schedulability(tasks):
    utilization = processor_utilization(tasks)
    if utilization > len(tasks) * (2 ** (1 / len(tasks)) - 1):
        return dms_exact_analysis(tasks)
    return True

# RMS Exact Analysis (Exact Test)
def rms_exact_analysis(tasks):
    # Compute processor utilization
    utilization = processor_utilization(tasks)

    # Utilization-based test for RMS
    utilization_bound = len(tasks) * (2 ** (1 / len(tasks)) - 1)
    if utilization > utilization_bound:
        return False  # If utilization exceeds bound, tasks are not schedulable by RMS

    # Exact response-time analysis
    tasks = sorted(tasks, key=lambda x: x.period)  # Sort tasks by period (shortest first)

    for i, task in enumerate(tasks):
        response_time = task.cost
        while True:
            # Calculate the total interference from higher-priority tasks
            interference = sum(
                np.ceil(response_time / higher_task.period) * higher_task.cost
                for higher_task in tasks[:i]
            )
            new_response_time = task.cost + interference

            # If response time exceeds task's period, it's not schedulable
            if new_response_time > task.period:
                return False

            # If response time converges, move to the next task
            if new_response_time == response_time:
                break

            response_time = new_response_time

    return True  # All tasks passed the exact analysis

# DMS Exact Analysis (Exact Test)
def dms_exact_analysis(tasks):
    # Sort tasks by their deadlines (shortest deadline = highest priority)
    tasks = sorted(tasks, key=lambda x: x.deadline)

    # Perform utilization-based schedulability test for each task
    for i in range(len(tasks)):
        # Current task under analysis
        task = tasks[i]

        # Calculate the total utilization of all higher-priority tasks
        total_utilization = sum(
            [tasks[j].cost / min(tasks[j].deadline, task.deadline) for j in range(i)]
        )

        # Add the utilization of the current task
        total_utilization += task.cost / task.deadline

        # Check if the total utilization exceeds 1
        if total_utilization > 1:
            return False  # Task set is not schedulable under DMS

    return True  # All tasks pass the schedulability test

# Helper function to calculate LCM of two numbers
def lcm(a, b):
    return abs(a * b) // gcd(a, b)

# Helper function to calculate LCM of multiple numbers
def lcm_of_periods(periods):
    return reduce(lcm, periods)

# Function to generate the Rate Monotonic Schedule
def RMS(tasks):
     # Sort tasks by period (shorter period = higher priority)
    tasks = sorted(tasks, key=lambda x: x.period)

    # Compute LCM of all task periods to determine the simulation duration
    periods = [task.period for task in tasks]
    simulation_time = lcm_of_periods(periods)

    # Initialize task states and schedule
    task_states = {
        task: {
            "remaining_time": task.cost,
            "next_period_start": 0,
            "deadline": task.deadline
        }
        for task in tasks
    }

    # Track all tasks that are ready to execute (highest priority tasks)
    ready_tasks = []
    schedule = []
    t = 0  # Start time

    for t in range(simulation_time):
        # Check for any tasks that have recently come up
        for task in tasks:
            if t >= task_states[task]["next_period_start"]:
                task_states[task]["remaining_time"] = task.cost
                task_states[task]["next_period_start"] += task.period
                ready_tasks.append(task)

        # Sort tasks by priority (shortest period first)
        # Ties broken by shortest reamining time left
        ready_tasks.sort(key=lambda task: (task.period, task.deadline, task.cost))

        # If there are tasks ready to run, schedule the highest priority task
        if ready_tasks:
            # Set the highest priority task (shortest period)
            highest_priority_task = ready_tasks[0]  
            
            # Execute the task for its entire cost
            schedule.extend([f"Task {highest_priority_task.id}"])

            task_states[highest_priority_task]["remaining_time"] -= 1

            if task_states[highest_priority_task]["remaining_time"] <= 0:
                ready_tasks.pop(0)
        else:
            # If no tasks are ready, mark CPU as idle
            schedule.append("Idle")

        previous_t = t

    return schedule

# Function to generate the Earliest Deadline First schedule
def EDF(tasks):
    # Compute LCM of all task periods to determine the simulation duration
    periods = [task.period for task in tasks]
    simulation_time = lcm_of_periods(periods)

    # Initialize task states and schedule
    task_states = {
        task: {
            "remaining_time": task.cost,
            "next_period_start": 0,
            "deadline": task.deadline,
            "last_run": 0
        }
        for task in tasks
    }
    schedule = []

    # Track all tasks that are ready to execute (highest priority tasks based on earliest deadline)
    ready_tasks = []
    t = 0  # Start time

    for t in range(simulation_time):
        # Check for any tasks that have recently come up
        for task in tasks:
            # If task period starts at or before the current time step, add it to the ready tasks
            if t >= task_states[task]["next_period_start"]:
                task_states[task]["remaining_time"] = task.cost  # Reset the task's remaining time
                task_states[task]["next_period_start"] += task.period  # Update next period start
                task_states[task]["deadline"] = t + task.deadline  # Update the task's deadline
                ready_tasks.append(task)

        # Sort tasks by earliest deadline first
        # Ties broken by shortest remaining time left
        ready_tasks.sort(key=lambda task: (task_states[task]["deadline"], 
                                           task_states[task]["remaining_time"], 
                                           task.period, t-task_states[task]["last_run"]))

        # If there are tasks ready to run, schedule the task with the earliest deadline
        if ready_tasks:
            earliest_deadline_task = ready_tasks[0]  # Peek at the task with the earliest deadline
            
            # Execute the task for its entire cost
            schedule.extend([f"Task {earliest_deadline_task.id}"])

            # After executing the task, we update its state (it has completed its current period)
            task_states[earliest_deadline_task]["remaining_time"] -= 1
            task_states[earliest_deadline_task]["last_run"] = t

            if task_states[earliest_deadline_task]["remaining_time"] <= 0:
                ready_tasks.pop(0)
        else:
            # If no tasks are ready, mark CPU as idle
            schedule.append("Idle")

    return schedule

# Function to generate the Least Laxity First schedule
def LLF(tasks):
    # Compute LCM of all task periods to determine the simulation duration
    periods = [task.period for task in tasks]
    simulation_time = lcm_of_periods(periods)

    # Initialize task states and schedule
    task_states = {
        task: {
            "remaining_time": task.cost,
            "next_period_start": 0,
            "deadline": task.deadline,
            "laxity": task.deadline - task.cost,
            "last_run": 0
        }
        for task in tasks
    }
    schedule = []

    # Track all tasks that are ready to execute
    ready_tasks = []
    t = 0  # Start time

    for t in range(simulation_time):
        # Check for any tasks that have recently come up
        for task in tasks:
            # If task period starts at or before the current time step, add it to the ready tasks
            if t >= task_states[task]["next_period_start"]:
                task_states[task]["remaining_time"] = task.cost  # Reset the task's remaining time
                task_states[task]["next_period_start"] += task.period  # Update next period start
                task_states[task]["deadline"] = t + task.deadline  # Update the task's deadline
                ready_tasks.append(task)

        # If there are tasks ready to run, prioritize by slack time (laxity)
        if ready_tasks:
            # Calculate slack time (laxity) for each ready task
            for task in ready_tasks:
                task_states[task]["laxity"] = task_states[task]["deadline"] - t - task_states[task]["remaining_time"]

            # Sort tasks by laxity (least laxity first)
            ready_tasks.sort(key=lambda task: (task_states[task]["laxity"], 
                                               task_states[task]["remaining_time"], 
                                               t - task_states[task]["last_run"]))

            # Execute the task with the least laxity
            least_laxity_task = ready_tasks[0]

            # Execute the task for its entire cost
            schedule.extend([f"Task {least_laxity_task.id}"])

            # After executing the task, we update its state (it has completed its current period)
            task_states[least_laxity_task]["remaining_time"] -= 1
            task_states[least_laxity_task]["last_run"] = t

            if task_states[least_laxity_task]["remaining_time"] <= 0:
                ready_tasks.pop(0)
        else:
            # If no tasks are ready, mark CPU as idle
            schedule.append("Idle")

    return schedule

# Function to generate the Deadline Monotonic Schedule
def DMS(tasks):
     # Sort tasks by period (shorter period = higher priority)
    tasks = sorted(tasks, key=lambda x: x.period)

    # Compute LCM of all task periods to determine the simulation duration
    periods = [task.period for task in tasks]
    simulation_time = lcm_of_periods(periods)

    # Initialize task states and schedule
    task_states = {
        task: {
            "remaining_time": task.cost,
            "next_period_start": 0,
            "deadline": task.deadline
        }
        for task in tasks
    }

    # Track all tasks that are ready to execute (highest priority tasks)
    ready_tasks = []
    schedule = []
    t = 0  # Start time

    for t in range(simulation_time):
        # Check for any tasks that have recently come up
        for task in tasks:
            if t >= task_states[task]["next_period_start"]:
                task_states[task]["remaining_time"] = task.cost
                task_states[task]["next_period_start"] += task.period
                ready_tasks.append(task)

        # Sort tasks by priority (shortest period first)
        # Ties broken by shortest reamining time left
        ready_tasks.sort(key=lambda task: (task.deadline, task.cost))

        # If there are tasks ready to run, schedule the highest priority task
        if ready_tasks:
            # Set the highest priority task (shortest period)
            highest_priority_task = ready_tasks[0]  
            
            # Execute the task for its entire cost
            schedule.extend([f"Task {highest_priority_task.id}"])

            task_states[highest_priority_task]["remaining_time"] -= 1

            if task_states[highest_priority_task]["remaining_time"] <= 0:
                ready_tasks.pop(0)
        else:
            # If no tasks are ready, mark CPU as idle
            schedule.append("Idle")

        previous_t = t

    return schedule

# Function to visualize the schedule and mark periods with triangles
def visualize_schedule(tasks, algorithm):
    if algorithm == "RMS":
        schedule = RMS(tasks)
    elif algorithm == "EDF":
        schedule = EDF(tasks)
    elif algorithm == "LLF":
        schedule = LLF(tasks)
    elif algorithm == "DMS":
        schedule = DMS(tasks)
    else:
        raise NotImplementedError(f"{algorithm} is not supported in this visualization.")

    # Compute the LCM of task periods
    periods = [task.period for task in tasks]
    total_time = lcm_of_periods(periods)  # LCM of all task periods
    times = np.arange(0, total_time, 1)

    # Visualize the schedule in matplotlib
    plt.figure(figsize=(20, 4))
    periods = [task.period for task in tasks]

    for idx, task in enumerate(tasks):
        # Find indices where the task executes
        task_indices = [i for i, x in enumerate(schedule) if x == f"Task {idx + 1}"]
        plt.bar(
            task_indices, 
            [1] * len(task_indices), 
            width=1,
            align="edge",  # Align bars to the left edge of the timestamp
            color=f"C{idx}", 
            label=f"Task {task.id}", 
            edgecolor="black"
        )

        # Mark the start of a new period with a triangle in the same color
        # Stagger the triangles if multiple tasks start at the same time
        for t in range(0, total_time, task.period):
            y_offset = 1.05 + 0.1 * idx  # Incremental vertical offset for each task
            plt.plot(
                t, 
                y_offset,  # Position based on staggered offset
                marker='v', 
                color=f'C{idx}', 
                markersize=8, 
                label=""  # No label to prevent legend duplication
            )
    
    # Add vertical grid lines to mark every unit of time
    for t in range(total_time):
        plt.axvline(x=t, 
                    color='gray', 
                    linestyle='--', 
                    linewidth=0.5, 
                    alpha=0.7
                )

    # Configure plot
    plt.xticks(np.arange(0, total_time + 1, 1))  # Add ticks at every time unit
    plt.yticks([])
    plt.grid(visible=True, axis='x', linestyle='--', alpha=0.5)  # Show only vertical gridlines
    plt.xlabel('Time')
    plt.ylabel('Task Execution')
    plt.title(f'Task Schedule ({algorithm}) up to LCM ({total_time})')
    plt.legend()
    plt.show()

class SchedulerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Task Scheduler")

        # Task input fields
        self.tasks = []
        
        # Set default values for tasks and algorithm
        self.default_periods = "12, 6, 24, 12"
        self.default_costs = "2, 1, 4, 1"
        self.default_deadlines = "8, 6, 6, 2" 
        self.default_algorithm = "RMS"

        self.period_label = tk.Label(root, text="Task Period (separated by commas): ")
        self.period_label.grid(row=0, column=0)
        self.periods = tk.Entry(root)
        self.periods.insert(0, self.default_periods)  # Set default periods
        self.periods.grid(row=0, column=1)

        self.cost_label = tk.Label(root, text="Task Cost (separated by commas): ")
        self.cost_label.grid(row=1, column=0)
        self.costs = tk.Entry(root)
        self.costs.insert(0, self.default_costs)  # Set default costs
        self.costs.grid(row=1, column=1)

        self.deadline_label = tk.Label(root, text="Task Deadline (separated by commas): ")
        self.deadline_label.grid(row=2, column=0)
        self.deadlines = tk.Entry(root)
        self.deadlines.insert(0, self.default_deadlines)  # Set default deadlines
        self.deadlines.grid(row=2, column=1)

        # Informational label for default behavior of deadlines
        self.deadline_info_label = tk.Label(
            root, 
            text="If left blank, deadlines will default to the periods. \n *Note: Deadlines will be ignored for RMS and EDF.",
            font=("Arial", 12), 
            fg="gray"
        )
        self.deadline_info_label.grid(row=3, column=0, columnspan=2)

        self.algorithm_label = tk.Label(root, text="Select Algorithm: ")
        self.algorithm_label.grid(row=4, column=0)
        self.algorithm_var = tk.StringVar()
        self.algorithm_var.set(self.default_algorithm)  # Set default algorithm to RMS
        self.algorithm_menu = tk.OptionMenu(root, self.algorithm_var, "RMS", "EDF", "LLF", "DMS", command=self.update_deadlines_state)
        self.algorithm_menu.grid(row=4, column=1)

        self.submit_button = tk.Button(root, text="Submit", command=self.submit)
        self.submit_button.grid(row=5, column=0, columnspan=2)

        # Initialize the state of the deadlines box
        self.update_deadlines_state(self.default_algorithm)

    # Updates the state of the deadlines entry box based on the selected algorithm.
    def update_deadlines_state(self, selected_algorithm):
        if selected_algorithm in ["RMS", "EDF"]:
            self.deadlines.config(state="disabled")
            #self.deadlines.delete(0, tk.END)  # Clear deadlines input
        else:
            self.deadlines.config(state="normal")

    def submit(self):
        try:
            periods = list(map(int, self.periods.get().split(',')))
            costs = list(map(int, self.costs.get().split(',')))
            deadlines_input = self.deadlines.get().split(',')

            # Dynamically determine the number of tasks
            num_tasks = len(periods)

            # Check if deadlines are empty, and if so, use periods as deadlines
            deadlines = []
            if deadlines_input[0]:  # If there is input for deadlines
                deadlines = list(map(int, deadlines_input))
            else:
                deadlines = periods  # Use periods as deadlines if no input is provided
            
            algorithm = self.algorithm_var.get()

            if len(costs) != num_tasks or (len(deadlines) != num_tasks and (algorithm == "LLF" or algorithm == "DMS")):
                raise ValueError("Number of periods, costs, and deadlines must match.")

            # Assign task IDs and create Task objects
            self.tasks = [Task(periods[i], costs[i], deadlines[i]) for i in range(num_tasks)]
            for i, task in enumerate(self.tasks):
                task.id = i + 1  # Assign IDs (starting from 1)
                if algorithm == "RMS" or algorithm == "EDF":
                    task.deadline = task.period
                if task.deadline > task.period:
                    raise ValueError("All deadlines must be less than or equal to the corresponding period")

            # Check schedulability based on selected algorithm
            if algorithm == "RMS":
                if not rms_schedulability(self.tasks):
                    messagebox.showerror("Schedulability", "Tasks are not schedulable under RMS.")
                    return
            elif algorithm == "EDF":
                if not edf_schedulability(self.tasks):
                    messagebox.showerror("Schedulability", "Tasks are not schedulable under EDF.")
                    return
            elif algorithm == "LLF":
                if not llf_schedulability(self.tasks):
                    messagebox.showerror("Schedulability", "Tasks are not schedulable under LLF.")
                    return
            elif algorithm == "DMS":
                if not dms_schedulability(self.tasks):
                    messagebox.showerror("Schedulability", "Tasks are not schedulable under DMS.")
                    return

            # If feasible, visualize schedule
            visualize_schedule(self.tasks, algorithm)
        
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = SchedulerApp(root)
    root.mainloop()